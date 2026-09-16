#!/bin/bash
# ==============================================================================
# Dateiname: symbio.sh
# Funktion: Master-Launcher für das Symbio Nano-AI Core Framework
# ==============================================================================
# STAND 2026-09-16 (Angleichung an den Hauptzweig):
# Der alte Build-Befehl (nim c -d:danger -d:release --opt:speed) kann den
# heutigen Quelltext nicht mehr übersetzen bzw. übersetzt ihn still falsch:
#
#   --threads:on --passC:-fopenmp --passL:-fopenmp
#       Ohne diese laufen sämtliche ||-Schleifen in cortex.nim seriell.
#
#   --passL:'-Wl,-rpath,\$ORIGIN'
#       libcortex.so lädt libcortex_cuda.so per dlopen() mit blossem
#       Dateinamen. Ohne RUNPATH sucht dlopen() das cwd NICHT - CUDA fällt
#       dann STILL auf OpenCL zurück, ohne Fehlermeldung. Im Hauptzweig ist
#       genau das fünf Tage lang unbemerkt geblieben.
#       Der Backslash ist zwingend: nim ruft den Linker über /bin/sh, dort
#       expandiert ein nacktes $ORIGIN zur leeren Zeichenkette.
#         --passL:'-Wl,-rpath,\$ORIGIN'  -> RUNPATH [$ORIGIN]      RICHTIG
#         --passL:'-Wl,-rpath,$ORIGIN'   -> RUNPATH []             wirkungslos
#         --passL:'-Wl,-rpath,$$ORIGIN'  -> RUNPATH [1234ORIGIN]   PID, schlimmer
#       Deshalb wird der RUNPATH nach dem Bauen GEPRÜFT, nicht geglaubt.
#       Hinweis: binutils erzeugt DT_RUNPATH, nicht DT_RPATH - ein grep auf
#       "rpath" findet das nicht. Richtig ist grep -i path.
# ==============================================================================

COMMAND=$1

function setup_engine() {
    echo -e "\e[32m[Symbio]\e[0m Initialisiere Base Engine..."
    if ! command -v nim &> /dev/null; then
        echo -e "\e[31m Fatal: Nim Compiler nicht gefunden! Bitte installieren.\e[0m"
        exit 1
    fi

    echo -e "\e[36m  > Erstelle Verzeichnis-Struktur...\e[0m"
    mkdir -p lib
    mkdir -p factory/{experts,checkpoints,input,royal,apex,archive,tributes,royal_fragments}

    echo -e "\e[36m  > Kompiliere JIT-Core (libcortex.so) nach ./lib ...\e[0m"
    nim c -d:danger --threads:on -f \
          --passC:-fopenmp --passL:-fopenmp \
          --passL:'-Wl,-rpath,\$ORIGIN' \
          --app:lib --out:lib/libcortex.so core/cortex.nim || exit 1

    # --- Pruefung 1: RUNPATH ---
    RP=$(readelf -d lib/libcortex.so | grep -oP '(RPATH|RUNPATH).*\[\K[^]]*' || true)
    if [ "$RP" != '$ORIGIN' ]; then
        echo -e "\e[31m  ABBRUCH: RUNPATH ist '$RP', erwartet '\$ORIGIN'.\e[0m"
        echo "  Der Flag wurde von einer Shell-Ebene expandiert - siehe Kopfkommentar."
        exit 1
    fi
    echo -e "\e[36m  > RUNPATH : [\$ORIGIN]  OK\e[0m"

    # --- Pruefung 2: Pflichtsymbole ---
    # Symbolliste EINMAL einlesen. Nicht "nm | grep -q" je Symbol: grep -q
    # beendet sich beim ersten Treffer, nm bekommt SIGPIPE, und unter
    # pipefail gilt die Pipeline dann als fehlgeschlagen - der Check meldet
    # dann Symbole als fehlend, die vorhanden sind.
    SYMS=$(nm -D lib/libcortex.so)
    for sym in cortex_train_step cortex_init cortex_forward cortex_init_gpu \
               cortex_get_gpu_backend cortex_set_weight_tying cortex_set_skip_adam \
               cortex_extract_grad; do
        if ! printf '%s\n' "$SYMS" | grep -q " $sym\$"; then
            echo -e "\e[31m  ABBRUCH: Symbol $sym fehlt in lib/libcortex.so.\e[0m"
            exit 1
        fi
    done
    echo -e "\e[36m  > Symbole : $(printf '%s\n' "$SYMS" | grep -c ' T cortex_') exportiert, Pflichtsymbole vollständig  OK\e[0m"

    # --- Tensor-Hilfsbibliothek (Merge/Upscale der Destille) ---
    # --panics:on ist wegen OpenMP Pflicht.
    echo -e "\e[36m  > Kompiliere Tensor-Utils (libtensorutils.so) nach ./lib ...\e[0m"
    nim c -d:danger --panics:on --app:lib \
          --out:lib/libtensorutils.so core/snai_tensor_utils.nim || exit 1

    # --- CUDA-Backend, nur wenn nvcc vorhanden ---
    if command -v nvcc &> /dev/null; then
        echo -e "\e[36m  > Kompiliere CUDA-Backend (libcortex_cuda.so) nach ./lib ...\e[0m"
        nvcc -shared -Xcompiler -fPIC -lcublas \
             -o lib/libcortex_cuda.so core/cortex_cuda.cu || exit 1
    else
        echo -e "\e[33m  > nvcc nicht gefunden - CUDA wird übersprungen (OpenCL/CPU bleiben).\e[0m"
    fi

    echo -e "\e[32m[Symbio] System ist einsatzbereit.\e[0m\n"
}

# Auto-Setup beim allerersten Start
if [ ! -f "lib/libcortex.so" ]; then
    setup_engine
fi

# libcortex.so lädt libcortex_cuda.so per dlopen(); der RUNPATH deckt den Fall
# ab, in dem beide in lib/ liegen. Der Export ist der Gürtel zum Hosenträger
# und kostet nichts.
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH}"

case $COMMAND in
    "cli")
        # Leitet alle weiteren Argumente an die PHP CLI weiter
        php src/snai_cli.php "${@:2}"
        ;;
    "swarm")
        echo -e "\e[35m[Symbio] Starte P2P Netzwerk (Schwarm-Modus)...\e[0m"

        # Starte den Transport-Layer im Hintergrund
        php src/snai_gossip_node.php &
        NODE_PID=$!

        sleep 1

        # Starte das Daemon im Hintergrund
        php src/snai_daemon.php &
        DAEMON_PID=$!

        trap "echo -e '\n\e[31m[Symbio] Fahre Schwarm herunter...\e[0m'; kill $NODE_PID $DAEMON_PID 2>/dev/null; exit" INT TERM
        wait
        ;;
    "init")
        setup_engine
        ;;
    *)
        echo -e "\e[36m====================================================\e[0m"
        echo -e "\e[36m  SYMBIO NANO-AI CORE LAUNCHER\e[0m"
        echo -e "\e[36m====================================================\e[0m"
        echo "Verwendung:"
        echo -e "  \e[32m./symbio.sh init\e[0m   - Kompiliert die C-Engine neu"
        echo -e "  \e[32m./symbio.sh cli\e[0m    - Startet das Manuelle CLI Tool"
        echo -e "  \e[32m./symbio.sh swarm\e[0m  - Startet den P2P-Node & Autonomie-Daemon"
        echo ""
        echo "Beispiel manuelles Training:"
        echo "  ./symbio.sh cli train ./factory/input/nektar.txt ./factory/experts/drone.snai"
        echo ""
        echo "Beispiel Inferenz:"
        echo "  ./symbio.sh cli infer ./factory/experts/drone.snai \"Die Bahnstrecke\""
        echo -e "\e[36m====================================================\e[0m"
        ;;
esac
