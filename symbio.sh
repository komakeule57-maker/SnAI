#!/bin/bash
# ==============================================================================
# Dateiname: symbio.sh
# Funktion: Master-Launcher für das Symbio Nano-AI Core Framework
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
    mkdir -p factory/{experts,checkpoints,input,royal,apex,archive}
    
    echo -e "\e[36m  > Kompiliere JIT-Core (libcortex.so) nach ./lib ...\e[0m"
    nim c -d:danger -d:release --app:lib --opt:speed --out:lib/libcortex.so core/cortex.nim
    
    echo -e "\e[32m[Symbio] System ist einsatzbereit.\e[0m\n"
}

# Auto-Setup beim allerersten Start
if [ ! -f "lib/libcortex.so" ]; then
    setup_engine
fi

case $COMMAND in
    "cli")
        # Leitet alle weiteren Argumente an die PHP CLI weiter
        php src/snai_cli.php "${@:2}"
        ;;
    "swarm")
        echo -e "\e[35m[Symbio] Zünde Myzelium & Orchestrator (Schwarm-Modus)...\e[0m"
        
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
        echo -e "\e[36m====================================================\e[0m"
        ;;
esac