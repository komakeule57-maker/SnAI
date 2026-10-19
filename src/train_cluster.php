<?php
// Dateiname: src/train_cluster.php
// Funktion: Orchestriert das verteilte Training (BPE-Worker), verwaltet Epochen und verschmilzt Tensoren (OOM Zero Stream).

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Distributed Cluster Training (Phase VII.1 - Heisenbug Patch)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Orchestriert BPE-Worker für parallelisiertes Training.
 * FIX: 'average_models' lädt nichts mehr in den RAM, sondern streamt und 
 * verschmilzt die Tensoren Chunk für Chunk direkt von der SSD (Zero OOM).
 * AUDIT-FIX: Darwin-Entropie-Patch. Zirkuläres Chunk-Reading anhand des 
 * Hostnames verhindert identische Klon-Tensoren beim FedAvg über P2P!
 * AUDIT-V9.4: Event-Driven Pipe Polling (stream_select), I/O Telemetry-Spam eliminiert, Batch-Pack.
 * AUDIT-PHASE-VII.1: Exitcode-Caching (verhindert Heisenbug-Crashes), 
 * Zombie-Prozess-Terminierung und Division-by-Zero Protection.
 */

// AUDIT-FIX (2026-08-11): siehe beehive_core.php - date.timezone-Ini steht auf
// UTC statt Europe/Berlin, betrifft hier v.a. die "[H:i:s] Worker..."-Zeilen.
date_default_timezone_set('Europe/Berlin');
ini_set('memory_limit', '1024M'); // Limit extrem reduziert, wir brauchen keinen RAM mehr

// DISTRO: Wurzel ist der Distro-Ordner, nicht src/ - alle './factory/...'-
// Pfade unten haengen daran.
chdir(dirname(__DIR__));
require_once 'tensor_io.php';
require_once 'tensor_ffi_adapter.php';
require_once __DIR__ . '/tensor_utils_ffi.php';
require_once 'snai_tokenizer.php';
require_once __DIR__ . '/head_config.php';
require_once __DIR__ . '/growth_config.php';
require_once __DIR__ . '/revival_config.php';

/**
 * AUDIT-FIX: fread() liefert auf Netzwerk-Mounts (GVFS/SMB/NFS) regelmaessig
 * weniger Bytes als angefordert, ohne dass ein Fehler auftritt. Header wurden
 * dadurch als "inkompatibel" verworfen, obwohl sie identisch sind.
 */
function cluster_robust_fread($fp, int $length): string {
    $data = '';
    while (strlen($data) < $length && !feof($fp)) {
        $chunk = fread($fp, $length - strlen($data));
        if ($chunk === false || $chunk === '') break;
        $data .= $chunk;
    }
    return $data;
}

/**
 * Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
 * Punkt 3): waehrend des Watchdog-Tests entdeckt - proc_terminate() allein
 * beendet NUR den direkten Kindprozess von proc_open(). Da die Worker-
 * Kommandos hier IMMER "2>&1" enthalten (Shell-Redirection), fuehrt PHP sie
 * ueber "sh -c ..." aus - proc_terminate() traf bisher also nur die
 * Shell-Huelle, nicht den eigentlichen "php train_worker.php"-Prozess
 * darunter, der als Waise WEITERLIEF (empirisch verifiziert: "sleep 999"
 * ueberlebte einen proc_terminate() auf den Wrapper-Prozess klaglos).
 * Betraf nicht nur den neuen Watchdog unten, sondern auch die BESTEHENDE
 * Zombie-Kill-Logik beim Worker-Absturz (siehe beide Aufrufstellen unten) -
 * seit jeher. Fix: erst die direkten Kinder des Wrapper-Prozesses per
 * "pkill -P" beenden, dann den Wrapper selbst.
 */
function cluster_kill_worker_tree($process): void {
    if (!is_resource($process)) return;
    $status = proc_get_status($process);
    $wrapper_pid = $status['pid'] ?? null;
    if ($wrapper_pid) {
        @shell_exec('pkill -TERM -P ' . (int)$wrapper_pid . ' 2>/dev/null');
    }
    proc_terminate($process);
    proc_close($process);
}

/**
 * AUDIT-FIX: Die Head-Zahl liegt in einer .heads-Sidecar-Datei neben dem Modell
 * (siehe head_config.php). Beim Merge und beim finalen rename ging sie bisher
 * verloren -> resolve_num_heads() fiel danach auf den Legacy-Wert 1 zurueck und
 * deutete die trainierten Attention-Gewichte um.
 *
 * Bewusst NICHT ueber resolve_num_heads(): die Funktion schreibt bei fehlendem
 * Sidecar selbst den Legacy-Wert fest. Hier wird nur weitergereicht, was da ist.
 */
function propagate_head_sidecar(string $src_model, string $dst_model): void {
    $src = head_sidecar_path($src_model);
    if (!file_exists($src)) return;
    @copy($src, head_sidecar_path($dst_model));
}

function move_head_sidecar(string $src_model, string $dst_model): void {
    $src = head_sidecar_path($src_model);
    if (!file_exists($src)) return;
    @rename($src, head_sidecar_path($dst_model));
}

if ($argc < 6) {
    echo "Nutzung: php train_cluster.php <input.txt> <output.snai> <num_workers> <max_epochs> <base_model> [datatype] [custom_lr] [target_loss] [start_epoch] [weight_decay] [dropout_rate] [lr_schedule_epochs] [label_smoothing]\n";
    exit(1);
}

$input_file   = $argv[1];
$output_model = $argv[2];
$num_workers  = (int)$argv[3];
$max_epochs   = (int)$argv[4];
$base_model   = $argv[5];
$target_dtype = isset($argv[6]) ? (int)$argv[6] : 1;
$custom_lr    = isset($argv[7]) ? $argv[7] : "0.0005";
$target_loss  = isset($argv[8]) ? (float)$argv[8] : 0.0; // NEU: Target Loss Notbremse
// AUDIT-FIX (2026-08-07): Fortsetzung nach Absturz/Abbruch ab einem Checkpoint
// startete das LR-Schedule (Warmup+Decay, siehe Schleife unten) bisher immer
// bei Epoche 1 neu, unabhaengig davon, welchen Trainingsstand der uebergebene
// <base_model>-Checkpoint tatsaechlich hat - injiziert einen ungewollten LR-
// Warm-Restart statt nahtlos fortzusetzen. start_epoch (Default 1, bestehende
// Aufrufe unveraendert) erlaubt, die Schleife an der richtigen Stelle wieder
// aufzusetzen; max_epochs bleibt der Bezugspunkt fuer die Decay-Kurve.
$start_epoch  = isset($argv[9]) ? max(1, (int)$argv[9]) : 1;
// Optionale Regularisierung, durchgereicht an jeden Worker (siehe
// train_worker.php). Default 0.0 = No-Op, bestehende Aufrufe ohne diese
// Argumente (z.B. die automatische royal_distillery.php-Pipeline) bleiben
// unveraendert.
$weight_decay = isset($argv[10]) ? (float)$argv[10] : 0.0;
$dropout_rate = isset($argv[11]) ? (float)$argv[11] : 0.0;
// Update 2026-08-14 (Session-Entscheidung, docs/ROADMAP.md): entkoppelt die
// Warmup+Cosine-Decay-Kurve vom Schleifen-Ende DIESES Prozesses. Noetig, weil
// royal_distillery.php ab Matriarchin/Empress jetzt einzelne Epochen als
// getrennte train_cluster.php-Prozesse verkettet (jede mit frischem statt
// wiederverwendetem Nektar, siehe execute_calibration()), aber die Decay-
// Kurve trotzdem ueber den GESAMTEN verketteten Lauf glatt verlaufen soll,
// nicht bei jedem Einzelaufruf neu ab Epoche 1 anfangen. Default = $max_epochs,
// also exakt das bisherige Verhalten fuer jeden Aufrufer, der dieses Argument
// nicht mitgibt (alle bestehenden Aufrufstellen).
$lr_schedule_epochs = isset($argv[12]) ? max(1, (int)$argv[12]) : $max_epochs;
// Update 2026-08-15 (docs/ROADMAP.md, "Naechste Schritte" Punkt 7): als
// letzter positioneller Arg angehaengt statt zwischen dropout_rate und
// lr_schedule_epochs eingefuegt - sonst verschiebt sich lr_schedule_epochs'
// bereits etablierte Position 12 und es entstehen stille Off-by-one-Bugs bei
// bestehenden Aufrufstellen.
$label_smoothing = isset($argv[13]) ? (float)$argv[13] : 0.0;

if ($num_workers <= 0 || $max_epochs <= 0) {
    echo "🚨 [Fatal] num_workers und max_epochs müssen > 0 sein.\n";
    exit(1);
}

// Update 2026-08-09 (Session-Gespraech): Gesamtlaufzeit fuer die Abschluss-
// Logs (cluster_drones.log + train_<session>.jsonl-Summary unten) - bisher
// stand dort nur EIN Zeitpunkt (Fertigstellung), keine Dauer. Bewusst vor
// dem Net2Net-Merge-Block erfasst, damit auch dessen Laufzeit in die
// Gesamtzeit einfliesst, nicht nur die reine Epochenschleife.
$run_start_time = microtime(true);

$session_id = pathinfo($output_model, PATHINFO_FILENAME);

if (!file_exists($input_file)) {
    echo "[Fatal] Input-Datei $input_file fehlt.\n";
    exit(1);
}

$file_size = filesize($input_file);
// AUDIT-FIX: Division by Zero verhindern, falls die Nektar-Datei komplett leergefiltert wurde
if ($file_size <= 0) {
    echo "🚨 [Fatal] Input-Datei $input_file ist leer! (Vielleicht durch Aegis-Filter komplett gelöscht?).\n";
    exit(1);
}

// AUDIT-FIX (2026-08-16, RAM-Checkpoints fuer kleine Tiers, siehe
// ram_checkpoint_config.php): optionaler Override per Env-Var statt
// zusaetzlichem Positions-Argument - vermeidet eine argv[N]-Verschiebung bei
// den zahlreichen bestehenden Aufrufstellen (siehe die wiederholten
// Kommentare zu genau diesem Problem weiter oben, z.B. bei
// $lr_schedule_epochs/$label_smoothing). Default unveraendert, greift nur,
// wenn ein Aufrufer (beehive_core.php/royal_distillery.php) die Var explizit
// setzt.
$checkpoint_dir = rtrim(getenv('SNAI_CHECKPOINT_DIR') ?: './factory/checkpoints/', '/') . '/';
if (!is_dir($checkpoint_dir)) @mkdir($checkpoint_dir, 0777, true);

// Update 2026-08-08 (docs/ROADMAP.md, "Best-Checkpoint-Retention"): haelt
// zusaetzlich zum aktuellsten Checkpoint den mit dem bisher besten Held-out-
// Loss auf. Bewusst hier oben geladen (nicht erst in der Epochenschleife) -
// bei einem fortgesetzten Lauf (start_epoch > 1) muss der bisherige beste
// Wert aus einer FRUEHEREN Prozess-Instanz bekannt sein, sonst wuerde ein
// Wiedereinstieg einen tatsaechlich besseren alten Checkpoint stillschweigend
// mit einem schlechteren ueberschreiben.
$best_state_path = $checkpoint_dir . "best_{$session_id}.json";
$best_heldout_loss = INF;
if (file_exists($best_state_path)) {
    $best_state_raw = @json_decode(file_get_contents($best_state_path), true);
    if (is_array($best_state_raw) && isset($best_state_raw['heldout_loss'])) {
        $best_heldout_loss = (float)$best_state_raw['heldout_loss'];
        echo "  > [Best-Checkpoint] Bisheriger bester Held-out-Loss (fortgesetzter Lauf): " .
             round($best_heldout_loss, 6) . " (Epoche " . ($best_state_raw['epoch'] ?? '?') . ")\n";
    }
}

echo "=== [CLUSTER MASTER | SESSION: $session_id] INITIALISIERUNG ===\n";

// --- SMART FORGE (Die Anti-Lobotomie-Schutzschicht) ---
if ($base_model === 'none') {
    if (file_exists($output_model)) {
        echo "[Forge] Ziel-Modell existiert bereits. Nutze es als Basis (Fine-Tuning)...\n";
        $base_model = $output_model;
    } else {
        $tokenizer = new SnaiTokenizer();
        $dynamic_vocab_size = $tokenizer->getVocabSize();
        
        echo "[Forge] Kein Basismodell gefunden. Erschaffe Multi-Layer Genesis-Matrix (Vocab: $dynamic_vocab_size)...\n";
        $base_model = $checkpoint_dir . "genesis_{$session_id}.snai";
        $init_dim = 128;
        $init_cmd = sprintf("php " . escapeshellarg(__DIR__ . '/init_base.php') . " %s %d %d 512 2 2>&1", escapeshellarg($base_model), $dynamic_vocab_size, $init_dim);
        // 2026-10-19: exec statt shell_exec, damit der Exitcode ankommt -
        // file_exists() allein hielt eine abgebrochene Genesis fuer gelungen.
        $init_lines = [];
        $init_rc = 0;
        exec($init_cmd, $init_lines, $init_rc);
        echo implode("\n", $init_lines), "\n";
        if ($init_rc !== 0 || !file_exists($base_model)) {
            echo "🚨 [Fatal] Genesis-Modell konnte nicht erstellt werden (init_base.php rc=$init_rc)!\n";
            exit(1);
        }

        // Head-Sidecar SOFORT bei Geburt festschreiben (siehe head_config.php).
        // Fuer ein frisches, untrainiertes Modell gibt es nichts zu migrieren -
        // write_num_heads() direkt statt migrate_heads(), die Warnschranke ist
        // nur fuer bereits trainierte Checkpoints gedacht.
        require_once __DIR__ . '/head_config.php';
        $genesis_heads = suggest_num_heads($init_dim);
        write_num_heads($base_model, $genesis_heads);
        echo "[Forge] Head-Staffelung: $genesis_heads Heads bei dim=$init_dim (headSize=" .
             intdiv($init_dim, $genesis_heads) . ") festgeschrieben.\n";
    }
} elseif (!file_exists($base_model)) {
    echo "🚨 [Fatal] Angegebenes Basismodell $base_model fehlt.\n";
    exit(1);
}
// -----------------------------------------------------------

// 1. DATA PREP: Chunking mit Darwin-Entropie (Zirkulärer Puffer)
$min_chunk_size = 2048; 
if ($file_size < ($min_chunk_size * $num_workers)) {
    $num_workers = max(1, floor($file_size / $min_chunk_size));
}

$chunk_size = ceil($file_size / $num_workers);

// VETO-FIX: Der Darwin-Entropie-Patch!
// Wir erzeugen einen deterministischen Offset basierend auf dem PC-Namen.
// Desktop und Laptop fangen dadurch an völlig unterschiedlichen Stellen an zu lesen!
$host_hash = crc32(gethostname() . $session_id);
$entropy_offset = $host_hash % $file_size;

// AUDIT-FIX: Der rohe Byte-Offset konnte mitten in einem Wort oder sogar
// mitten in einem UTF-8-Mehrbyte-Zeichen (Umlaute!) landen - Trainingschunks
// starteten dadurch teils mit kaputten/abgeschnittenen ersten Zeichen. Wir
// schieben den Offset bis zum naechsten Whitespace vor, damit jeder Chunk an
// einer sauberen Wortgrenze beginnt (Ringpuffer-Suche, falls das Fenster ans
// Dateiende laeuft - analog zum eigentlichen Chunk-Lesen weiter unten).
$scan_window = min($file_size, 4096);
$boundary_fp = fopen($input_file, 'r');
fseek($boundary_fp, $entropy_offset);
$probe = fread($boundary_fp, $scan_window);
if (strlen($probe) < $scan_window) {
    rewind($boundary_fp);
    $probe .= fread($boundary_fp, $scan_window - strlen($probe));
}
fclose($boundary_fp);
$ws_pos = strcspn($probe, " \n\t\r");
if ($ws_pos < strlen($probe)) {
    $entropy_offset = ($entropy_offset + $ws_pos + 1) % $file_size;
}

echo "[I/O] Splitting $file_size Bytes in $num_workers Chunks\n";
echo "  > [Darwin-Patch] Host-Entropie aktiviert. Start-Offset: $entropy_offset Bytes.\n";

$fp = fopen($input_file, 'r');
fseek($fp, $entropy_offset); // Wir springen an unseren individuellen Startpunkt!

$chunk_files = [];
for ($i = 0; $i < $num_workers; $i++) {
    $chunk_data = "";
    $bytes_to_read = $chunk_size;

    while ($bytes_to_read > 0) {
        $read = fread($fp, $bytes_to_read);
        if ($read === false || strlen($read) === 0) {
            // Wenn wir am Ende der Datei sind, springen wir einfach wieder an den Anfang! (Ringpuffer)
            rewind($fp);
            $read = fread($fp, $bytes_to_read);
            if ($read === false || strlen($read) === 0) break; // Notfall-Abbruch
        }
        $chunk_data .= $read;
        $bytes_to_read -= strlen($read);
    }
    
    $chunk_name = $checkpoint_dir . "chunk_{$session_id}_{$i}.txt"; 
    file_put_contents($chunk_name, $chunk_data);
    $chunk_files[] = $chunk_name;
}
fclose($fp);

// 2. FEDERATED AVERAGING (STREAM-MODUS - OOM ZERO)
function average_models_stream(array $worker_models, string $output_path) {
    if (empty($worker_models)) {
        echo "🚨 [Fatal] Keine Worker-Outputs zum Mergen vorhanden!\n";
        exit(1);
    }
    
    $num_models = count($worker_models);
    
    $expected_size = filesize($worker_models[0]);
    foreach ($worker_models as $file) {
        if (!file_exists($file) || filesize($file) !== $expected_size) {
            echo "🚨 [Fatal] Inkonsistente oder fehlende Worker-Dateien vor dem Merge: $file\n";
            exit(1);
        }
    }

    if ($num_models === 1) {
        // AUDIT-FIX: Rueckgabewert wurde nie geprueft. Genau dieser Pfad laeuft
        // bei jeder Destillation (execute_calibration nutzt $workers = 1).
        // Schlug die Kopie fehl - auf Netzwerk-Mounts durchaus moeglich - lief
        // alles weiter, bis am Ende stumm kein Modell da war.
        if (!copy($worker_models[0], $output_path)) {
            echo "🚨 [Fatal] Konnte Worker-Modell nicht nach $output_path kopieren!\n";
            exit(1);
        }
        propagate_head_sidecar($worker_models[0], $output_path);
        propagate_growth_sidecar($worker_models[0], $output_path);
        return;
    }

    $handles = [];
    $headers = [];
    foreach ($worker_models as $file) {
        $fp = fopen($file, 'rb');
        if (!$fp) {
            echo "🚨 [Fatal] Kann Worker-Modell nicht lesen: $file\n";
            exit(1);
        }
        $handles[] = $fp;
        $headers[] = cluster_robust_fread($fp, 28); // Magic + 5 Params
    }

    if (count(array_unique($headers)) > 1) {
        echo "🚨 [Fatal] Worker-Modelle haben inkonsistente Header (Vocab/Dimension Mismatch)!\n";
        exit(1);
    }

    $out_fp = fopen($output_path, 'wb');
    if (!$out_fp) {
        foreach ($handles as $h) fclose($h);
        echo "🚨 [Fatal] Kann Output-Modell nicht schreiben: $output_path\n";
        exit(1);
    }

    fwrite($out_fp, $headers[0]);

    // AUDIT: vorher PHP-seitiges unpack()/Schleifen-Mitteln pro einzelnem
    // Float (siehe tensor_utils_ffi.php) - jetzt ein nativer Aufruf pro Chunk
    // wie schon in royal_distillery.php::merge_tributes().
    $chunk_floats = 8192;
    $chunk_bytes = $chunk_floats * 4;

    while (!feof($handles[0])) {
        $chunks = [];
        for ($i = 0; $i < $num_models; $i++) {
            $chunks[] = fread($handles[$i], $chunk_bytes);
        }

        if (strlen($chunks[0]) === 0) break;

        // AUDIT-FIX: nur volle Float-Elemente an die native Mittelung geben -
        // ungleich lange Chunks wuerden FFI::memcpy() sonst ueber das Ende
        // des kuerzeren PHP-Strings hinauslesen lassen (siehe merge_tributes()).
        $min_len = min(array_map('strlen', $chunks));
        $usable_len = $min_len - ($min_len % 4);
        if ($usable_len === 0) break;
        if ($usable_len !== strlen($chunks[0])) {
            $chunks = array_map(fn($c) => substr($c, 0, $usable_len), $chunks);
        }

        fwrite($out_fp, SnaiTensorUtils::merge($chunks));
    }

    foreach ($handles as $h) fclose($h);
    fclose($out_fp);

    // AUDIT-FIX: Head-Zahl an den Checkpoint weiterreichen. Alle Worker teilen
    // dieselbe Head-Zahl (sie starten vom selben Basismodell), daher genuegt der
    // erste als Quelle.
    propagate_head_sidecar($worker_models[0], $output_path);
    propagate_growth_sidecar($worker_models[0], $output_path);

    echo "[Merge] Multi-Layer Tensoren & Adam-Gedächtnis erfolgreich via Stream verschmolzen.\n";
}

// 3. DIE EPOCHEN-SCHLEIFE
$checkpoint_state = ""; 
$current_epoch_loss = 999.0;
$base_lr = (float)str_replace(',', '.', $custom_lr); 

// RAM-Cache für die Telemetrie
$telemetry_cache = [];
$telemetry_dir = './factory/telemetry';
if (!is_dir($telemetry_dir)) @mkdir($telemetry_dir, 0777, true);
$telemetry_file = $telemetry_dir . '/race_data.json';

// Update 2026-08-09 (Session-Gespraech): expliziter Zaehler statt den
// Endwert von $epoch nach der Schleife zurueckzurechnen - bei einem
// vorzeitigen `break` (Notbremse/Killswitch weiter unten) behaelt $epoch den
// Wert der zuletzt abgeschlossenen Epoche, bei normalem Schleifenende steht
// es dagegen schon einen Schritt WEITER (max_epochs+1) - zwei verschiedene
// Faelle, die sich aus dem Endwert allein nicht sauber unterscheiden lassen.
$epochs_completed = 0;

// Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
// Punkt 3): einmalig VOR der Epochen-Schleife definiert (nicht darin - ein
// erneutes define() derselben Konstante in Epoche 2 waere ein Fatal Error).
define('WORKER_STALL_TIMEOUT_SEC', 600);

for ($epoch = $start_epoch; $epoch <= $max_epochs; $epoch++) {
    // Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #8 "Backend-/Hardware-
    // Telemetrie"): Wall-Clock-Dauer der Epoche fuer train_<session>.jsonl unten.
    $epoch_start_time = microtime(true);

    $warmup_epochs = max(1, floor($lr_schedule_epochs * 0.15));
    $active_lr = 0.0;

    if ($epoch <= $warmup_epochs) {
        $active_lr = $base_lr * ($epoch / $warmup_epochs);
    } else {
        $progress = ($epoch - $warmup_epochs) / ($lr_schedule_epochs - $warmup_epochs);
        $active_lr = $base_lr * (0.5 * (1 + cos(pi() * $progress)));
    }
    
    $active_lr = max($active_lr, $base_lr * 0.1);
    $lr_display = number_format($active_lr, 6, '.', '');

    echo "\n>>> EPOCHE $epoch/$lr_schedule_epochs | LR: $lr_display | Starte $num_workers Worker...\n";
    
    $processes = [];
    $pipes = [];
    $worker_outputs = [];
    $epoch_losses = [];
    // Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #8/#9): pro Worker
    // der zuletzt gemeldete Wert - gleiches "letzter Stand gewinnt"-Muster wie
    // $epoch_losses oben, da bei stream_select mehrere Zeilen pro Poll ankommen
    // koennen.
    $epoch_backends = [];
    $epoch_grad_norms = [];
    // Update 2026-08-09 (Session-Gespraech): Loss der letzten
    // RECENT_LOSS_WINDOW Schritte pro Worker (siehe RECENT_LOSS= in
    // train_worker.php) - anders als $epoch_losses (Epochendurchschnitt) ein
    // Snapshot des tatsaechlichen Endstands der Epoche, direkt vergleichbar
    // mit dem Held-out-Loss (der auch immer den fertigen Epochenstand misst).
    $epoch_recent_losses = [];

    // AUDIT-FIX: Heisenbug Exitcode-Cache initialisieren
    $worker_exitcodes = [];

    // Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
    // Punkt 3): die stream_select-Schleife unten hatte bisher kein
    // Gesamt-Zeitlimit - ein HAENGENDER (nicht abgestuerzter) Worker
    // (Treiber-Deadlock, Endlosschleife) blockierte die Epoche fuer immer,
    // ohne Fehler, ohne Log. Zeitstempel des letzten Lebenszeichens pro
    // Worker, initial auf den Spawn-Zeitpunkt gesetzt (damit auch ein Worker
    // erfasst wird, der von Anfang an NIE etwas ausgibt).
    $worker_last_output_time = [];

    for ($i = 0; $i < $num_workers; $i++) {
        $worker_out = $checkpoint_dir . "worker_{$session_id}_{$i}_e{$epoch}.snai"; 
        $worker_outputs[] = $worker_out;
        // AUDIT-FIX (2026-08-07): war hart gegen Epoche 1 geprueft - bei einer
        // Fortsetzung ueber start_epoch (z.B. 11) ist $checkpoint_state in dieser
        // frischen Prozess-Instanz noch leer, das echte Basismodell steckt in
        // $base_model. Vergleich muss gegen den tatsaechlichen Startpunkt laufen.
        $start_model = ($epoch === $start_epoch) ? $base_model : $checkpoint_state;
        
        // 2026-10-19: Lernraten als %.17g statt %F (exakter Rundlauf). %F schreibt sechs
        // Nachkommastellen: eine Lernrate unter 5e-7 kam als 0.000000 an, und
        // der Worker ersetzte sie still durch seine Vorgabe 0,0005.
        $cmd = sprintf(
            "php " . escapeshellarg(__DIR__ . '/train_worker.php') . " %s %s 1 %s %d %.17g %d %.17g %F %F %F 2>&1",
            escapeshellarg($chunk_files[$i]),
            escapeshellarg($worker_out),
            escapeshellarg($start_model),
            $target_dtype,
            $active_lr,
            $epoch,
            $base_lr,
            $weight_decay,
            $dropout_rate,
            $label_smoothing
        );

        // TEMP-DEBUG: Zeigt das komplette, tatsaechlich ausgefuehrte Kommando.
        // Nach der Diagnose wieder entfernen oder auskommentieren.
        echo "[DEBUG] Worker-Kommando: $cmd\n";

        $process = proc_open($cmd, [0 => ["pipe", "r"], 1 => ["pipe", "w"], 2 => ["pipe", "w"]], $pipes_arr);
        
        if (is_resource($process)) {
            stream_set_blocking($pipes_arr[1], false);
            stream_set_blocking($pipes_arr[2], false);
            $processes[$i] = $process;
            $pipes[$i] = $pipes_arr;
            $worker_last_output_time[$i] = microtime(true);
        }
    }

    $active_workers = $num_workers;
    
    // Event-Driven Orchestration via stream_select
    while ($active_workers > 0) {
        $read_pipes = [];
        foreach ($processes as $i => $process) {
            if (is_resource($pipes[$i][1])) $read_pipes[] = $pipes[$i][1];
        }

        $write_pipes = null; $except_pipes = null;
        $num_changed = 0;
        
        if (!empty($read_pipes)) {
            $num_changed = stream_select($read_pipes, $write_pipes, $except_pipes, 0, 200000);
        }

        if ($num_changed > 0) {
            $telemetry_updated = false;
            foreach ($read_pipes as $ready_pipe) {
                $worker_id = -1;
                foreach ($pipes as $idx => $p) {
                    if ($p[1] === $ready_pipe) { $worker_id = $idx; break; }
                }
                
                if ($worker_id === -1) continue;

                $output = stream_get_contents($ready_pipe);
                if ($output) {
                    echo "[" . date('H:i:s') . "] " . trim($output) . "\n";
                    // Update 2026-08-09 ("Engine-/Framework-Verbesserungen",
                    // Punkt 3): Lebenszeichen fuer den Stall-Watchdog unten.
                    $worker_last_output_time[$worker_id] = microtime(true);

                    // Update 2026-08-09 (Nachruest-Paket #8/#9): letzter Stand
                    // reicht - beide Werte aendern sich innerhalb einer Epoche
                    // kaum/gar nicht (Backend steht beim Start fest, GradNorm
                    // schwankt nur leicht), kein Grund fuer eine eigene
                    // Match-Iteration wie beim Loss oben.
                    if (preg_match('/GPU_BACKEND=(\w+)/', $output, $bm)) {
                        $epoch_backends[$worker_id] = $bm[1];
                    }
                    if (preg_match_all('/GradNorm:\s*([0-9.]+)/', $output, $gm)) {
                        $epoch_grad_norms[$worker_id] = (float)end($gm[1]);
                    }
                    // RECENT_LOSS= wird genau EINMAL pro Worker/Epoche ausgegeben
                    // (am Ende der Mini-Batch-Schleife) - kein "letzter Treffer
                    // gewinnt" noetig, aber schadet nicht, falls die Zeile ueber
                    // zwei Polls gesplittet an derselben Stelle nochmal matcht.
                    if (preg_match('/RECENT_LOSS=([0-9.]+)/', $output, $rm)) {
                        $epoch_recent_losses[$worker_id] = (float)$rm[1];
                    }

                    if (preg_match_all('/Step\s+(\d+)\s*\|\s*Avg Loss:\s*([0-9\.]+)/', $output, $matches)) {
                        $last_idx = count($matches[0]) - 1;
                        $step = (int)$matches[1][$last_idx];
                        $avg_loss = (float)$matches[2][$last_idx];
                        $epoch_losses[$worker_id] = $avg_loss;

                        $status = proc_get_status($processes[$worker_id]);

                        // AUDIT-FIX: Sichere den Exitcode sofort, wenn der Worker genau JETZT gestorben ist
                        if (!$status['running'] && !isset($worker_exitcodes[$worker_id])) {
                            $worker_exitcodes[$worker_id] = $status['exitcode'];
                        }

                        $telemetry_cache["Worker_" . $worker_id] = [
                            'loss' => round($avg_loss, 5),
                            'step' => $step,
                            'epoch' => $epoch,
                            'pid' => $status['pid'] ?? 0
                        ];
                        $telemetry_updated = true;
                    } elseif (preg_match('/Avg Loss:\s*([0-9\.]+)/', $output, $matches)) {
                        $epoch_losses[$worker_id] = (float)$matches[1];
                    }
                }
            }
            
            if ($telemetry_updated) {
                file_put_contents($telemetry_file, json_encode($telemetry_cache, JSON_PRETTY_PRINT), LOCK_EX);
            }
        }
        
        // Status der Prozesse prüfen & Bereinigen
        foreach ($processes as $i => $process) {
            if (!is_resource($process)) continue;
            
            $status = proc_get_status($process);
            if (!$status['running']) {
                // AUDIT-FIX: Caching des echten Exitcodes, um den -1 Bug von proc_get_status zu umgehen
                $exitcode = $worker_exitcodes[$i] ?? $status['exitcode'];
                if (!isset($worker_exitcodes[$i])) $worker_exitcodes[$i] = $exitcode;
                
                $final_out = stream_get_contents($pipes[$i][1]);
                if ($final_out) {
                    echo trim($final_out) . "\n";

                    // AUDIT-FIX: Diese Rest-Ausgabe wurde bisher nur ausgegeben,
                    // aber nie geparst. Ein schneller Worker, dessen komplette
                    // Ausgabe erst nach Prozessende ankommt, lieferte dadurch
                    // NIE einen Loss -> Epochen-Loss fiel auf den 999er-Default.
                    if (preg_match_all('/Avg Loss:\s*([0-9\.]+)/', $final_out, $fm)) {
                        $epoch_losses[$i] = (float)$fm[1][count($fm[1]) - 1];
                    }
                    // Update 2026-08-09 (Nachruest-Paket #8/#9): gleiche
                    // Nachhol-Logik wie beim Loss oben, falls Backend-/GradNorm-
                    // Zeilen erst nach Prozessende in den Pipes ankommen.
                    if (!isset($epoch_backends[$i]) && preg_match('/GPU_BACKEND=(\w+)/', $final_out, $fbm)) {
                        $epoch_backends[$i] = $fbm[1];
                    }
                    if (preg_match_all('/GradNorm:\s*([0-9.]+)/', $final_out, $fgm)) {
                        $epoch_grad_norms[$i] = (float)end($fgm[1]);
                    }
                    if (!isset($epoch_recent_losses[$i]) && preg_match('/RECENT_LOSS=([0-9.]+)/', $final_out, $frm)) {
                        $epoch_recent_losses[$i] = (float)$frm[1];
                    }
                }

                foreach ($pipes[$i] as $pipe) {
                    if (is_resource($pipe)) fclose($pipe);
                }
                
                proc_close($process); 
                unset($processes[$i]);
                $active_workers--;

                if ($exitcode !== 0) {
                    echo "🚨 [Cluster] Fatal: Worker $i ist abgestürzt (Exit-Code: $exitcode). Breche Epoche sofort ab.\n";
                    
                    // AUDIT-FIX: Zombie-Prozesse der übrigen Worker hart töten
                    // (cluster_kill_worker_tree() statt proc_terminate() allein
                    // - siehe Funktionskommentar oben, Session-Gespraech 2026-08-09)
                    foreach ($processes as $j => $other_process) {
                        if (is_resource($other_process)) {
                            echo "  > Terminiere verwaisten Zombie-Worker $j...\n";
                            cluster_kill_worker_tree($other_process);
                        }
                    }
                    exit(1);
                }
            }
        }

        // Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
        // Punkt 3): Stall-Watchdog. Ein haengender (nicht abgestuerzter) Worker
        // (Treiber-Deadlock, Endlosschleife, blockierender FFI-Aufruf) wird von
        // der obigen Bereinigung nie erfasst - proc_get_status() meldet
        // weiterhin "running", stream_select() liefert einfach nie wieder
        // etwas. Ohne diese Pruefung wuerde die Epoche fuer immer blockieren,
        // ohne jeden Fehler. Gleiche Reaktion wie bei einem echten Absturz
        // (alle Worker terminieren, Fatal-Log, exit(1)) - kein automatischer
        // Neustart, da die Ursache (z.B. GPU-Treiber-Deadlock) vermutlich
        // fortbesteht und ein Automatik-Retry nur denselben Hang wiederholen
        // wuerde.
        foreach ($processes as $i => $process) {
            if (!is_resource($process)) continue;
            // AUDIT-FIX (Projekt-Review 2026-08-09): Fallback war "?? 0" -
            // ein fehlender Zeitstempel haette damit ein Idle von ~1,7 Mrd.
            // Sekunden (Unix-Epoch) ergeben und SOFORT den Notaus ausgeloest,
            // obwohl gar kein Stall vorliegt. Aktuell nicht erreichbar
            // ($worker_last_output_time und $processes werden gemeinsam
            // gesetzt), aber der sichere Default ist "jetzt" = kein Stall,
            // nicht "vor 55 Jahren" = sofort toeten.
            $idle_sec = microtime(true) - ($worker_last_output_time[$i] ?? microtime(true));
            if ($idle_sec > WORKER_STALL_TIMEOUT_SEC) {
                echo "🚨 [Cluster] Fatal: Worker $i haengt (" . round($idle_sec) . "s ohne Lebenszeichen, "
                   . "Limit " . WORKER_STALL_TIMEOUT_SEC . "s). Breche Epoche sofort ab.\n";

                foreach ($processes as $j => $other_process) {
                    if (is_resource($other_process)) {
                        echo "  > Terminiere Worker $j (Stall-Watchdog)...\n";
                        cluster_kill_worker_tree($other_process);
                    }
                }
                exit(1);
            }
        }
    }

    $checkpoint_state = $checkpoint_dir . "cp_{$session_id}_e{$epoch}.snai";
    average_models_stream($worker_outputs, $checkpoint_state); 
    
    foreach ($worker_outputs as $wo) {
        @unlink($wo);
        @unlink(head_sidecar_path($wo)); // AUDIT-FIX: Sidecar mit entfernen
        @unlink(growth_sidecar_path($wo));
    }

    // Update 2026-08-14 (Neuron-Revival, siehe diagnose/snai_neuron_revival.php):
    // per Config-Flag (revival_config.json) ein-/ausschaltbarer Hook - scannt
    // den frisch geschriebenen Epochen-Checkpoint auf tote Neuronen und
    // ersetzt deren Zeilen durch frisches Genesis-Skala-Rauschen (inkl.
    // Adam-m/v-Reset fuer genau diese Zeilen). Laeuft SYNCHRON im selben
    // Prozess direkt nach average_models_stream() - kein externer, parallel
    // laufender Prozess, der den Checkpoint gleichzeitig lesen und
    // ueberschreiben koennte (das war der ursprüngliche Grund, warum ein
    // automatischer Hook zurueckgestellt wurde: load_model() nimmt beim
    // Lesen kein flock()). Additiv wie der Held-out-Eval-Block direkt
    // darunter: ein Fehlschlag hier bricht den Trainingslauf NIE ab, nur
    // eine Warnung, der unrevivte Checkpoint bleibt unangetastet bestehen.
    // Schreibt in eine temporaere Datei und tauscht sie erst per rename()
    // (atomar) ein - der Checkpoint existiert also nie in einem halb
    // geschriebenen Zustand.
    $revival_cfg = load_revival_config();
    if ($revival_cfg['enabled']) {
        $revival_tmp = $checkpoint_state . '.revival_tmp';
        $revival_log = $checkpoint_dir . "revival_log_{$session_id}.json";
        $revival_cmd = sprintf(
            'php %s %s %s --tensors=%s --scale=%F --log=%s --cooldown=%d 2>&1',
            escapeshellarg(__DIR__ . '/diagnose/snai_neuron_revival.php'),
            escapeshellarg($checkpoint_state),
            escapeshellarg($revival_tmp),
            escapeshellarg($revival_cfg['tensors']),
            $revival_cfg['scale'],
            escapeshellarg($revival_log),
            $revival_cfg['cooldown']
        );
        $revival_output = @shell_exec($revival_cmd);
        $revival_ok = $revival_output !== null
            && file_exists($revival_tmp)
            && filesize($revival_tmp) === filesize($checkpoint_state);

        if ($revival_ok && rename($revival_tmp, $checkpoint_state)) {
            @unlink($revival_tmp . '.heads'); // Checkpoint behaelt seine eigene, unveraenderte .heads-Sidecar
            if (preg_match('/Gesamt:\s*(\d+)\s*\/\s*(\d+)\s*Neuronen revivt/', $revival_output, $rm) && (int)$rm[1] > 0) {
                echo "  > [Neuron-Revival] {$rm[1]} tote Neuronen revivt (von {$rm[2]} geprueft, Skala ±{$revival_cfg['scale']}).\n";
            }
        } else {
            echo "⚠️ [Neuron-Revival] übersprungen/fehlgeschlagen (Trainingslauf läuft unbeeinflusst mit dem unrevivten Checkpoint weiter):\n";
            if ($revival_output !== null) echo "  " . trim($revival_output) . "\n";
            @unlink($revival_tmp);
            @unlink($revival_tmp . '.heads');
        }
    }

    // AUDIT-FIX: Kein einziger gemeldeter Loss bedeutet, dass KEIN Worker auch nur
    // einen Trainingsschritt ausgefuehrt hat. Bisher wurde daraus der Platzhalter
    // 999.0, der als echter Messwert weitergereicht und in den Drohnennamen
    // geschrieben wurde (_L999.0000) - ein untrainierter Klon sah damit aus wie
    // ein regulaerer Experte. Das ist ein Fehlschlag, kein Messwert.
    if (count($epoch_losses) === 0) {
        echo "🚨 [Fatal] Epoche $epoch hat keinen einzigen Loss gemeldet - es wurde nicht trainiert.\n";
        echo "  > Haeufigste Ursache: Chunk enthaelt zu wenige Tokens fuer einen Batch.\n";
        exit(1);
    }

    $current_epoch_loss = array_sum($epoch_losses) / count($epoch_losses);
    echo "[Metrik] Globaler Avg Loss: " . round($current_epoch_loss, 5) . "\n";

    // Update 2026-08-08 (docs/ROADMAP.md, "Held-out-Set + periodische Eval"):
    // reiner Forward-Pass-Loss auf einem separaten, nie trainierten Korpus
    // (factory/holdout/) - opt-in (laeuft nur, wenn der Ordner tatsaechlich
    // Dateien enthaelt, kein Pflicht-Setup fuer bestehende Aufrufer/Laeufe),
    // rein additiv, und NIE fatal fuer den eigentlichen Trainingslauf - ein
    // Fehlschlag hier wird nur als Warnung geloggt, bricht die Epochenschleife
    // nicht ab. eval_held_out.php schreibt sein eigenes strukturiertes Log
    // (factory/checkpoints/heldout_<session>.jsonl), hier nur eine kurze
    // STDOUT-Zeile neben dem Trainingsloss.
    // Explizit pro Epoche zurueckgesetzt (nicht nur implizit "unset"): sonst
    // wuerde eine Epoche ohne Held-out-Lauf (fehlender Ordner) oder mit
    // fehlgeschlagenem Parse stillschweigend den Wert der VORHERIGEN Epoche in
    // das gleich folgende train_<session>.jsonl uebernehmen.
    $heldout_loss_val = null;
    // Update 2026-08-09 (Session-Gespraech): eval_held_out.php gibt seit
    // Nachruest-Paket #11 auch HELDOUT_PPL= aus (siehe eval_held_out.php),
    // landete bisher aber nur in heldout_<session>.jsonl - im Konsolen-/
    // Cluster-Log tauchte nie mehr als der rohe Loss auf. Kann "INF" sein
    // (is_finite()-Guard in eval_held_out.php bei entartetem Loss), deshalb
    // Zahl ODER "INF" im Regex.
    $heldout_ppl_val = null;
    $holdout_probe = glob('./factory/holdout/*.txt');
    if ($holdout_probe !== false && count($holdout_probe) > 0) {
        $heldout_cmd = 'php ' . escapeshellarg(__DIR__ . '/eval_held_out.php') . ' ' . escapeshellarg($checkpoint_state) . ' 2>&1';
        $heldout_output = @shell_exec($heldout_cmd);
        if ($heldout_output !== null && preg_match('/HELDOUT_LOSS=([0-9.]+)/', $heldout_output, $hm)) {
            $heldout_loss_val = (float)$hm[1];
            $ppl_display = 'n/a';
            if (preg_match('/HELDOUT_PPL=([0-9.]+|INF)/', $heldout_output, $pm)) {
                $ppl_display = $pm[1];
                $heldout_ppl_val = is_numeric($pm[1]) ? (float)$pm[1] : null;
            }
            echo "[Metrik] Held-out Loss: {$hm[1]} | Perplexity: {$ppl_display}\n";

            // Update 2026-08-08 (docs/ROADMAP.md, "Best-Checkpoint-Retention"):
            // zusaetzlich zum aktuellen Epochen-Checkpoint (bleibt wie bisher
            // Startmodell der naechsten Epoche) eine STABIL benannte Kopie
            // best_{session}.snai halten, sobald ein neuer bester Held-out-
            // Loss erreicht wird - ohne Early Stopping gaebe es sonst nichts,
            // worauf man zurueckfallen koennte, falls ein spaeterer Lauf
            // schlechter wird. Platzbedarf skaliert stark mit dem Tier (siehe
            // Tabelle im Nachruest-Paket-Abschnitt) - deshalb ein Freiplatz-
            // Check VOR dem Kopieren statt eines pauschalen Tier-Schwellwerts:
            // reduziert sich bei kleinen Tiers (immer genug Platz) automatisch
            // auf "immer halten", bei grossen Tiers auf "nur den Wert loggen,
            // wenn's eng wird" - exakt die in der Roadmap vorgesehene
            // Ausweichoption, ohne eine zweite, separate Codepfad-Verzweigung.
            //
            // ================== WARNUNG (2026-08-30) ==========================
            // "best" IST HIER EIN EIN-FENSTER-ARGMAX, KEINE AUSWAHL.
            // $heldout_loss_val kommt oben aus eval_held_out.php, und das ist
            // absichtlich auf max_tokens=1000 gekappt und zusaetzlich auf
            // ctx_size beschnitten - also EIN Fenster. Der Held-out-Korpus hat
            // ~632k Token; entschieden wird auf rund 0,2 % davon.
            //
            // Im LoRA-Fork ist genau diese Auswahl nachweislich GEKIPPT, nicht
            // nur ungenau: fuer ab_B_r16 waehlte der Argmax Epoche 1 mit
            // 5,7515 - auf dem vollen Korpus der SCHLECHTESTE verfuegbare
            // Stand (E10: 5,4077). Deren Fazit: "keine Auswahl, sondern ein
            // Zufallsgriff" (PLAN_LORA.md, Abschnitt 4a).
            //
            // Fuer diesen Zweig ist das bislang folgenlos - nachgeprueft am
            // 2026-08-30: weder royal_distillery.php noch beehive_core.php
            // noch snai_inference.php lesen best_*.snai, und alle Experimente
            // unter experiments/ greifen auf explizite Epochen zu. Die Datei
            // ist ein Rueckfallnetz, kein Pipeline-Eingang.
            //
            // ABER: der Dateiname behauptet etwas, das die Messung nicht
            // traegt. Wer best_<session>.snai von Hand als "den guten Stand"
            // nimmt, greift auf eine 1000-Token-Stichprobe. Fuer eine echte
            // Auswahl gehoert eval_held_out_full.php ueber die Epochenkurve -
            // siehe docs/ROADMAP.md, Update 2026-08-30 (7).
            // =================================================================
            if ($heldout_loss_val < $best_heldout_loss) {
                $best_snai = $checkpoint_dir . "best_{$session_id}.snai";
                $needed_bytes = @filesize($checkpoint_state);
                $free_bytes = @disk_free_space($checkpoint_dir);
                $has_room = ($needed_bytes !== false && $free_bytes !== false && $free_bytes > $needed_bytes * 2);

                $best_heldout_loss = $heldout_loss_val;
                $best_state = [
                    'epoch'        => $epoch,
                    'heldout_loss' => $heldout_loss_val,
                    'checkpoint'   => null,
                    'timestamp'    => date('c'),
                ];

                if ($has_room && @copy($checkpoint_state, $best_snai)) {
                    @copy(head_sidecar_path($checkpoint_state), head_sidecar_path($best_snai));
                    @copy(growth_sidecar_path($checkpoint_state), growth_sidecar_path($best_snai));
                    $best_state['checkpoint'] = basename($best_snai);
                    echo "  > [Best-Checkpoint] Neuer bester Held-out-Loss ($heldout_loss_val) -> " .
                         basename($best_snai) . " aktualisiert.\n";
                } else {
                    // Grossmodell-Ausweichoption aus der Roadmap: kein Platz (oder
                    // filesize/disk_free_space fehlgeschlagen) fuer eine zweite
                    // volle Kopie - nur der Wert wird fuer kuenftige Vergleiche
                    // und einen fortgesetzten Lauf festgehalten.
                    echo "  > [Best-Checkpoint] Neuer bester Held-out-Loss ($heldout_loss_val), aber zu wenig " .
                         "Plattenplatz fuer eine zweite Kopie - nur der Wert wird geloggt.\n";
                }
                @file_put_contents($best_state_path, json_encode($best_state));
            }
        } else {
            echo "⚠️ [Held-out-Eval] übersprungen/fehlgeschlagen (Trainingslauf läuft unbeeinflusst weiter):\n";
            if ($heldout_output !== null) echo "  " . trim($heldout_output) . "\n";
        }
    }

    // Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #7/#8/#9/#10):
    // strukturiertes Pendant zu heldout_<session>.jsonl (siehe oben) fuer den
    // Trainingsloss selbst plus Werte, die bisher nur im Stdout-Log
    // (Epochendauer, LR) bzw. gar nicht (Backend, GradNorm) sichtbar waren -
    // Grundlage fuer spaeteres Plotten von Train- gegen Held-out-Kurve, ohne
    // manuelle ps aux/Log-Archaeologie. Rein additiv, kein Einfluss auf den
    // eigentlichen Trainingslauf.
    // Update 2026-08-09 (Session-Gespraech): Loss der letzten N Schritte statt
    // des Epochendurchschnitts, als eigenes Feld NEBEN 'train_loss' - nicht als
    // Ersatz, damit die Bedeutung von 'train_loss' ueber die gesamte JSONL-
    // Historie hinweg konsistent bleibt (bestehende Zeilen vor diesem Update
    // haben dort schon den Epochendurchschnitt stehen, ein stiller
    // Bedeutungswechsel im selben Feldnamen waere fuer eine spaetere
    // Auswertung nicht mehr unterscheidbar). $current_epoch_loss (treibt
    // Notbremse/cluster_drones.log) bleibt bewusst unveraendert auf dem
    // Epochendurchschnitt - nur die Metrik-Ausgabe hier aendert sich.
    $train_loss_recent_val = count($epoch_recent_losses) > 0
        ? round(array_sum($epoch_recent_losses) / count($epoch_recent_losses), 6)
        : null;

    $train_log_entry = [
        'timestamp'          => date('c'),
        'epoch'              => $epoch,
        'train_loss'         => round($current_epoch_loss, 6),
        'train_loss_recent'  => $train_loss_recent_val,
        'lr'                 => round($active_lr, 6),
        'heldout_loss'       => $heldout_loss_val,
        'heldout_ppl'        => $heldout_ppl_val,
        'gpu_backend'        => count($epoch_backends) > 0 ? implode(',', array_unique($epoch_backends)) : 'NONE',
        'grad_norm'          => count($epoch_grad_norms) > 0 ? round(array_sum($epoch_grad_norms) / count($epoch_grad_norms), 4) : null,
        'epoch_duration_sec' => round(microtime(true) - $epoch_start_time, 2),
    ];
    @file_put_contents($checkpoint_dir . "train_{$session_id}.jsonl", json_encode($train_log_entry) . "\n", FILE_APPEND);
    $epochs_completed++;

    // AUDIT-FIX: Checkpoint der Vorepoche freigeben (sonst waechst checkpoints/
    // mit jeder Epoche jeder Drohne weiter). Der aktuelle bleibt - er ist das
    // Startmodell der naechsten Epoche und das spaetere Endergebnis.
    if ($epoch > 1) {
        $prev_cp = $checkpoint_dir . "cp_{$session_id}_e" . ($epoch - 1) . ".snai";
        if (file_exists($prev_cp)) {
            @unlink($prev_cp);
            @unlink(head_sidecar_path($prev_cp));
            @unlink(growth_sidecar_path($prev_cp));
        }
    }
    
    // =========================================================================
    // DIE SYMBIO-NOTBREMSE (DYNAMIC EARLY STOPPING)
    // =========================================================================
    if ($target_loss > 0.0 && $current_epoch_loss <= $target_loss) {
        echo "\n=================================================================\n";
        echo "[👑 QUEEN] ABSOLUTE KONVERGENZ ERREICHT! Loss liegt bei: " . round($current_epoch_loss, 5) . "\n";
        echo "[👑 QUEEN] Leite Notbrems-Protokoll ein. Gefriere Gewichte der Titanin...\n";
        echo "=================================================================\n";
        break; 
    }

    // 2026-10-19: Der feste Killswitch (Abbruch ab Epoche 2 bei Loss < 2,5,
    // sobald KEIN target_loss gesetzt war) ist entfernt. Er griff genau dann,
    // wenn der Aufrufer ausdruecklich alle Epochen wollte, und ein Loss von 2,5
    // ist bei 8192 Token kein Entropie-Minimum. Fruehes Anhalten bleibt ueber
    // target_loss (argv 8) moeglich.
}

// 4. CLEANUP & LOGGING
$log_file = './factory/cluster_drones.log';
$timestamp = date('Y-m-d H:i:s');
$caste_tag = 'GEN';
if (preg_match('/expert_([A-Z]+)_/', basename($output_model), $m)) {
    $caste_tag = $m[1];
}
$uid = str_replace(['expert_', $caste_tag, '_', '.snai'], '', basename($output_model));

// Update 2026-08-09 (Session-Gespraech): Gesamtlaufzeit des kompletten Laufs
// (nicht nur einzelne Epochendauern, siehe train_<session>.jsonl) - bisher
// stand im Endlog nur der Fertigstellungs-Zeitpunkt, keine Dauer, das
// Terminal zeigte nur Zeit-pro-Step. $run_start_time oben vor dem Merge-
// Block erfasst, deckt also den kompletten Lauf ab, nicht nur die
// Epochenschleife.
$total_duration_sec = round(microtime(true) - $run_start_time, 1);
$dur_h = (int)($total_duration_sec / 3600);
$dur_m = (int)(($total_duration_sec % 3600) / 60);
$dur_s = (int)($total_duration_sec % 60);
$duration_display = $dur_h > 0 ? sprintf('%dh %dm %ds', $dur_h, $dur_m, $dur_s) : sprintf('%dm %ds', $dur_m, $dur_s);

$log_entry = sprintf("[%s] KASTE: %-8s | UID: %-12s | END-LOSS: %.5f | EPOCHEN: %d | DAUER: %s\n", $timestamp, $caste_tag, $uid, $current_epoch_loss, $max_epochs, $duration_display);
file_put_contents($log_file, $log_entry, FILE_APPEND | LOCK_EX);

// Update 2026-08-09 (Session-Gespraech): abschliessende Zusammenfassungszeile
// in train_<session>.jsonl - dieselbe Datei wie die Pro-Epoche-Zeilen aus dem
// Nachruest-Paket #7, per "epoch":"TOTAL" von den normalen Eintraegen
// unterscheidbar, damit ein Parser beides klar auseinanderhalten kann. Fasst
// fuer eine Gesamtuebersicht zusammen, was bisher ueber viele Einzelzeilen
// verstreut war (Epochendauern) bzw. gar nicht vorlag (Gesamtdauer). Bewusst
// dieselbe Datei statt einer neuen - ein Parser muss nur eine Quelle lesen.
$train_log_path = $checkpoint_dir . "train_{$session_id}.jsonl";
$summary_entry = [
    'timestamp'          => date('c'),
    'epoch'              => 'TOTAL',
    'epochs_run'         => $epochs_completed,
    'final_train_loss'   => round($current_epoch_loss, 6),
    'final_train_loss_recent' => $train_loss_recent_val ?? null,
    'final_heldout_loss' => is_finite($best_heldout_loss) ? $best_heldout_loss : null,
    'gpu_backend'        => count($epoch_backends ?? []) > 0 ? implode(',', array_unique($epoch_backends)) : 'NONE',
    'total_duration_sec' => $total_duration_sec,
    'total_duration'     => $duration_display,
];
@file_put_contents($train_log_path, json_encode($summary_entry) . "\n", FILE_APPEND);

foreach ($chunk_files as $chunk) @unlink($chunk);

// AUDIT-FIX: Vorher wurde der rename bei fehlendem Checkpoint stumm uebersprungen
// und danach trotzdem "Schwarm-Modell bereit" mit exit(0) gemeldet. Der Aufrufer
// (royal_distillery / beehive_core) sah einen Erfolg ohne Modelldatei.
if (!file_exists($checkpoint_state)) {
    echo "🚨 [Fatal] Kein Checkpoint vorhanden - es wurde kein Modell erzeugt.\n";
    exit(1);
}

$output_dir = dirname($output_model);
if (!is_dir($output_dir) && !mkdir($output_dir, 0777, true)) {
    echo "🚨 [Fatal] Zielordner $output_dir konnte nicht angelegt werden.\n";
    exit(1);
}

if (!rename($checkpoint_state, $output_model)) {
    echo "🚨 [Fatal] Konnte $checkpoint_state nicht nach $output_model verschieben.\n";
    exit(1);
}

// Head-Zahl wandert mit dem Modell mit.
move_head_sidecar($checkpoint_state, $output_model);
move_growth_sidecar($checkpoint_state, $output_model);

if (strpos(basename($output_model), 'expert_') === 0) {
    if (strpos($base_model, 'genesis_') !== false) {
        @unlink($base_model);
    }
}

echo "\n[System] Schwarm-Modell bereit: $output_model\n";
exit(0);