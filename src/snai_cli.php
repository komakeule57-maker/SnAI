<?php
// Dateiname: src/snai_cli.php
// Funktion: Offizielles Command-Line Interface
//
// STAND 2026-09-16 (Distro-Angleichung an den Hauptzweig, v2.0):
// Zwei Verben sind ersatzlos entfallen und durch eines ersetzt worden:
//   'upscale' und 'merge' -> 'distill'
// Grund: net2net_forge.php gibt es nicht mehr. Seine beiden Fähigkeiten sind in
// RoyalDistillery aufgegangen, dort aber als PRIVATE Methoden
// (merge_tributes(), upscale_matrix_stream()) - der einzige Weg hinein ist
// distill_princess()/distill_apex_entity(), die Merge, Expansion und
// Kalibrierung ohnehin als eine Einheit fahren. Ein Einzelaufruf von aussen
// wäre nicht wiederherstellbar, ohne die Klasse aufzubohren.
//
// Ebenfalls entfallen: der [dim]-Parameter von 'train'. Er ging früher als 9.
// Argument an train_cluster.php; dort steht heute 'start_epoch'. Ein
// './symbio.sh cli train ... 128' hätte also still "starte bei Epoche 128"
// bedeutet. Die Genesis-Dimension ist in train_cluster.php:208 fest auf 128
// (Drohne, siehe EVOLUTION_LADDER) verdrahtet.

$root_dir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');
chdir($root_dir);

date_default_timezone_set('Europe/Berlin');

if ($argc < 2) {
    echo "========================================================================\n";
    echo "  SYMBIO NANO-AI CORE CLI (v2.0)\n";
    echo "========================================================================\n";
    echo "Usage (via symbio.sh):\n";
    echo "  ./symbio.sh cli train   <input.txt> <output.snai> [epochs]\n";
    echo "  ./symbio.sh cli infer   <modell.snai> \"<prompt>\" [beam_width]\n";
    echo "  ./symbio.sh cli distill [Prinzessin|Matriarchin|Empress|Demi Goddess|Goddess]\n";
    echo "========================================================================\n";
    exit(1);
}

$command = strtolower($argv[1]);

switch ($command) {
    case 'train':
        if ($argc < 4) die("Usage: ./symbio.sh cli train <input.txt> <output.snai> [epochs]\n");
        $input  = escapeshellarg($argv[2]);
        $output = escapeshellarg($argv[3]);
        $epochs = isset($argv[4]) ? max(1, (int)$argv[4]) : 20;

        echo "[Symbio CLI] Starte BPTT Training (Drohne, 128 Dim, $epochs Epochen)...\n";

        // Subprozess wird mit absolutem Pfad über __DIR__ verankert.
        // Positionen: input output workers epochs base_model datatype lr target_loss
        $script = escapeshellarg(__DIR__ . '/train_cluster.php');
        $cmd = "php $script $input $output 4 $epochs \"none\" 1 0.0002 7.5";

        passthru($cmd, $exit_code);
        exit($exit_code);

    case 'infer':
        if ($argc < 4) die("Usage: ./symbio.sh cli infer <modell.snai> \"<prompt>\" [beam_width]\n");
        $script = escapeshellarg(__DIR__ . '/snai_inference.php');
        $cmd = "php $script " . escapeshellarg($argv[2]) . ' ' . escapeshellarg($argv[3]);
        if (isset($argv[4])) $cmd .= ' ' . escapeshellarg((string)(int)$argv[4]);

        passthru($cmd, $exit_code);
        exit($exit_code);

    case 'distill':
        require_once __DIR__ . '/royal_distillery.php';
        $tier = $argv[2] ?? 'Prinzessin';
        $distillery = new RoyalDistillery();

        // Die Prinzessin entsteht aus 3 Drohnen in factory/tributes/; jede
        // höhere Stufe aus dem Vorgänger in factory/royal/ bzw. factory/apex/.
        // Für 'Prinzessin' muss der VIP-Raum vorher gestellt sein - das macht
        // im Normalbetrieb der Daemon (./symbio.sh swarm).
        if ($tier === 'Prinzessin') {
            $ok = $distillery->distill_princess(uniqid());
        } else {
            $ok = $distillery->distill_apex_entity(uniqid(), $tier);
        }

        if ($ok) {
            echo "\e[32m[Erfolg]\e[0m Destillation zur Stufe '$tier' abgeschlossen.\n";
        } else {
            echo "\e[31m[Fehler]\e[0m Destillation zur Stufe '$tier' fehlgeschlagen.\n";
            exit(1);
        }
        break;

    case 'node':
        echo "\e[33m[Hinweis]\e[0m Der direkte Aufruf über 'cli node' ist veraltet.\n";
        echo "Bitte benutze: \e[32m./symbio.sh swarm\e[0m um das Netzwerk und den Orchestrator synchron zu starten.\n";
        break;

    default:
        echo "Unbekannter Befehl: $command\n";
        exit(1);
}
