<?php
// Dateiname: src/snai_cli.php
// Funktion: Offizielles Command-Line Interface

$root_dir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');
chdir($root_dir);

if ($argc < 2) {
    echo "========================================================================\n";
    echo "  SYMBIO NANO-AI CORE CLI (v1.1)\n";
    echo "========================================================================\n";
    echo "Usage (via symbio.sh):\n";
    echo "  ./symbio.sh cli train <input.txt> <output.snai> [dim]\n";
    echo "  ./symbio.sh cli upscale <input.snai> <output.snai> <target_dim>\n";
    echo "  ./symbio.sh cli merge <output.snai> <input1.snai> <input2.snai> ...\n";
    echo "========================================================================\n";
    exit(1);
}

$command = strtolower($argv[1]);

switch ($command) {
    case 'train':
        if ($argc < 4) die("Usage: ./symbio.sh cli train <input.txt> <output.snai> [dim]\n");
        $input = escapeshellarg($argv[2]);
        $output = escapeshellarg($argv[3]);
        
        // Dynamischer Dimension-Override (Standard: 128)
        $target_dim = isset($argv[4]) ? (int)$argv[4] : 128;
        
        echo "[Symbio CLI] Starte BPTT Training (Genesis-Dim: $target_dim)...\n";
        
        // Subprozess wird mit absolutem Pfad über __DIR__ verankert
        $script = escapeshellarg(__DIR__ . '/train_cluster.php');
        
        // target_dim wird als 9. Parameter an den Cluster Master übergeben!
        $cmd = "php $script $input $output 4 20 \"none\" 1 0.0002 7.5 $target_dim";
        
        passthru($cmd);
        break;

    case 'upscale':
        if ($argc < 5) die("Usage: ./symbio.sh cli upscale <in.snai> <out.snai> <target_dim>\n");
        require_once __DIR__ . '/net2net_forge.php';
        $forge = new Net2NetForge();
        $forge->upscale_model($argv[2], $argv[3], (int)$argv[4]);
        break;

    case 'merge':
        if ($argc < 5) die("Usage: ./symbio.sh cli merge <out.snai> <in1.snai> <in2.snai> ...\n");
        require_once __DIR__ . '/net2net_forge.php';
        
        $output = $argv[2];
        $inputs = array_slice($argv, 3);
        
        echo "[Symbio CLI] Verschmelze Tensoren via Net2Net Forge...\n";
        $forge = new Net2NetForge();
        if ($forge->merge_models($inputs, $output)) {
            echo "\e[32m[Erfolg]\e[0m Modelle erfolgreich zu $output verschmolzen.\n";
        } else {
            echo "\e[31m[Fehler]\e[0m Verschmelzung fehlgeschlagen.\n";
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
