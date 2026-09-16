<?php
// Dateiname: src/snai_daemon.php
// Funktion: Standalone Swarm Orchestrator. Überwacht Ordner und triggert automatische KI-Synthesen.
//
// STAND 2026-09-16 (Distro-Angleichung an den Hauptzweig):
// Die Schleife und ihre zwei Prioritäten sind unverändert - geändert hat sich,
// WER die Synthese ausführt. Bis Juni waren das drei Handschritte über
// net2net_forge.php (merge -> upscale 256 -> train). Diese Datei gibt es nicht
// mehr; merge_tributes() und upscale_matrix_stream() sind heute private
// Methoden von RoyalDistillery und nur noch über distill_princess() erreichbar,
// das Merge, Expansion auf die EVOLUTION_LADDER-Werte (Prinzessin: 256 Dim,
// 1024 Ctx, 3 Layer) und die Kalibrierung in einem Zug erledigt.
// Ebenfalls entfallen: build_calibration_nectar(). Die Destille baut sich ihren
// Kalibrierkorpus selbst (write_calib_file()/draw_epoch_slice()).

$root = rtrim(realpath(__DIR__ . '/..'), DIRECTORY_SEPARATOR);
chdir($root);

date_default_timezone_set('Europe/Berlin');

require_once __DIR__ . '/royal_distillery.php';
require_once __DIR__ . '/head_config.php';

// SINGLE-INSTANCE MUTEX
$lock_fp = fopen(__DIR__ . '/daemon.lock', 'c');
if (!flock($lock_fp, LOCK_EX | LOCK_NB)) {
    echo "\n\e[41m\e[97m 🚨 [DAEMON] KLON-KOLLISION (ZOMBIE DAEMON ENTDECKT) \e[0m\n";
    echo "\e[31mEin alter Daemon läuft bereits heimlich im Hintergrund und stiehlt Dateien!\e[0m\n";
    echo "Aktion erforderlich: Führe im Terminal \e[33mkillall php\e[0m aus, um alle alten Instanzen zu beenden!\n\n";
    exit(1);
}

error_reporting(E_ALL & ~E_NOTICE);
ini_set('display_errors', '1');

$dirs = [
    $root.'/factory/input',     $root.'/factory/experts',
    $root.'/factory/royal',     $root.'/factory/checkpoints',
    $root.'/factory/archive',   $root.'/factory/tributes',
    $root.'/factory/royal_fragments', $root.'/factory/apex',
];
foreach ($dirs as $d) if (!is_dir($d)) @mkdir($d, 0777, true);

echo "\e[32m====================================================\e[0m\n";
echo "\e[32m=== [DAEMON] SYMBIO SCHWARM AKTIVIERT            ===\e[0m\n";
echo "\e[32m====================================================\e[0m\n";
echo "[Daemon] Überwache P2P Ordner auf neues Trainingsmaterial...\n\n";

$distillery = new RoyalDistillery();

while (true) {
    $experts       = glob($root . '/factory/experts/*.snai');
    $nectar_inputs = glob($root . '/factory/input/*.txt');

    // PRIO 1: Prinzessinnen-Synthese
    if (count($experts) >= 3) {
        echo "\n\e[36m[Daemon]\e[0m Kritische Masse erreicht (3+ Drohnen). Initiiere Synthese...\n";

        // Die Destille liest ihre Tribute aus factory/tributes/, nicht aus
        // factory/experts/ - der VIP-Raum wird vor jeder Synthese frisch
        // gestellt. Das .heads-Sidecar MUSS mitkopiert werden, sonst fällt
        // distill_princess() mangels Sidecar auf Legacy-1-Head zurück und
        // mittelt Modelle mit unterschiedlicher Attention-Struktur zusammen.
        $tribute_dir = $root . '/factory/tributes/';
        array_map('unlink', glob($tribute_dir . '*.snai')  ?: []);
        array_map('unlink', glob($tribute_dir . '*.heads') ?: []);

        usort($experts, function($a, $b) { return filemtime($b) <=> filemtime($a); });
        foreach (array_slice($experts, 0, 3) as $exp) {
            copy($exp, $tribute_dir . basename($exp));
            $exp_heads_sidecar = head_sidecar_path($exp);
            if (file_exists($exp_heads_sidecar)) {
                @copy($exp_heads_sidecar, head_sidecar_path($tribute_dir . basename($exp)));
            }
        }

        // Merge, Expansion und Kalibrierung in einem Zug. Räumt bei Erfolg die
        // verbrauchten Drohnen selbst nach factory/royal_fragments/ ab und legt
        // die Prinzessin in factory/royal/.
        if ($distillery->distill_princess(uniqid())) {
            echo "\e[32m[Daemon] ERFOLG! Neue Prinzessin erschaffen.\e[0m\n";
        } else {
            echo "\e[33m[Daemon] Synthese abgebrochen - Original-Drohnen bleiben unangetastet.\e[0m\n";
        }

        sleep(5);
        continue;
    }

    // PRIO 2: Drohnen-Schmiede
    if (count($nectar_inputs) > 0) {
        $fresh_nectar = realpath($nectar_inputs[0]);
        if (!$fresh_nectar) {
            sleep(2);
            continue;
        }

        $uid = uniqid();
        $tag = "GEN";
        if (preg_match('/hf_([A-Z]+)_/', basename($fresh_nectar), $matches)) {
            $tag = $matches[1];
        }

        $expert_out = $root . "/factory/experts/expert_{$tag}_{$uid}.snai";

        echo "\n\e[36m[Daemon]\e[0m Neue Trainingsdaten entdeckt (" . basename($fresh_nectar) . "). Starte Drohnentraining...\n";

        // Drohne = 128 Dim, 2 Layer, Genesis-Init ("none"), siehe
        // EVOLUTION_LADDER['Drohne'] in royal_distillery.php. Die acht
        // Positionen unten sind die Pflichtargumente von train_cluster.php;
        // die fünf optionalen dahinter (start_epoch, weight_decay,
        // dropout_rate, lr_schedule_epochs, label_smoothing) bleiben auf ihren
        // Vorgaben - die Destille setzt sie, das Drohnentraining nicht.
        $train_cmd = sprintf(
            "php %s %s %s 4 10 none %d 0.0002 7.0",
            escapeshellarg(__DIR__ . '/train_cluster.php'),
            escapeshellarg($fresh_nectar),
            escapeshellarg($expert_out),
            SNAI_TYPE_FP32
        );
        passthru($train_cmd, $exit_code);

        // Verschieben ins Archiv
        $archive_path = str_replace(DIRECTORY_SEPARATOR . 'input' . DIRECTORY_SEPARATOR, DIRECTORY_SEPARATOR . 'archive' . DIRECTORY_SEPARATOR, $fresh_nectar);

        if ($exit_code === 0 && file_exists($expert_out)) {
            echo "\e[32m[Daemon] Drohne erfolgreich fertig gestellt: $expert_out\e[0m\n";
            @rename($fresh_nectar, $archive_path);
        } else {
            echo "\e[31m[Daemon] Fehler beim Drohnen-Training. Trainingsdaten werden archiviert.\e[0m\n";
            @rename($fresh_nectar, $archive_path);
        }

        sleep(5);
        continue;
    }

    sleep(30);
}
