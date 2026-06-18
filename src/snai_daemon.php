<?php
/** Dateiname: src/snai_daemon.php
  * Funktion: Standalone Swarm Orchestrator. Überwacht Ordner und triggert automatische KI-Synthesen.
  * SYMBIO NANO-AI FRAMEWORK
  * Modul: Base Model Initializer
  * This file is part of SnAI.
  *
  * SnAI is free software: you can redistribute it and/or modify
  * it under the terms of the GNU Affero General Public License as published
  * by the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
*/
$root = rtrim(realpath(__DIR__ . '/..'), DIRECTORY_SEPARATOR);
chdir($root);


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

$dirs = [$root.'/factory/input', $root.'/factory/experts', $root.'/factory/royal', $root.'/factory/checkpoints', $root.'/factory/archive'];
foreach ($dirs as $d) if (!is_dir($d)) @mkdir($d, 0777, true);

echo "\e[32m====================================================\e[0m\n";
echo "\e[32m=== [DAEMON] SYMBIO SCHWARM AKTIVIERT            ===\e[0m\n";
echo "\e[32m====================================================\e[0m\n";
echo "[Daemon] Überwache P2P Ordner auf neues Trainingsmaterial...\n\n";

function build_calibration_nectar($target_files = 5) {
    global $root;
    $archives = glob($root . '/factory/archive/*.txt');
    $inputs = glob($root . '/factory/input/*.txt');
    $nectar_pool = array_merge($archives, $inputs);
    
    if (count($nectar_pool) < $target_files) return false;

    shuffle($nectar_pool);
    $selected = array_slice($nectar_pool, 0, $target_files);
    
    $calib_file = $root . '/factory/checkpoints/calib_tmp_' . uniqid() . '.txt';
    $out_fp = fopen($calib_file, 'wb');
    foreach ($selected as $file) {
        $in_fp = fopen($file, 'rb');
        if ($in_fp) {
            stream_copy_to_stream($in_fp, $out_fp);
            fwrite($out_fp, "\n");
            fclose($in_fp);
        }
    }
    fclose($out_fp);
    return $calib_file;
}

while (true) {
    $experts = glob($root . '/factory/experts/*.snai');
    $nectar_inputs = glob($root . '/factory/input/*.txt');
    
    // PRIO 1: Prinzessinnen-Synthese
    if (count($experts) >= 3) {
        echo "\n\e[36m[Daemon]\e[0m Kritische Masse erreicht (3+ Drohnen). Initiiere Synthese...\n";
        
        $calib_file = build_calibration_nectar(5);
        if (!$calib_file) {
            echo "  > \e[33m[Warnung]\e[0m Nicht genug Trainingsdaten im Archiv/Input (5 benötigt). Warte...\n";
            sleep(30);
            continue;
        }

        $session = uniqid();
        $proto = $root . "/factory/checkpoints/proto_{$session}.snai";
        $expanded = $root . "/factory/checkpoints/exp_{$session}.cortex";
        $final = $root . "/factory/royal/princess_GEN_{$session}.cortex";
        
        $tributes = array_slice($experts, 0, 3);
        $tribute_args = implode(' ', array_map('escapeshellarg', $tributes));
        
        $merge_cmd = sprintf("php %s merge %s %s", escapeshellarg(__DIR__ . '/net2net_forge.php'), escapeshellarg($proto), $tribute_args);
        echo "  > [Merge] " . exec($merge_cmd) . "\n";

        if (file_exists($proto)) {
            $up_cmd = sprintf("php %s upscale %s %s 256", escapeshellarg(__DIR__ . '/net2net_forge.php'), escapeshellarg($proto), escapeshellarg($expanded));
            echo "  > [Upscale] " . exec($up_cmd) . "\n";
            
            if (file_exists($expanded)) {
                echo "  > [Training] Starte lokalen Cluster...\n";
                
                $train_cmd = sprintf(
                    "php %s %s %s 4 3 %s 1 0.0001 0.0",
                    escapeshellarg(__DIR__ . '/train_cluster.php'),
                    escapeshellarg($calib_file),
                    escapeshellarg($final),
                    escapeshellarg($expanded)
                );
                passthru($train_cmd, $exit_code);
                
                if ($exit_code === 0 && file_exists($final)) {
                    echo "\e[32m[Daemon] ERFOLG! Neue Prinzessin erschaffen: $final\e[0m\n";
                    foreach ($tributes as $t) @unlink($t);
                }
            }
        }
        
        @unlink($proto);
        @unlink($expanded);
        @unlink($calib_file);
        
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
        
        $train_cmd = sprintf(
            "php %s %s %s 4 10 none 1 0.0002 7.0",
            escapeshellarg(__DIR__ . '/train_cluster.php'),
            escapeshellarg($fresh_nectar),
            escapeshellarg($expert_out)
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
