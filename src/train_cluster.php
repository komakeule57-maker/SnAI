<?php
// Dateiname: src/train_cluster.php
// Funktion: Steuert das verteilte Training (BPE-Worker), verwaltet Epochen und verschmilzt Tensoren.

ini_set('memory_limit', '1024M');

// ABSOLUTE PFAD-VERANKERUNG (Bulletproof - Eliminiert CWD-Bugs)
$root_dir = realpath(__DIR__ . '/..') ?: (__DIR__ . '/..');
chdir($root_dir);

require_once __DIR__ . '/tensor_io.php';
require_once __DIR__ . '/tensor_ffi_adapter.php';
require_once __DIR__ . '/snai_tokenizer.php'; 

if ($argc < 6) {
    echo "Nutzung: php src/train_cluster.php <input.txt> <output.snai> <num_workers> <max_epochs> <base_model> [datatype] [custom_lr] [target_loss]\n";
    exit(1);
}

// Eingabepfade erzwingen Absolutheit (immun gegen CWD-Verschiebungen)
$input_file = $argv[1];
if (strpos($input_file, '/') !== 0 && strpos($input_file, ':') !== 1) {
    $input_file = $root_dir . '/' . ltrim($input_file, './');
}

$output_model = $argv[2];
if (strpos($output_model, '/') !== 0 && strpos($output_model, ':') !== 1) {
    $output_model = $root_dir . '/' . ltrim($output_model, './');
}

$num_workers  = (int)$argv[3];
$max_epochs   = (int)$argv[4];
$base_model   = $argv[5];
if ($base_model !== 'none' && strpos($base_model, '/') !== 0 && strpos($base_model, ':') !== 1) {
    $base_model = $root_dir . '/' . ltrim($base_model, './');
}

$target_dtype = isset($argv[6]) ? (int)$argv[6] : 1; 
$custom_lr    = isset($argv[7]) ? $argv[7] : "0.0005"; 
$target_loss  = isset($argv[8]) ? (float)$argv[8] : 0.0;

if ($num_workers <= 0 || $max_epochs <= 0) {
    echo "[Fatal] num_workers und max_epochs müssen > 0 sein.\n";
    exit(1);
}

$session_id = pathinfo($output_model, PATHINFO_FILENAME);

if (!file_exists($input_file)) {
    echo "[Fatal] Input-Datei $input_file fehlt.\n";
    exit(1);
}

$checkpoint_dir = $root_dir . '/factory/checkpoints/';
if (!is_dir($checkpoint_dir)) @mkdir($checkpoint_dir, 0777, true);

echo "=== [CLUSTER MASTER | SESSION: $session_id] INITIALISIERUNG ===\n";

// SMART FORGE
if ($base_model === 'none') {
    if (file_exists($output_model)) {
        echo "[Forge] Ziel-Modell existiert bereits. Nutze es als Basis (Fine-Tuning)...\n";
        $base_model = $output_model;
    } else {
        // FIX: Nutzt jetzt absolut verankerten lib-Ordner
        $tokenizer = new SnaiTokenizer(__DIR__ . '/../lib/symbio_vocab.json');
        $dynamic_vocab_size = $tokenizer->getVocabSize();
        
        echo "[Forge] Kein Basismodell gefunden. Erschaffe Multi-Layer Genesis-Matrix (Vocab: $dynamic_vocab_size)...\n";
        $base_model = $checkpoint_dir . "genesis_{$session_id}.snai";
        
        $init_cmd = sprintf(
            "php %s %s %d 128 512 2 2>&1",
            escapeshellarg(__DIR__ . '/init_base.php'),
            escapeshellarg($base_model),
            $dynamic_vocab_size
        );
        $init_output = shell_exec($init_cmd);
        echo $init_output;
        
        if (!file_exists($base_model)) {
            echo "[Fatal] Genesis-Modell konnte nicht erstellt werden!\n";
            exit(1);
        }
    }
} elseif (!file_exists($base_model)) {
    echo "[Fatal] Angegebenes Basismodell $base_model fehlt.\n";
    exit(1);
}


// 1. DATA PREP (Zirkulärer Puffer)
$file_size = filesize($input_file);
$min_chunk_size = 2048; 
if ($file_size < ($min_chunk_size * $num_workers)) {
    $num_workers = max(1, floor($file_size / $min_chunk_size));
}

$chunk_size = ceil($file_size / $num_workers);
$host_hash = crc32(gethostname() . $session_id);
$entropy_offset = $host_hash % $file_size;

echo "[I/O] Splitting $file_size Bytes in $num_workers Chunks\n";
echo "  > Host-Entropie aktiviert. Start-Offset: $entropy_offset Bytes.\n";

$fp = fopen($input_file, 'r');
fseek($fp, $entropy_offset);

$chunk_files = [];
for ($i = 0; $i < $num_workers; $i++) {
    $chunk_data = "";
    $bytes_to_read = $chunk_size;

    while ($bytes_to_read > 0) {
        $read = fread($fp, $bytes_to_read);
        if ($read === false || strlen($read) === 0) {
            // (Ringpuffer)
            rewind($fp);
            $read = fread($fp, $bytes_to_read);
            if ($read === false || strlen($read) === 0) break;
        }
        $chunk_data .= $read;
        $bytes_to_read -= strlen($read);
    }
    
    $chunk_name = $checkpoint_dir . "chunk_{$session_id}_{$i}.txt"; 
    file_put_contents($chunk_name, $chunk_data);
    $chunk_files[] = $chunk_name;
}
fclose($fp);

// 2. FEDERATED AVERAGING
function average_models_stream(array $worker_models, string $output_path) {
    if (empty($worker_models)) {
        echo "[Fatal] Keine Worker-Outputs zum Mergen vorhanden!\n";
        exit(1);
    }
    
    $num_models = count($worker_models);
    
    $expected_size = filesize($worker_models[0]);
    foreach ($worker_models as $file) {
        if (!file_exists($file) || filesize($file) !== $expected_size) {
            echo "[Fatal] Inkonsistente oder fehlende Worker-Dateien vor dem Merge: $file\n";
            exit(1);
        }
    }
    
    if ($num_models === 1) {
        copy($worker_models[0], $output_path);
        return;
    }

    $handles = [];
    $headers = [];
    foreach ($worker_models as $file) {
        $fp = fopen($file, 'rb');
        if (!$fp) {
            echo "[Fatal] Kann Worker-Modell nicht lesen: $file\n";
            exit(1);
        }
        $handles[] = $fp;
        $headers[] = fread($fp, 28);
    }

    if (count(array_unique($headers)) > 1) {
        echo "[Fatal] Worker-Modelle haben inkonsistente Header (Vocab/Dimension Mismatch)!\n";
        exit(1);
    }

    $out_fp = fopen($output_path, 'wb');
    if (!$out_fp) {
        foreach ($handles as $h) fclose($h);
        echo "[Fatal] Kann Output-Modell nicht schreiben: $output_path\n";
        exit(1);
    }
    fwrite($out_fp, $headers[0]);

    // Batch-Packing eliminiert CPU-Flaschenhals
    $chunk_floats = 8192; 
    $chunk_bytes = $chunk_floats * 4; 

    while (!feof($handles[0])) {
        $chunks = [];
        for ($i = 0; $i < $num_models; $i++) $chunks[] = fread($handles[$i], $chunk_bytes);
        if (strlen($chunks[0]) === 0) break;

        $num_floats_read = strlen($chunks[0]) / 4;
        $arrays = [];
        for ($i = 0; $i < $num_models; $i++) $arrays[] = array_values(unpack("g*", $chunks[$i]));

        $buffer = [];
        $bin_out = '';
        $batch_size = 8192;
        
        for ($f = 0; $f < $num_floats_read; $f++) {
            $sum = 0.0;
            for ($i = 0; $i < $num_models; $i++) $sum += $arrays[$i][$f] ?? 0.0;
            $buffer[] = $sum / $num_models;
            
            if (count($buffer) >= $batch_size) {
                $bin_out .= pack("g*", ...$buffer);
                $buffer = [];
            }
        }
        if (!empty($buffer)) $bin_out .= pack("g*", ...$buffer);
        fwrite($out_fp, $bin_out);
    }

    foreach ($handles as $h) fclose($h);
    fclose($out_fp);
    echo "[Merge] Multi-Layer Tensoren & Adam-Gedächtnis erfolgreich via Stream verschmolzen.\n";
}

// 3. DIE EPOCHEN-SCHLEIFE
$checkpoint_state = ""; 
$current_epoch_loss = 999.0;
$base_lr = (float)str_replace(',', '.', $custom_lr); 

for ($epoch = 1; $epoch <= $max_epochs; $epoch++) {
    
    $warmup_epochs = max(1, floor($max_epochs * 0.15));
    $active_lr = 0.0;

    if ($epoch <= $warmup_epochs) {
        $active_lr = $base_lr * ($epoch / $warmup_epochs);
    } else {
        $progress = ($epoch - $warmup_epochs) / ($max_epochs - $warmup_epochs);
        $active_lr = $base_lr * (0.5 * (1 + cos(pi() * $progress)));
    }
    
    $active_lr = max($active_lr, $base_lr * 0.1);
    $lr_display = number_format($active_lr, 6, '.', '');

    echo "\n>>> EPOCHE $epoch/$max_epochs | LR: $lr_display | Starte $num_workers Worker...\n";
    
    $processes = [];
    $pipes = [];
    $worker_outputs = [];
    $epoch_losses = [];

    for ($i = 0; $i < $num_workers; $i++) {
        $worker_out = $checkpoint_dir . "worker_{$session_id}_{$i}_e{$epoch}.snai"; 
        $worker_outputs[] = $worker_out;
        $start_model = ($epoch === 1) ? $base_model : $checkpoint_state; 
        
        $cmd = sprintf(
            "php %s %s %s 1 %s %d %F %d %F 2>&1",
            escapeshellarg(__DIR__ . '/train_worker.php'),
            escapeshellarg($chunk_files[$i]),
            escapeshellarg($worker_out),
            escapeshellarg($start_model),
            $target_dtype,
            $active_lr,
            $epoch,
            $base_lr
        );

        $process = proc_open($cmd, [0 => ["pipe", "r"], 1 => ["pipe", "w"], 2 => ["pipe", "w"]], $pipes_arr);
        
        if (is_resource($process)) {
            stream_set_blocking($pipes_arr[1], false);
            stream_set_blocking($pipes_arr[2], false); 
            $processes[$i] = $process;
            $pipes[$i] = $pipes_arr;
        }
    }

    $active_workers = $num_workers;
    
    // Event-Driven Steuerung via stream_select
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
            foreach ($read_pipes as $ready_pipe) {
                $worker_id = -1;
                foreach ($pipes as $idx => $p) {
                    if ($p[1] === $ready_pipe) { $worker_id = $idx; break; }
                }
                
                if ($worker_id === -1) continue;

                $output = stream_get_contents($ready_pipe);
                if ($output) {
                    echo "[" . date('H:i:s') . "] " . trim($output) . "\n";
                    
                    if (preg_match_all('/Step\s+(\d+)\s*\|\s*Avg Loss:\s*([0-9\.]+)/', $output, $matches)) {
                        $last_idx = count($matches[0]) - 1;
                        $avg_loss = (float)$matches[2][$last_idx];
                        $epoch_losses[$worker_id] = $avg_loss; 
                    } elseif (preg_match('/Avg Loss:\s*([0-9\.]+)/', $output, $matches)) {
                        $epoch_losses[$worker_id] = (float)$matches[1];
                    }
                }
            }
        }
        
        foreach ($processes as $i => $process) {
            if (!is_resource($process)) continue;
            
            $status = proc_get_status($process);
            if (!$status['running']) {
                $exitcode = $status['exitcode'];
                
                $final_out = stream_get_contents($pipes[$i][1]);
                if ($final_out) echo trim($final_out) . "\n";

                foreach ($pipes[$i] as $pipe) {
                    if (is_resource($pipe)) fclose($pipe);
                }
                
                proc_close($process); 
                unset($processes[$i]);
                $active_workers--;

                if ($exitcode !== 0) {
                    echo "[Cluster] Fatal: Worker $i ist abgestürzt (Exit-Code: $exitcode). Breche Epoche ab.\n";
                    exit(1); 
                }
            }
        }
    }
    
    $checkpoint_state = $checkpoint_dir . "cp_{$session_id}_e{$epoch}.snai";
    average_models_stream($worker_outputs, $checkpoint_state); 
    
    foreach ($worker_outputs as $wo) { @unlink($wo); }

    $current_epoch_loss = count($epoch_losses) > 0 ? (array_sum($epoch_losses) / count($epoch_losses)) : 999.0;
    echo "[Metrik] Globaler Avg Loss: " . round($current_epoch_loss, 5) . "\n";
    
    // DIE NOTBREMSE (DYNAMIC EARLY STOPPING)
    if ($target_loss > 0.0 && $current_epoch_loss <= $target_loss) {
        echo "\n=================================================================\n";
        echo "[System] ABSOLUTE KONVERGENZ ERREICHT! Loss liegt bei: " . round($current_epoch_loss, 5) . "\n";
        echo "Leite Notbrems-Protokoll ein. Gefriere Gewichte der Titanin...\n";
        echo "=================================================================\n";
        break; 
    }

    // Der alte BPE-Killswitch (Nur aktiv, wenn keine spezifische Notbremse gesetzt wurde)
    if ($epoch > 1 && $current_epoch_loss < 2.5 && $target_loss == 0.0) {
        echo "[Killswitch] Entropie-Minimum für BPE erreicht. Beende vorzeitig.\n";
        break; 
    }
}

// 4. CLEANUP & LOGGING
foreach ($chunk_files as $chunk) @unlink($chunk);

if (file_exists($checkpoint_state)) {
    $output_dir = dirname($output_model);
    if (!is_dir($output_dir)) mkdir($output_dir, 0777, true);
    rename($checkpoint_state, $output_model);
}

if (strpos(basename($output_model), 'expert_') === 0) {
    if (strpos($base_model, 'genesis_') !== false) {
        @unlink($base_model);
    }
}

echo "\n[System] Schwarm-Modell bereit: $output_model\n";
exit(0);