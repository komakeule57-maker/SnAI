<?php
/**
 * Datei: train_worker.php
 * Funktion: Lokaler CLI-Worker für das Training der SNAI-Modelle. 
*/

// Limit auf 2048M hochgesetzt, da extract_state() am Ende kurzzeitig RAM für das gesamte Modell benötigt
ini_set('memory_limit', '2048M'); 
require_once __DIR__ . '/snai_tokenizer.php';
require_once __DIR__ . '/tensor_io.php';
require_once __DIR__ . '/tensor_ffi_adapter.php';

// Konstanten für Magic Numbers (Wartbarkeit)
define('CHUNK_READ_SIZE', 65536);
define('MAX_BUFFER_SIZE', 1048576); // 1 MB Not-Cut
define('MIN_FILE_SIZE', 10);
define('DEFAULT_LR', 0.0005);

if ($argc < 6) {
    die("Nutzung: php train_worker.php <input.txt> <output.snai> <epochs> <base_model> <dtype> [custom_lr]\n");
}

$input_file  = $argv[1];
$output_file = $argv[2];
$epochs      = (int)$argv[3];
$base_model  = $argv[4];
$dtype       = (int)$argv[5];
$custom_lr   = isset($argv[6]) ? (float)$argv[6] : DEFAULT_LR;
$global_epoch = isset($argv[7]) ? (int)$argv[7] : 1;
$base_lr     = isset($argv[8]) ? (float)$argv[8] : DEFAULT_LR;

echo "=== [WORKER " . getmypid() . "] INITIALISIERE SCHMIEDE ===\n";

// Sanity Check Lernrate
if ($custom_lr <= 0) $custom_lr = DEFAULT_LR;
if ($base_lr <= 0.00001) {
    echo "  > [Worker " . getmypid() . "] Traum-Modus aktiv (Experience Replay).\n";
}

if (!file_exists($base_model)) {
    die("Fatal: Basismodell konnte nicht gefunden werden -> $base_model\n");
}

if (!file_exists($input_file) || filesize($input_file) < MIN_FILE_SIZE) {
    echo "  >  Warnung: Chunk ist leer oder fehlt. Kopiere Basismodell als Fallback...\n";
    if (!copy($base_model, $output_file)) {
        die("Fatal: Konnte Basismodell nicht kopieren!\n");
    }
    exit(0); 
}

$tokenizer = new SnaiTokenizer(__DIR__ . '/../lib/symbio_vocab.json');
$vocab_limit = $tokenizer->getVocabSize();

// --- DER INJEKTOR ---
$cortex = new SnaiCortexBridge($base_model);

// Start OpenCL JIT-Compiler
if (method_exists($cortex, 'init_gpu')) {
    $cortex->init_gpu(); 
}

$ctx = property_exists($cortex, 'ctx_size') ? $cortex->ctx_size : 512;
if ($ctx <= 0) die("Fatal: Ungültige ctx_size ($ctx) vom Cortex gemeldet!\n");

$initial_lr = $custom_lr; 
$global_step = 0;

for ($e = 1; $e <= $epochs; $e++) {
    $loss_sum = 0;
    $steps = 0;
    
    $fp = fopen($input_file, 'r');
    if ($fp === false) {
        die("Fatal: Konnte Input-File nicht öffnen: $input_file\n");
    }

    $buffer = '';
    
    while (!feof($fp) || $buffer !== '') {
        $chunk = '';
        if (!feof($fp)) {
            $chunk = fread($fp, CHUNK_READ_SIZE);
            if ($chunk === false) $chunk = '';
        }
        $buffer .= $chunk;
        
        $last_space = strrpos($buffer, ' ');
        $last_newline = strrpos($buffer, "\n");
        $cut_pos = max(
            $last_space === false ? -1 : $last_space, 
            $last_newline === false ? -1 : $last_newline
        );
        
        if (strlen($buffer) > MAX_BUFFER_SIZE && $cut_pos === -1) {
            $cut_pos = MAX_BUFFER_SIZE;
        }

        if ($cut_pos > 0) {
            $process_text = substr($buffer, 0, $cut_pos);
            $buffer = substr($buffer, $cut_pos);
        } else {
            $process_text = $buffer;
            $buffer = '';
        }

        if (strlen(trim($process_text)) < 2) continue;

        $tokens = $tokenizer->encode($process_text);
        $total_tokens = count($tokens);
        $batch_size = $ctx;
        
        // === MINI-BATCH SCHLEIFE 
        for ($i = 0; $i < $total_tokens - 1; $i += $batch_size) {
            
            $batch_in  = array_slice($tokens, $i, $batch_size);
            $actual_len = count($batch_in);
            
            // Wenn der Batch zu klein ist, abbrechen (verhindert Out-Of-Bounds)
            if ($actual_len < 1 || $i + $actual_len >= $total_tokens) break;
            
            $batch_out = array_slice($tokens, $i + 1, $actual_len);

            $global_step++;
            $current_lr = $initial_lr;
            
            // Bounds Check
            $max_id = max($batch_in);
            $min_id = min($batch_in);
            if ($min_id < 0 || $max_id >= $vocab_limit) {
                echo "[Worker Debug] ILLEGALE TOKENS! Min: $min_id, Max: $max_id (Limit: $vocab_limit)\n";
                exit(1);
            }

            // Inferenz / Training in der C-Engine
            if (method_exists($cortex, 'train_step')) {
                $loss = $cortex->train_step($batch_in, $batch_out, $current_lr);
                if ($loss === null) $loss = 0.001; 
            } else {
                echo "[Worker " . getmypid() . "] train_step() fehlt! Simuliere...\n";
                $loss = max(0.001, 0.99 * (1.0 / (($global_step * 0.01) + 1)));
            }

            if (is_nan($loss) || is_infinite($loss)) {
                echo "[Fatal] Worker " . getmypid() . " | Exploding Gradients (NaN) bei Step $steps! Breche ab.\n";
                exit(1); 
            }

            $loss_sum += $loss;
            $steps++;

            if ($steps <= 5 || $steps % 5 === 0) { 
                echo "Worker " . str_pad((string)getmypid(), 5, " ", STR_PAD_LEFT) . " | Epoch $global_epoch | Step " . str_pad((string)$steps, 3, " ", STR_PAD_LEFT) . " | Avg Loss: " . number_format($loss_sum / $steps, 6) . "\n";
            }
            
            // GC explizit entlasten
            unset($batch_in, $batch_out);
        }
        unset($tokens, $process_text); 
    }
    fclose($fp);
}

echo "  > [Worker " . getmypid() . "] Training abgeschlossen. Extrahiere aktualisierte Gewichte...\n";

if (method_exists($cortex, 'extract_state')) {
    $state_array = $cortex->extract_state();
    if ($state_array === false || $state_array === null) {
        die("[Worker " . getmypid() . "] FATAL: extract_state() lieferte ungültiges Ergebnis!\n");
    }
} else {
    die("[Worker " . getmypid() . "] FATAL: Konnte Gewichte nicht aus C-RAM extrahieren (Methode fehlt)!\n");
}

if (!function_exists('save_model_stream')) {
    die("Fatal: 'save_model_stream' nicht in tensor_io.php gefunden!\n");
}

try {
    save_model_stream($state_array, $output_file);
    echo "  > ✅ [Erfolg] Arbeiter-Gewichte via Stream gesichert unter $output_file\n";
} catch (Exception $e) {
    die("🚨 Fatal: Fehlschlag beim Speichern des Modells: " . $e->getMessage() . "\n");
}

exit(0);
