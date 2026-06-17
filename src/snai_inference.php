<?php
/**
 * Dateiname: snai_inference.php
 * Funktion: SYMBIO NANO-AI FRAMEWORK - API Inference Worker (Phase VI.9)
 */
chdir(__DIR__ . '/..');
error_reporting(E_ALL & ~E_NOTICE & ~E_WARNING);
ini_set('display_errors', '0');
ini_set('memory_limit', '128M'); // OOM Safe, da PHP keine Arrays mehr hält!

require_once 'snai_tokenizer.php';

if ($argc < 3) {
    fwrite(STDERR, "Nutzung: php snai_inference.php <modell.snai> <prompt>\n");
    exit(1);
}

$model_file = $argv[1];
$user_input = $argv[2];
$temperature = 0.25;         
$repetition_penalty = 1.15; 
$max_new_tokens = 256;

if (!file_exists($model_file)) {
    fwrite(STDERR, "🚨 Fehler: Matrix nicht gefunden: $model_file\n");
    echo "System-Fehler: Cortex-Artefakt nicht gefunden."; 
    exit(1);
}

// 1. DYNAMISCHE HEADER-ANALYSE (SNAI_V4 Binär-Format)
// -------------------------------------------------------------------------
$fp = fopen($model_file, 'rb');
$magic = fread($fp, 8);
if (trim($magic) !== 'SNAI_V4') {
    fwrite(STDERR, "🚨 Fatal: Invalider Header. Kein SNAI_V4 Artefakt!\n");
    echo "System-Fehler: Artefakt-Generation inkompatibel.";
    exit(1);
}
// 'V' = 32-bit unsigned Integer (Little Endian)
$params = unpack('Vvocab/Vhidden/Vffn/Vctx/Vlayers', fread($fp, 20));
fclose($fp);

$vocab_size = $params['vocab'];
$hidden_dim = $params['hidden'];
$ctx_size = $params['ctx'];
$num_layers = $params['layers'];
$ffn_dim = $params['ffn'];
$num_heads = max(1, (int)($hidden_dim / 64)); // Heuristische Standard-Ableitung

fwrite(STDERR, "[System] Binär-Artefakt decodiert: " . basename($model_file) . "\n");
fwrite(STDERR, "  > Architektur: $hidden_dim Dim | $num_layers Layers | $num_heads Heads\n");

// =========================================================================
// Strikte Dateigrößen-Validierung
// Verhindert, dass der C-Stream-Loader an unvollständigen Dateien erstickt.
// =========================================================================
$expected_size = 28; // Header
$expected_size += ($vocab_size * $hidden_dim * 4) * 2; // embeddings, w_out
$expected_size += ($hidden_dim * 4); // rms_final
$layer_size = ($hidden_dim * 4) * 2; // rms_att, rms_ffn
$layer_size += ($hidden_dim * $hidden_dim * 4) * 4; // q, k, v, o
$layer_size += ($ffn_dim * $hidden_dim * 4) * 2; // up, down
$expected_size += $layer_size * $num_layers;

$actual_size = filesize($model_file);
if ($actual_size !== $expected_size) {
    fwrite(STDERR, "\n🚨 [SYMBIO VETO] KORRUPTION ENTDECKT!\n");
    fwrite(STDERR, "  > Artefakt ist unvollständig (Trainings-Abbruch / OOM beim Speichern).\n");
    fwrite(STDERR, "  > Ist-Größe:  " . number_format($actual_size) . " Bytes\n");
    fwrite(STDERR, "  > Soll-Größe: " . number_format($expected_size) . " Bytes\n\n");
    echo "System-Fehler: Corrupt Cortex Artifact (Incomplete Write).";
    exit(1);
}

// 2. DIE FFI-BRÜCKE (Zero-Copy)
// -------------------------------------------------------------------------
$ffi = FFI::cdef(
    "void cortex_init_gpu();
     void* cortex_load_stream(const char* filepath, int num_heads);
     void cortex_forward(void* state, int* tokens, int num_tokens, float* logits);
     void cortex_reset_cache(void* state);
     void cortex_free(void* state);", 
    "./libcortex.so"
);

// Wecke die GPU auf!
fwrite(STDERR, "  > Initiiere OpenCL JIT-Compiler...\n");
$ffi->cortex_init_gpu();

// Stream Load: Die C-Library zieht die Datei direkt in den VRAM/Heap
$state_ptr = $ffi->cortex_load_stream($model_file, $num_heads);
if ($state_ptr == null) {
    fwrite(STDERR, "Fatal: C-Stream-Loader gescheitert (Unbekannter I/O Fehler).\n");
    echo "System-Fehler: Memory Allocation Error.";
    exit(1);
}

// 3. TOKENIZER & PROMPT-ENGINEERING (SFT Korsett)
// -------------------------------------------------------------------------
$tokenizer = new SnaiTokenizer('symbio_vocab.json');

// Wir rahmen den Input des Dashboards strikt ein, damit die Drohne antwortet
$formatted_prompt = "[USER] " . trim($user_input) . "\n[SYMBIO]";
$tokens = $tokenizer->encode($formatted_prompt);

if (count($tokens) > $ctx_size - $max_new_tokens) {
    $tokens = array_slice($tokens, -($ctx_size - $max_new_tokens)); // Context Window erzwingen
}

// 4. INFERENZ (Autoregressiver Loop)
// -------------------------------------------------------------------------
$c_tokens = $ffi->new("int[" . $ctx_size . "]");
$c_logits = $ffi->new("float[" . $vocab_size . "]");
$history = []; // Tracking für Repetition Penalty

// Initialer Forward Pass (Den gesamten Prompt fressen)
$ffi->cortex_reset_cache($state_ptr);
foreach ($tokens as $i => $tok) {
    $c_tokens[$i] = $tok;
    $history[] = $tok; // Prompt-Tokens ebenfalls penalisieren
}

$start_time = microtime(true);
$ffi->cortex_forward($state_ptr, $c_tokens, count($tokens), $c_logits);
$prompt_time = round((microtime(true) - $start_time) * 1000);
fwrite(STDERR, "  > Prompt verdaut in {$prompt_time}ms. Beginne Synthese...\n");

$generated_text = "";
$stop_tokens = [2]; // EOS Token
if (isset($tokenizer->encode("[USER]")[0])) $stop_tokens[] = $tokenizer->encode("[USER]")[0];

for ($i = 0; $i < $max_new_tokens; $i++) {
    $best_token = 0;

    // =========================================================================
    // NaN Tripwire
    // Wenn der C-Kernel kollabiert ist, fangen wir das hier ab!
    // =========================================================================
    if (is_nan($c_logits[0])) {
        fwrite(STDERR, "\n🚨 [SYMBIO VETO] KERNEL PANIC: Cortex spuckt NaNs! Die Matriarchin hat korrupte Gewichte (Explodierte Gradienten beim Merge).\n");
        break;
    }
    
    // Temperatur Scaling (In-Place)
    for ($v = 0; $v < $vocab_size; $v++) {
        $c_logits[$v] = $c_logits[$v] / $temperature;
    }

    // =========================================================================
    // REPETITION PENALTY
    // =========================================================================
    foreach (array_unique($history) as $prev_tok) {
        if ($c_logits[$prev_tok] > 0) {
            $c_logits[$prev_tok] /= $repetition_penalty;
        } else {
            $c_logits[$prev_tok] *= $repetition_penalty;
        }
    }

    // Max-Wert für numerisch stabiles Softmax (Dynamischer Startwert)
    $max_val = -INF;
    for ($v = 0; $v < $vocab_size; $v++) {
        if ($c_logits[$v] > $max_val) {
            $max_val = $c_logits[$v];
            if ($temperature <= 0.01) $best_token = $v; // Greedy Fallback
        }
    }

    // Softmax & Sampling
    if ($temperature > 0.01) {
        $sum = 0.0;
        $probs = [];
        for ($v = 0; $v < $vocab_size; $v++) {
            $p = exp($c_logits[$v] - $max_val);
            $probs[$v] = $p;
            $sum += $p;
        }
        
        $rand = (mt_rand() / mt_getrandmax()) * $sum;
        $cumulative = 0.0;
        for ($v = 0; $v < $vocab_size; $v++) {
            $cumulative += $probs[$v];
            if ($cumulative >= $rand) {
                $best_token = $v;
                break;
            }
        }
    }

    // Abbruchbedingungen prüfen
    if (in_array($best_token, $stop_tokens)) break;

    $history[] = $best_token;
    $word = $tokenizer->decode([$best_token]);
    
    // Halluzinations-Notbremse
    if (strpos($word, "[USER]") !== false) break;

    // Ausgabe (Fängt das Dashboard per STDOUT ab)
    echo $word; 
    $generated_text .= $word;

    // Feedback Loop: Nächstes Token berechnen
    $c_tokens[0] = $best_token;
    $ffi->cortex_forward($state_ptr, $c_tokens, 1, $c_logits);
    
    if (count($tokens) + $i >= $ctx_size - 1) break;
}

$ffi->cortex_free($state_ptr);
fwrite(STDERR, "\n[⚙️ Metrik: " . mb_strlen($generated_text) . " Chars generiert]\n");