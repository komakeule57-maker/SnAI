<?php
/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Base Model Initializer (Phase VIII - Adam Persistence)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Gießt die komplexe Architektur direkt in einen 
 * sequentiellen, C/Nim-kompatiblen Binär-Stream (Little Endian).
 * UPDATE: Schreibt nun das SNAI_V5 Format. Generiert interleaved
 * Zero-Arrays für das Momentum (m) und die Varianz (v) des Adam-Optimizers.
 */
require_once 'snai_tokenizer.php';

if ($argc < 2) {
    die("Nutzung: php init_base.php <output.snai> [vocab_size=8192] [hidden_dim=128] [ctx_size=512] [num_layers=2] [ffn_dim=auto]\n");
}

$output_file = $argv[1];

if (!isset($argv[2]) || $argv[2] == 0) {
    $tokenizer = new SnaiTokenizer();
    $vocab_size = $tokenizer->getVocabSize(); 
    echo "> Automatische Vocab-Größe erkannt: $vocab_size\n";
} else {
    $vocab_size = (int)$argv[2];
}

$hidden_dim  = isset($argv[3]) ? (int)$argv[3] : 128;
$ctx_size    = isset($argv[4]) ? (int)$argv[4] : 512;
$num_layers  = isset($argv[5]) ? (int)$argv[5] : 2;

// AUDIT-FIX: Explizite FFN-Dim Unterstützung (Für Llama/Mistral Architektur-Treue)
$ffn_dim = isset($argv[6]) ? (int)$argv[6] : ($hidden_dim * 4);

echo "=== INITIALISIERE MULTI-LAYER CORTEX (STREAM-MODUS - V5) ===\n";
echo "Parameter: Vocab=$vocab_size | Dim=$hidden_dim | FFN=$ffn_dim | Ctx=$ctx_size | Layers=$num_layers\n";

/**
 * Generiert und schreibt Tensor-Daten direkt in Chunks auf die Festplatte.
 * Schreibt bei V5 IMMER drei Blobs nacheinander: Gewichte (w), Momentum (m), Varianz (v).
 */
function write_tensor_with_adam($fp, int $size, bool $is_ones = false, float $scale = 0.02) {
    $chunk_size = 8192; 
    
    // 1. Die tatsächlichen Gewichte (w) schreiben
    $buffer_w = '';
    for ($i = 0; $i < $size; $i++) {
        $val = $is_ones ? 1.0 : ((mt_rand() / mt_getrandmax()) * 2 - 1) * $scale;
        $buffer_w .= pack('g', $val); 
        
        if (($i + 1) % $chunk_size === 0) {
            fwrite($fp, $buffer_w);
            $buffer_w = ''; 
        }
    }
    if (strlen($buffer_w) > 0) fwrite($fp, $buffer_w);

    // 2. Das Momentum (m) für Adam mit Nullen (0.0) überschreiben
    $buffer_zero = '';
    $zero_val = pack('g', 0.0);
    for ($i = 0; $i < $size; $i++) {
        $buffer_zero .= $zero_val;
        if (($i + 1) % $chunk_size === 0) {
            fwrite($fp, $buffer_zero); // Schreibe Momentum (m)
            $buffer_zero = '';
        }
    }
    if (strlen($buffer_zero) > 0) fwrite($fp, $buffer_zero);
    
    // 3. Die Varianz (v) für Adam mit Nullen (0.0) überschreiben
    $buffer_zero = '';
    for ($i = 0; $i < $size; $i++) {
        $buffer_zero .= $zero_val;
        if (($i + 1) % $chunk_size === 0) {
            fwrite($fp, $buffer_zero); // Schreibe Varianz (v)
            $buffer_zero = '';
        }
    }
    if (strlen($buffer_zero) > 0) fwrite($fp, $buffer_zero);
}

$start_time = microtime(true);

$dir = dirname($output_file);
if (!is_dir($dir) && $dir !== '.') {
    mkdir($dir, 0777, true);
}

$fp = fopen($output_file, 'wb');
if (!$fp) {
    die("❌ [Fehler] Konnte Datei nicht öffnen: $output_file\n");
}

echo "> Schreibe V5-Header...\n";
// V5 Magic String (auf 8 Bytes gepaddet)
fwrite($fp, str_pad('SNAI_V5', 8, "\0"));
fwrite($fp, pack('VVVVV', $vocab_size, $hidden_dim, $ffn_dim, $ctx_size, $num_layers));

echo "> Gieße Globale Tensoren (Embeddings & Output) + Adam State...\n";
write_tensor_with_adam($fp, $vocab_size * $hidden_dim, false, 1.0 / sqrt($hidden_dim));
write_tensor_with_adam($fp, $vocab_size * $hidden_dim, false, 0.0001); // w_out (Softmax-Schutz)
write_tensor_with_adam($fp, $hidden_dim, true); // rms_final

// AUDIT-FIX (Session-Diskussion 2026-08-16): GPT-2-artige Tiefenskalierung der beiden
// "Rueckschreib"-Projektionen pro Block (w_o fuer Attention, w_down fuer FFN). Ohne
// diesen Faktor waechst die Varianz des Residual-Streams mit der Layerzahl an (jeder
// Layer addiert unveraendert auf den Stream) - reiner Init-Zeitpunkt-Effekt, macht
// tiefere Genesis-Netze von Anfang an optimierungsfreundlicher (weniger Warmup/kleinere
// noetige LR bei mehr Tiefe), ohne die Forward-Architektur zu aendern. Bei den aktuell
// 2 Drohnen-Layern kaum spuerbar, zahlt sich erst aus, falls Genesis-Init je tiefer wird.
$depth_scale = 1.0 / sqrt(2.0 * $num_layers);

for ($l = 0; $l < $num_layers; $l++) {
    echo "> Gieße Layer $l (QKV, Attention, FFN, RMSNorm) + Adam State...\n";
    write_tensor_with_adam($fp, $hidden_dim, true); // rms_att
    write_tensor_with_adam($fp, $hidden_dim, true); // rms_ffn
    write_tensor_with_adam($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_q
    write_tensor_with_adam($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_k
    write_tensor_with_adam($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_v
    write_tensor_with_adam($fp, $hidden_dim * $hidden_dim, false, (1.0 / sqrt($hidden_dim)) * $depth_scale); // w_o (Residual-Rueckschreib -> tiefenskaliert)
    write_tensor_with_adam($fp, $ffn_dim * $hidden_dim, false, 0.02); // w_up
    write_tensor_with_adam($fp, $hidden_dim * $ffn_dim, false, 0.02 * $depth_scale); // w_down (Residual-Rueckschreib -> tiefenskaliert)
}

fclose($fp);

$duration = round(microtime(true) - $start_time, 2);
$mb = round(filesize($output_file) / 1024 / 1024, 2);

echo "✅ [Erfolg] Genesis-Cortex (V5) gegossen: $output_file ($mb MB) in $duration Sekunden.\n";