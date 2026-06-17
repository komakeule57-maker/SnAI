<?php
/**
 * SYMBIO NANO-AI FRAMEWORK
 * Modul: Base Model Initializer
 */
chdir(__DIR__ . '/..');
require_once 'snai_tokenizer.php';

if ($argc < 2) {
    die("Nutzung: php init_base.php <output.snai> [vocab_size=8192] [hidden_dim=128] [ctx_size=512] [num_layers=2]\n");
}

$output_file = $argv[1];
// 2. Dynamisches Vocab-Limit: Falls kein Argument übergeben wurde, Tokenizer fragen
if (!isset($argv[2]) || $argv[2] == 0) {
    $tokenizer = new SnaiTokenizer('symbio_vocab.json');
    $vocab_size = $tokenizer->getVocabSize(); // Holt z.B. 16384 aus der JSON
    echo "> Automatische Vocab-Größe erkannt: $vocab_size\n";
} else {
    $vocab_size = (int)$argv[2];
}

$hidden_dim  = isset($argv[3]) ? (int)$argv[3] : 128;
$ctx_size    = isset($argv[4]) ? (int)$argv[4] : 512;
$num_layers  = isset($argv[5]) ? (int)$argv[5] : 2;
$ffn_dim     = $hidden_dim * 4;

echo "=== INITIALISIERE MULTI-LAYER CORTEX (STREAM-MODUS) ===\n";
echo "Parameter: Vocab=$vocab_size | Dim=$hidden_dim | Ctx=$ctx_size | Layers=$num_layers\n";

/**
 * Generiert und schreibt Tensor-Daten direkt in Chunks auf die Festplatte.
 * Absolut OOM-Sicher, RAM-Verbrauch bleibt konstant im Kilobyte-Bereich.
 */
function write_tensor($fp, int $size, bool $is_ones = false, float $scale = 0.02) {
    $chunk_size = 8192; // 8K Werte (32KB) pro Chunk
    $buffer = '';
    
    for ($i = 0; $i < $size; $i++) {
        $val = $is_ones ? 1.0 : ((mt_rand() / mt_getrandmax()) * 2 - 1) * $scale;
        // 'e' erzwingt 32-bit Float in Little Endian (Plattformunabhängig & Nim-kompatibel)
        $buffer .= pack('g', $val); 
        
        if (($i + 1) % $chunk_size === 0) {
            fwrite($fp, $buffer);
            $buffer = ''; // Buffer leeren
        }
    }
    // Restlichen Buffer schreiben
    if (strlen($buffer) > 0) {
        fwrite($fp, $buffer);
    }
}

$start_time = microtime(true);

$dir = dirname($output_file);
if (!is_dir($dir) && $dir !== '.') {
    mkdir($dir, 0777, true);
}

// Binär-Datei zum Schreiben öffnen
$fp = fopen($output_file, 'wb');
if (!$fp) {
    die("❌ [Fehler] Konnte Datei nicht öffnen: $output_file\n");
}

echo "> Schreibe Header...\n";
// Magic String (auf 8 Bytes gepaddet)
fwrite($fp, str_pad('SNAI_V4', 8, "\0"));
// Architektur-Parameter als 32-bit Unsigned Integer (Little Endian, 'V')
fwrite($fp, pack('VVVVV', $vocab_size, $hidden_dim, $ffn_dim, $ctx_size, $num_layers));

echo "> Gieße Globale Tensoren (Embeddings & Output)...\n";
// 1. Embeddings: Xavier-Dämpfung
write_tensor($fp, $vocab_size * $hidden_dim, false, 1.0 / sqrt($hidden_dim));
// 2. KILLER-FIX: w_out mit 0.0001 (Softmax-Overflow-Schutz)
write_tensor($fp, $vocab_size * $hidden_dim, false, 0.0001);
// 3. rms_final
write_tensor($fp, $hidden_dim, true);

// Layer Tensoren streamen
for ($l = 0; $l < $num_layers; $l++) {
    echo "> Gieße Layer $l (QKV, Attention, FFN, RMSNorm)...\n";
    write_tensor($fp, $hidden_dim, true); // rms_att
    write_tensor($fp, $hidden_dim, true); // rms_ffn
    write_tensor($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_q
    write_tensor($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_k
    write_tensor($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_v
    write_tensor($fp, $hidden_dim * $hidden_dim, false, 1.0 / sqrt($hidden_dim)); // w_o
    write_tensor($fp, $ffn_dim * $hidden_dim, false, 0.02); // w_up
    write_tensor($fp, $hidden_dim * $ffn_dim, false, 0.02); // w_down
}

fclose($fp);

$duration = round(microtime(true) - $start_time, 2);
$mb = round(filesize($output_file) / 1024 / 1024, 2);

echo "✅ [Erfolg] Genesis-Cortex gegossen: $output_file ($mb MB) in $duration Sekunden.\n";
