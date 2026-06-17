<?php
// Dateiname: tensor_io.php
// Funktion: Behandelt das Laden (Header) und Speichern (Stream) von Tensormodellen.


define('SNAI_MAGIC', 'SNAI_V4');
define('SNAI_HEADER_SIZE', 28); // 8 Bytes Magic + 20 Bytes Parameter

/**
 * Liest NUR die Architektur-Metadaten aus dem Binär-Stream.
 * RAM-Verbrauch: Unter 1 Kilobyte!
 */
function load_model_header(string $filepath): ?array {
    if (!file_exists($filepath)) {
        echo "[I/O] Fatal: Datei nicht gefunden -> $filepath\n";
        return null;
    }

    // Early Exit, wenn die Datei nicht mal einen Header fassen kann
    if (filesize($filepath) < SNAI_HEADER_SIZE) {
        echo "[I/O] Fatal: Datei zu klein für einen Header -> $filepath\n";
        return null;
    }

    $fp = fopen($filepath, 'rb');
    if (!$fp) return null;

    // rtrim mit explizitem Null-Byte statt trim() (verhindert Binär-Korruption)
    $magic = rtrim(fread($fp, 8), "\0");
    if ($magic !== SNAI_MAGIC) {
        echo "[I/O] Fatal: Korruptes oder veraltetes Format in $filepath\n";
        fclose($fp);
        return null;
    }

    // 2. Die 5 Architektur-Parameter lesen (5 * 4 Bytes = 20 Bytes)
    $data = fread($fp, 20);
    fclose($fp);

    // Fix: Prüfen ob wirklich 20 Bytes gelesen wurden
    if (strlen($data) !== 20) {
        echo "[I/O] Fatal: Unvollständiger Header in $filepath\n";
        return null;
    }

    // 'V' decodiert 32-bit Unsigned Integer (Little Endian)
    $header = unpack('Vvocab_size/Vhidden_dim/Vffn_dim/Vctx_size/Vnum_layers', $data);
    
    return $header;
}

/**
 * Speichert ein extrahiertes Netzwerk-Array als reinen Binär-Stream 
 * auf die Festplatte (kompatibel mit cortex_load_stream).
 */
function save_model_stream(array $state, string $filepath): bool {
    // Basis-Validierung, um fatalen Array-Key-Fehlern vorzubeugen
    $required = ['vocab_size', 'hidden_dim', 'ctx_size', 'num_layers', 'tensors'];
    foreach ($required as $req) {
        if (!isset($state[$req])) {
            echo "[I/O] Error: Fehlender Key '$req' im State-Array.\n";
            return false;
        }
    }

    $start_time = microtime(true);

    $dir = dirname($filepath);
    
    // Fix: @-Operator verhindert Race Conditions bei parallelem mkdir durch Worker
    if (!is_dir($dir) && $dir !== '.') {
        @mkdir($dir, 0777, true);
    }

    $fp = fopen($filepath, 'wb');
    if (!$fp) return false;

    // flock schützt vor parallelen Schreibzugriffen
    flock($fp, LOCK_EX);

    // Fix: Dynamische FFN-Dim (Fallback auf hidden_dim * 4)
    $ffn_dim = $state['ffn_dim'] ?? ($state['hidden_dim'] * 4);

    // Header schreiben
    fwrite($fp, str_pad(SNAI_MAGIC, 8, "\0"));
    fwrite($fp, pack('VVVVV', 
        $state['vocab_size'], $state['hidden_dim'], 
        $ffn_dim, $state['ctx_size'], $state['num_layers']
    ));

    // Globale Tensoren
    $g = $state['tensors']['global'];
    fwrite($fp, $g['embeddings']);
    fwrite($fp, $g['w_out']);
    fwrite($fp, $g['rms_final']);

    // Layer Tensoren
    for ($l = 0; $l < $state['num_layers']; $l++) {
        $ly = $state['tensors']['layers'][$l];
        fwrite($fp, $ly['rms_att']);
        fwrite($fp, $ly['rms_ffn']);
        fwrite($fp, $ly['w_q']);
        fwrite($fp, $ly['w_k']);
        fwrite($fp, $ly['w_v']);
        fwrite($fp, $ly['w_o']);
        fwrite($fp, $ly['w_up']);
        fwrite($fp, $ly['w_down']);
    }

    flock($fp, LOCK_UN);
    fclose($fp);

    return true;
}