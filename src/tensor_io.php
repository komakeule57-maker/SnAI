<?php
// Dateiname: src/tensor_io.php
// Funktion: Behandelt das Laden (Header) und Speichern (Stream) von Tensormodellen (Zero OOM).

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Tensor I/O (Phase VIII - The Adam Persistence Update)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. PHP liest nur noch den 28-Byte Header.
 * AUDIT-FIX: rtrim für Magic-Bytes, Filesize-Precheck, Race-Condition bei mkdir.
 * UPDATE (Phase VIII): Magic-Byte auf SNAI_V5 angehoben. Speichert nun neben 
 * den Gewichten (w) auch nahtlos das Momentum (m) und die Varianz (v) von Adam!
 */

// Neuer Magic-String trennt V4-Gewichte von V5-Gedächtnis-Modellen
define('SNAI_MAGIC', 'SNAI_V5'); 
define('SNAI_HEADER_SIZE', 28); // 8 Bytes Magic + 20 Bytes Parameter

/**
 * Liest NUR die Architektur-Metadaten aus dem Binär-Stream.
 * RAM-Verbrauch: Unter 1 Kilobyte!
 */
function load_model_header(string $filepath): ?array {
    if (!file_exists($filepath)) {
        echo "🚨 [I/O] Fatal: Datei nicht gefunden -> $filepath\n";
        return null;
    }

    if (filesize($filepath) < SNAI_HEADER_SIZE) {
        echo "🚨 [I/O] Fatal: Datei zu klein für einen Header -> $filepath\n";
        return null;
    }

    $fp = fopen($filepath, 'rb');
    if (!$fp) return null;

    $magic = rtrim(fread($fp, 8), "\0");
    if ($magic !== SNAI_MAGIC) {
        echo "🚨 [I/O] Fatal: Korruptes oder veraltetes Format in $filepath (Erwartet: " . SNAI_MAGIC . ", Gefunden: $magic)\n";
        fclose($fp);
        return null;
    }

    $data = fread($fp, 20);
    fclose($fp);

    if (strlen($data) !== 20) {
        echo "🚨 [I/O] Fatal: Unvollständiger Header in $filepath\n";
        return null;
    }

    $header = unpack('Vvocab_size/Vhidden_dim/Vffn_dim/Vctx_size/Vnum_layers', $data);
    
    return $header;
}

/**
 * Laedt ein KOMPLETTES Modell (Header + alle Tensoren) in ein PHP-Array.
 * Spiegelbild von save_model_stream() - liest exakt die Blob-Reihenfolge, die
 * dort geschrieben wird. Erkennt automatisch, ob die Datei Adam-Zustand (m, v)
 * enthaelt (SNAI_V5-Vollformat) oder nur Gewichte (Legacy-Format); im Legacy-
 * Fall fehlen die "m_"- und "v_"-Praefix-Schluessel im Rueckgabe-Array komplett, statt sie
 * mit Nullen aufzufuellen (spart RAM, Aufrufer die nur 'w' brauchen wie
 * snai_xray.php merken davon nichts).
 *
 * AUDIT-FIX: Diese Funktion fehlte komplett, obwohl snai_xray.php sie aufruft
 * (Fatal Error: Call to undefined function). royal_api.php hatte denselben
 * Aufruf, ist aber mittlerweile geloescht (unbenutzt, aus anderen Gruenden
 * kaputt).
 *
 * WARNUNG: Laedt das GESAMTE Modell in den PHP-Heap (kein OOM-Zero-Stream wie
 * cortex_load_stream in der Nim-Engine) - nur fuer Diagnose-Tools gedacht,
 * niemals im heissen Trainings-/Inferenz-Pfad verwenden.
 */
function load_model(string $filepath): ?array {
    if (!file_exists($filepath)) {
        echo "🚨 [I/O] Fatal: Datei nicht gefunden -> $filepath\n";
        return null;
    }

    $fp = fopen($filepath, 'rb');
    if (!$fp) return null;

    $magic = rtrim(fread($fp, 8), "\0");
    if ($magic !== SNAI_MAGIC) {
        echo "🚨 [I/O] Fatal: Korruptes oder veraltetes Format in $filepath (Erwartet: " . SNAI_MAGIC . ", Gefunden: $magic)\n";
        fclose($fp);
        return null;
    }

    $header_data = fread($fp, 20);
    if (strlen($header_data) !== 20) {
        echo "🚨 [I/O] Fatal: Unvollständiger Header in $filepath\n";
        fclose($fp);
        return null;
    }
    $h = unpack('Vvocab_size/Vhidden_dim/Vffn_dim/Vctx_size/Vnum_layers', $header_data);

    $vocab  = $h['vocab_size'];
    $dim    = $h['hidden_dim'];
    $ffn    = $h['ffn_dim'];
    $ctx    = $h['ctx_size'];
    $layers = $h['num_layers'];

    $size_emb = $vocab * $dim;
    $size_rms = $dim;
    $size_qkv = $dim * $dim;
    $size_up  = $ffn * $dim;

    $elems_per_layer = (2 * $size_rms) + (4 * $size_qkv) + (2 * $size_up);
    $elems_total = (2 * $size_emb) + $size_rms + ($layers * $elems_per_layer);
    $w_bytes = $elems_total * 4;

    $payload = filesize($filepath) - SNAI_HEADER_SIZE;

    if ($payload === $w_bytes * 3 || $payload === $w_bytes * 3 + 4) {
        $with_adam = true;
        $has_adam_t_trailer = ($payload === $w_bytes * 3 + 4);
    } elseif ($payload === $w_bytes || $payload === $w_bytes + 4) {
        $with_adam = false;
        $has_adam_t_trailer = ($payload === $w_bytes + 4);
    } else {
        echo "🚨 [I/O] Fatal: Groessen-Mismatch in $filepath. Payload=" . number_format($payload) .
             " B. Erwartet: " . number_format($w_bytes) . " (nur W) oder " . number_format($w_bytes * 3) . " (W+m+v).\n";
        fclose($fp);
        return null;
    }

    // Liest ein Tensor (Gewicht + optional Momentum/Varianz) und haengt die
    // vorhandenen Bloecke unter $prefix/m_$prefix/v_$prefix in $bucket ein.
    $read_tensor = function(string $prefix, int $floats) use ($fp, $with_adam, &$bucket) {
        $bytes = $floats * 4;
        $bucket[$prefix] = fread($fp, $bytes);
        if ($with_adam) {
            $bucket['m_' . $prefix] = fread($fp, $bytes);
            $bucket['v_' . $prefix] = fread($fp, $bytes);
        }
    };

    $bucket = [];
    $read_tensor('embeddings', $size_emb);
    $read_tensor('w_out', $size_emb);
    $read_tensor('rms_final', $size_rms);
    $global = $bucket;

    $layer_tensors = [];
    for ($l = 0; $l < $layers; $l++) {
        $bucket = [];
        $read_tensor('rms_att', $size_rms);
        $read_tensor('rms_ffn', $size_rms);
        $read_tensor('w_q', $size_qkv);
        $read_tensor('w_k', $size_qkv);
        $read_tensor('w_v', $size_qkv);
        $read_tensor('w_o', $size_qkv);
        $read_tensor('w_up', $size_up);
        $read_tensor('w_down', $size_up);
        $layer_tensors[$l] = $bucket;
    }

    // AUDIT-FIX (2026-08-14, im Rahmen des Neuron-Revival-Live-Hooks entdeckt):
    // der Adam-Bias-Correction-Schrittzaehler (adam_t, optionaler int32-Trailer
    // - siehe save_model_stream()) wurde bisher nur zur With-Adam-Erkennung per
    // Dateigroesse benutzt, aber nie selbst gelesen/zurueckgegeben. Jeder
    // load_model() -> save_model_stream() Rundlauf (z.B. snai_neuron_revival.php)
    // schrieb dadurch STILLSCHWEIGEND adam_t=0 zurueck, obwohl m/v "warm" sind -
    // cortex.nim faengt diese Inkonsistenz zwar mit einem WARN+Fallback
    // (adam_t=1000) ab, das setzt aber bei jedem Rundlauf erneut die effektive
    // Bias-Correction fuer ALLE Parameter zurueck, nicht nur die absichtlich
    // veraenderten Zeilen - ein Seiteneffekt, der nur durch den Fallback
    // verdeckt, nie behoben wurde.
    $adam_t = 0;
    if ($has_adam_t_trailer) {
        $trailer = fread($fp, 4);
        if (strlen($trailer) === 4) {
            $adam_t = unpack('V', $trailer)[1];
        }
    }

    fclose($fp);

    return [
        'magic' => SNAI_MAGIC,
        'vocab_size' => $vocab, 'hidden_dim' => $dim, 'ffn_dim' => $ffn,
        'ctx_size' => $ctx, 'num_layers' => $layers,
        'with_adam' => $with_adam, 'adam_t' => $adam_t,
        'tensors' => ['global' => $global, 'layers' => $layer_tensors],
    ];
}

/**
 * Speichert ein extrahiertes Netzwerk-Array samt Adam-Historie als reinen Binär-Stream.
 */
function save_model_stream(array $state, string $filepath): bool {
    $required = ['vocab_size', 'hidden_dim', 'ctx_size', 'num_layers', 'tensors'];
    foreach ($required as $req) {
        if (!isset($state[$req])) {
            echo "🚨 [I/O] Error: Fehlender Key '$req' im State-Array.\n";
            return false;
        }
    }

    $dir = dirname($filepath);
    
    if (!is_dir($dir) && $dir !== '.') {
        @mkdir($dir, 0777, true);
    }

    $fp = fopen($filepath, 'wb');
    if (!$fp) return false;

    flock($fp, LOCK_EX);

    $ffn_dim = $state['ffn_dim'] ?? ($state['hidden_dim'] * 4);

    // Header schreiben (SNAI_V5)
    fwrite($fp, str_pad(SNAI_MAGIC, 8, "\0"));
    fwrite($fp, pack('VVVVV', 
        $state['vocab_size'], $state['hidden_dim'], 
        $ffn_dim, $state['ctx_size'], $state['num_layers']
    ));

    // --- GLOBALE TENSOREN (w, m, v interleaved) ---
    $g = $state['tensors']['global'];
    
    fwrite($fp, $g['embeddings']); fwrite($fp, $g['m_embeddings']); fwrite($fp, $g['v_embeddings']);
    fwrite($fp, $g['w_out']);      fwrite($fp, $g['m_w_out']);      fwrite($fp, $g['v_w_out']);
    fwrite($fp, $g['rms_final']);  fwrite($fp, $g['m_rms_final']);  fwrite($fp, $g['v_rms_final']);

    // --- LAYER TENSOREN (w, m, v interleaved) ---
    for ($l = 0; $l < $state['num_layers']; $l++) {
        $ly = $state['tensors']['layers'][$l];
        
        fwrite($fp, $ly['rms_att']);  fwrite($fp, $ly['m_rms_att']);  fwrite($fp, $ly['v_rms_att']);
        fwrite($fp, $ly['rms_ffn']);  fwrite($fp, $ly['m_rms_ffn']);  fwrite($fp, $ly['v_rms_ffn']);
        
        fwrite($fp, $ly['w_q']);      fwrite($fp, $ly['m_w_q']);      fwrite($fp, $ly['v_w_q']);
        fwrite($fp, $ly['w_k']);      fwrite($fp, $ly['m_w_k']);      fwrite($fp, $ly['v_w_k']);
        fwrite($fp, $ly['w_v']);      fwrite($fp, $ly['m_w_v']);      fwrite($fp, $ly['v_w_v']);
        fwrite($fp, $ly['w_o']);      fwrite($fp, $ly['m_w_o']);      fwrite($fp, $ly['v_w_o']);
        
        fwrite($fp, $ly['w_up']);     fwrite($fp, $ly['m_w_up']);     fwrite($fp, $ly['v_w_up']);
        fwrite($fp, $ly['w_down']);   fwrite($fp, $ly['m_w_down']);   fwrite($fp, $ly['v_w_down']);
    }

    // Update 2026-08-09 ("Engine-/Framework-Verbesserungen", Punkt 1):
    // Adam-Bias-Correction-Schrittzaehler als optionaler int32-Trailer -
    // der Nim-Loader erkennt ihn rein anhand der Dateigroesse (Payload =
    // W+m+v-Bytes + genau 4 zusaetzliche Bytes, siehe cortex.nim "withT").
    // Kein neues Format, keine Versions-Aenderung - nur der bisher fehlende
    // letzte Baustein eines bereits vom Loader unterstuetzten Layouts.
    // Default 0 (= "kein Adam-State"/Cold-Start-Verhalten), falls
    // 'adam_t' im State-Array fehlt - haelt bestehende Aufrufer sicher.
    fwrite($fp, pack('V', $state['adam_t'] ?? 0));

    flock($fp, LOCK_UN);
    fclose($fp);

    return true;
}