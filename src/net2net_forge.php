<?php
// Dateiname: net2net_forge.php
// Funktion: CLI-Werkzeug. Verschmilzt Tensoren und erweitert sie durch Network Morphism.

class Net2NetForge {
    private const CHUNK_SIZE_BYTES = 1048576;
    private const NOISE_FACTOR = 0.005;

    private function robust_fread($fp, $length) {
        $data = '';
        while (strlen($data) < $length && !feof($fp)) {
            $chunk = fread($fp, $length - strlen($data));
            if ($chunk === false || $chunk === '') break;
            $data .= $chunk;
        }
        return $data;
    }

    public function merge_models(array $files, string $output_path): bool {
        $num_models = count($files);
        if ($num_models < 2) {
            echo "[Forge] Fehler: Zum Verschmelzen werden mindestens 2 Modelle benötigt.\n";
            return false;
        }

        $handles = [];
        foreach ($files as $file) {
            $fp = @fopen($file, 'rb');
            if (!$fp) {
                echo "[Forge] Fatal: Kann Tribut nicht lesen: $file\n";
                return false;
            }
            $handles[] = $fp;
        }

        $out_fp = fopen($output_path, 'wb');
        flock($out_fp, LOCK_EX);

        $header = fread($handles[0], 28);
        fwrite($out_fp, $header);

        for ($i = 1; $i < $num_models; $i++) {
            $h = fread($handles[$i], 28);
            if ($h !== $header) {
                echo "[Forge] Fatal: Inkompatible Header! Matrix-Dimensionen stimmen nicht überein.\n";
                foreach ($handles as $hndl) fclose($hndl);
                fclose($out_fp);
                return false;
            }
        }

        echo "[Forge] Verschmelze $num_models Tensoren (OOM Zero Stream)...\n";

        while (!feof($handles[0])) {
            $chunks = [];
            for ($i = 0; $i < $num_models; $i++) {
                $chunks[] = $this->robust_fread($handles[$i], self::CHUNK_SIZE_BYTES);
            }

            if (strlen($chunks[0]) === 0) break;

            $num_floats_read = intval(strlen($chunks[0]) / 4);
            $arrays = [];
            for ($i = 0; $i < $num_models; $i++) {
                $arrays[] = array_values(unpack("g*", $chunks[$i]));
            }

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
        flock($out_fp, LOCK_UN);
        fclose($out_fp);
        
        echo "[Forge] Verschmelzung erfolgreich: $output_path\n";
        return true;
    }

    private function upscale_matrix_stream($in_fp, $out_fp, int $old_r, int $old_c, int $new_r, int $new_c, float $scale = 1.0) {
        $expected_bytes = $old_r * $old_c * 4;
        $old_bin = $this->robust_fread($in_fp, $expected_bytes);
        
        if ($old_bin === false || strlen($old_bin) !== $expected_bytes) {
            die("[Forge] Fehler: Inkomplette Matrix-Struktur gelesen.\n");
        }
        
        $old_floats = array_values(unpack("g*", $old_bin));
        $buffer = [];
        $buf_size = 8192; 
        $new_bin = '';

        for ($r = 0; $r < $new_r; $r++) {
            for ($c = 0; $c < $new_c; $c++) {
                if ($r < $old_r && $c < $old_c) {
                    $idx = ($r * $old_c) + $c;
                    $val = $old_floats[$idx] * $scale;
                } else {
                    $val = ((mt_rand() / mt_getrandmax()) * 2 - 1) * self::NOISE_FACTOR; 
                }
                
                $buffer[] = $val;
                if (count($buffer) >= $buf_size) {
                    $new_bin .= pack("g*", ...$buffer);
                    $buffer = [];
                }
            }
        }
        
        if (!empty($buffer)) $new_bin .= pack("g*", ...$buffer);
        fwrite($out_fp, $new_bin);
    }

    public function upscale_model(string $input_path, string $output_path, int $target_dim): bool {
        if (!file_exists($input_path)) {
            echo "[Forge] Eingabedatei nicht gefunden: $input_path\n";
            return false;
        }

        $in_fp = fopen($input_path, 'rb');
        $out_fp = fopen($output_path, 'wb');
        if (!$in_fp || !$out_fp) return false;

        $magic = $this->robust_fread($in_fp, 8);
        $h_data = $this->robust_fread($in_fp, 20);
        $p = unpack('Vvocab/Vdim/Vffn/Vctx/Vlayers', $h_data);
        
        $vocab = $p['vocab'];
        $old_dim = $p['dim'];
        $old_ffn = $p['ffn'];
        $old_ctx = $p['ctx'];
        $old_layers = $p['layers'];

        if ($target_dim <= $old_dim) {
            echo "[Forge] Ziel-Dimension ($target_dim) muss größer sein als Ursprung ($old_dim).\n";
            return false;
        }

        $new_dim = $target_dim;
        $new_ffn = $new_dim * 4; 
        $new_ctx = $old_ctx * 2; 
        $new_layers = $old_layers * 2; 

        echo "[Forge] Initiiere Network Morphism...\n";
        echo "  > Dim: $old_dim -> $new_dim | Ctx: $old_ctx -> $new_ctx | Lyr: $old_layers -> $new_layers\n";

        fwrite($out_fp, $magic);
        fwrite($out_fp, pack('VVVVV', $vocab, $new_dim, $new_ffn, $new_ctx, $new_layers));

        $this->upscale_matrix_stream($in_fp, $out_fp, $vocab, $old_dim, $vocab, $new_dim, 1.0); 
        $this->upscale_matrix_stream($in_fp, $out_fp, $vocab, $old_dim, $vocab, $new_dim, 1.0); 
        $this->upscale_matrix_stream($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0); 

        $layer_0_offset = ftell($in_fp);
        $layer_byte_size = ((2 * $old_dim) + (4 * ($old_dim * $old_dim)) + (2 * ($old_dim * $old_ffn))) * 4;

        for ($l = 0; $l < $new_layers; $l++) {
            $src_l = $l % $old_layers;
            fseek($in_fp, $layer_0_offset + ($src_l * $layer_byte_size));

            $is_new_layer = ($l >= $old_layers);
            $out_scale = $is_new_layer ? 0.01 : 1.0;

            $this->upscale_matrix_stream($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, $out_scale); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_ffn, $old_dim, $new_ffn, $new_dim, 1.0); 
            $this->upscale_matrix_stream($in_fp, $out_fp, $old_dim, $old_ffn, $new_dim, $new_ffn, $out_scale); 
        }
        
        fclose($in_fp);
        fclose($out_fp);
        
        echo "[Forge] Expansion erfolgreich: $output_path\n";
        return true;
    }
}

// --- CLI BOOTSTRAP ---
if (php_sapi_name() === 'cli' && basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    $forge = new Net2NetForge();
    $cmd = $argv[1] ?? '';

    if ($cmd === 'merge' && $argc >= 5) {
        $output = $argv[2];
        $inputs = array_slice($argv, 3);
        $forge->merge_models($inputs, $output);
    } elseif ($cmd === 'upscale' && $argc === 5) {
        $input = $argv[2];
        $output = $argv[3];
        $dim = (int)$argv[4];
        $forge->upscale_model($input, $output, $dim);
    } else {
        echo "Verwendung:\n";
        echo "  php net2net_forge.php merge <output.snai> <in1.snai> <in2.snai> ...\n";
        echo "  php net2net_forge.php upscale <input.snai> <output.snai> <ziel_dim>\n";
    }
}