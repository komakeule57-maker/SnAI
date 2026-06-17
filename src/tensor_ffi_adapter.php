<?php
// Dateiname: tensor_ffi_adapter.php
// Funktion: FFI Adapter & Cortex Bridge zur Nim-Engine

class SnaiCortexBridge {
    public $ffi;
    public $state_ptr;
    public $vocab_size, $hidden_dim, $ctx_size, $num_layers, $ffn_dim, $num_heads;

    private $global_map = [
        0 => ['w' => 'embeddings', 'm' => 'm_embeddings', 'v' => 'v_embeddings'],
        1 => ['w' => 'w_out', 'm' => 'm_w_out', 'v' => 'v_w_out'],
        2 => ['w' => 'rms_final', 'm' => 'm_rms_final', 'v' => 'v_rms_final']
    ];

    private $layer_map = [
        10 => ['w' => 'rms_att', 'm' => 'm_rms_att', 'v' => 'v_rms_att'],
        11 => ['w' => 'rms_ffn', 'm' => 'm_rms_ffn', 'v' => 'v_rms_ffn'],
        12 => ['w' => 'w_q', 'm' => 'm_w_q', 'v' => 'v_w_q'],
        13 => ['w' => 'w_k', 'm' => 'm_w_k', 'v' => 'v_w_k'],
        14 => ['w' => 'w_v', 'm' => 'm_w_v', 'v' => 'v_w_v'],
        15 => ['w' => 'w_o', 'm' => 'm_w_o', 'v' => 'v_w_o'],
        16 => ['w' => 'w_up', 'm' => 'm_w_up', 'v' => 'v_w_up'],
        17 => ['w' => 'w_down', 'm' => 'm_w_down', 'v' => 'v_w_down']
    ];

    // Caching für FFI Buffer
    private $cached_c_logits = null;

     /**
     * Lädt das Modell direkt über Nim von der Festplatte.
     */
    public function __construct(string $filepath, int $num_heads = 1) {
        $lib_path = __DIR__ . '/../lib/libcortex.so';
        
        $this->ffi = FFI::cdef("
            void* cortex_init(int vocab_size, int hidden_dim, int ctx_size, int num_layers, int num_heads);
            void* cortex_load_stream(const char* filepath, int num_heads);
            float cortex_train_step(void* state, int* raw_inputs, int* raw_targets, int num_tokens, float lr);
            void cortex_free(void* state);
            void cortex_reset_cache(void* state);
            void cortex_forward(void* state, int* raw_tokens, int num_tokens, float* raw_logits);
            void cortex_extract_layer(void* state, int layer_idx, int tensor_type, float* raw_weights, int num_weights);
            void cortex_extract_adam_layer(void* state, int layer_idx, int tensor_type, bool is_variance, float* raw, int size);
            void cortex_load_layer(void* state, int layer_idx, int tensor_type, float* raw_weights, int num_weights);
            
            // ---(OpenCL & Utils) ---
            void cortex_init_gpu();
            int get_vocab_size(void* state);
            int get_ctx_size(void* state);
        ", $lib_path);

        require_once __DIR__ . '/tensor_io.php';
        $header = load_model_header($filepath);
        if (!$header) die("Fatal: Konnte Modell-Header nicht lesen.\n");

        $this->vocab_size = $header['vocab_size'];
        $this->hidden_dim = $header['hidden_dim'];
        $this->ctx_size   = $header['ctx_size'];
        $this->num_layers = $header['num_layers'];
        $this->ffn_dim    = $header['ffn_dim'];
        $this->num_heads  = $num_heads;

        $this->state_ptr = $this->ffi->cortex_load_stream($filepath, $this->num_heads);

        if ($this->state_ptr === null) {
            die("Fatal: Nim C-Core konnte den Binär-Stream nicht laden!\n");
        }

        // Buffer einmalig allokieren
        $this->cached_c_logits = $this->ffi->new("float[{$this->vocab_size}]");
    }

    public function init_gpu() {
        $this->ffi->cortex_init_gpu();
    }

    public function __destruct() {
        if ($this->state_ptr !== null) {
            try { $this->ffi->cortex_free($this->state_ptr); } catch (Throwable $e) {}
            $this->state_ptr = null;
        }
    }

    public function extract_state(): array {
        $state = [
            'magic' => 'SNAI_V4',
            'vocab_size' => $this->vocab_size, 'hidden_dim' => $this->hidden_dim,
            'ctx_size' => $this->ctx_size, 'num_layers' => $this->num_layers,
            'num_heads' => $this->num_heads,
            'tensors' => ['global' => [], 'layers' => []]
        ];

        $size_emb = $this->vocab_size * $this->hidden_dim;
        $size_rms = $this->hidden_dim;
        $size_qkv = $this->hidden_dim * $this->hidden_dim;
        $size_up = $this->ffn_dim * $this->hidden_dim;
        
        $sizes = [
            0 => $size_emb, 1 => $size_emb, 2 => $size_rms,
            10 => $size_rms, 11 => $size_rms,
            12 => $size_qkv, 13 => $size_qkv, 14 => $size_qkv, 15 => $size_qkv,
            16 => $size_up, 17 => $size_up
        ];

        foreach ($this->global_map as $type => $keys) {
            $c_array = $this->ffi->new("float[{$sizes[$type]}]");
            $this->ffi->cortex_extract_layer($this->state_ptr, -1, $type, $this->ffi->cast("float*", $c_array), $sizes[$type]);
            $state['tensors']['global'][$keys['w']] = FFI::string($c_array, $sizes[$type] * 4);
        }

        for ($l = 0; $l < $this->num_layers; $l++) {
            $layer = [];
            foreach ($this->layer_map as $type => $keys) {
                $c_array = $this->ffi->new("float[{$sizes[$type]}]");
                $this->ffi->cortex_extract_layer($this->state_ptr, $l, $type, $this->ffi->cast("float*", $c_array), $sizes[$type]);
                $layer[$keys['w']] = FFI::string($c_array, $sizes[$type] * 4);
            }
            $state['tensors']['layers'][$l] = $layer;
        }

        return $state;
    }

    public function train_step(array $inputs, array $targets, float $lr): float {
        $num_tokens = count($inputs);
        if ($num_tokens > $this->ctx_size) {
            $num_tokens = $this->ctx_size;
            $inputs = array_slice($inputs, -$num_tokens);
            $targets = array_slice($targets, -$num_tokens);
        }
        
        if ($num_tokens === 0) return 0.0;
        
        $c_in = $this->ffi->new("int[$num_tokens]");
        $c_tg = $this->ffi->new("int[$num_tokens]");
        foreach ($inputs as $i => $v) $c_in[$i] = $v;
        foreach ($targets as $i => $v) $c_tg[$i] = $v;
        
        return $this->ffi->cortex_train_step($this->state_ptr, $this->ffi->cast("int*", $c_in), $this->ffi->cast("int*", $c_tg), $num_tokens, $lr);
    }

    public function generate_next_token(array $input_tokens, float $temp = 0.7, int $top_k = 40, float $top_p = 0.9, float $rep_pen = 1.15): int {
        $num = count($input_tokens);
        if ($num > $this->ctx_size) {
            $input_tokens = array_slice($input_tokens, -$this->ctx_size);
            $num = $this->ctx_size;
        }

        $c_in = $this->ffi->new("int[$num]");
        foreach ($input_tokens as $i => $v) $c_in[$i] = $v;
        
        // VETO-FIX: Verwendet gecachten Array-Pointer gegen GC-Thrashing
        $c_logits = $this->cached_c_logits;
        
        $this->ffi->cortex_forward($this->state_ptr, $this->ffi->cast("int*", $c_in), $num, $this->ffi->cast("float*", $c_logits));
        
        $logits = [];
        $input_set = array_flip($input_tokens);
        
        for ($i = 0; $i < $this->vocab_size; $i++) {
            $val = $c_logits[$i];
            if (isset($input_set[$i])) $val -= $rep_pen;
            $logits[$i] = $val;
        }
        
        arsort($logits);
        $logits = array_slice($logits, 0, $top_k, true);
        
        if ($temp < 0.01) return array_key_first($logits);
        
        $probs = []; $sum = 0.0;
        foreach ($logits as $id => $val) {
            $p = exp($val / $temp); 
            $probs[$id] = $p;
            $sum += $p;
        }
        
        $cum = 0.0; $filtered = [];
        foreach ($probs as $id => $p) {
            $p_norm = $p / $sum; 
            $filtered[$id] = $p_norm;
            $cum += $p_norm;
            if ($cum >= $top_p) break;
        }

        $r = (mt_rand() / mt_getrandmax()) * $cum; 
        
        $cum_check = 0.0;
        foreach ($filtered as $id => $p) {
            $cum_check += $p;
            if ($r <= $cum_check) return $id;
        }
        return array_key_first($filtered);
    }
}

// Fallback Alias für alte Skripte
if (!class_exists('CortexBridge')) {
    class_alias('SnaiCortexBridge', 'CortexBridge');
}

// Eigenständige FFI Averaging Funktion für train_cluster.php
if (!function_exists('ffi_average_layer')) {
    function ffi_average_layer(array $binary_strings, int $dtype = 1): string {
        if (empty($binary_strings)) return "";
        $num_models = count($binary_strings);
        if ($num_models === 1) return $binary_strings[0];
        
        $num_floats = intval(strlen($binary_strings[0]) / 4);
        $sum = array_fill(0, $num_floats, 0.0);
        
        foreach ($binary_strings as $bin) {
            $floats = array_values(unpack("g*", $bin));
            for ($i = 0; $i < $num_floats; $i++) {
                $sum[$i] += $floats[$i] ?? 0.0;
            }
        }
        for ($i = 0; $i < $num_floats; $i++) {
            $sum[$i] /= $num_models;
        }
        return pack("g*", ...$sum);
    }
}