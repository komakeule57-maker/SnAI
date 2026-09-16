<?php
/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Cortex Bridge (Phase V - OOM Zero Inference)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Kapselt die C-Library für die pure, rasend schnelle Inferenz.
 * FIX: 'cdata_hold' und 'load_layer' komplett eliminiert.
 * FIX: Nutzt 'cortex_load_stream', sodass der Webserver 0 MB RAM für Gewichte benötigt!
 * AUDIT-V9.4: FFI-GC-Shredder eliminiert (Persistent Buffers). Context-Overflow Guard. TPS-Boost.
 */

require_once 'tensor_io.php'; // Für den 28-Byte Header Reader

// AUDIT-FIX: siehe snai_inference.php - libgomp-Exit-Race bei kurzlebigen Prozessen
// nach einem Forward-Pass. Muss vor dem FFI::cdef()-Laden von libcortex.so stehen.
putenv('OMP_WAIT_POLICY=PASSIVE');

class CortexBridge {
    private $ffi;
    private $state;
    public $vocab_size;
    public $ctx_size;
    
    // VETO-FIX: Persistente FFI-Buffer, um Allokationen im Hot-Loop (GC-Thrashing) zu verhindern
    private $c_tokens_buffer;
    private $c_logits_buffer;

    /**
     * @param string $filepath Pfad zur .snai oder .apex Datei
     * @param int $heads Attention-Heads (Standard 1, Matriarchin z.B. 2)
     */
    public function __construct(string $filepath, int $heads = 1) {
        if (!file_exists($filepath)) {
            die("🚨 FATAL: Cortex-Datei nicht gefunden -> $filepath\n");
        }

        // 1. PHP liest nur noch die Metadaten
        $header = load_model_header($filepath);
        if (!$header) {
            die("🚨 FATAL: Konnte Modell-Header nicht lesen. Datei korrupt?\n");
        }

        // Metadaten übernehmen
        $this->vocab_size = $header['vocab_size'];
        $this->ctx_size = $header['ctx_size'];

        // C-Header angepasst auf den neuen Stream-Injector
        $c_header = <<<C
            void* cortex_load_stream(const char* filepath, int num_heads);
            void cortex_reset_cache(void* state);
            void cortex_free(void* state);
            void cortex_forward(void* state, int* raw_tokens, int num_tokens, float* raw_logits);
C;
        try {
            $lib_path = __DIR__ . '/../lib/libcortex.so'; 
            $this->ffi = FFI::cdef($c_header, $lib_path);
        } catch (Exception $e) {
            die("🚨 FATAL FFI Error: Konnte libcortex.so nicht laden. Kompiliert?\n");
        }

        // 2. DER ZERO-RAM LADEVORGANG
        $this->state = $this->ffi->cortex_load_stream($filepath, $heads);
        
        if ($this->state === null) {
            die("🚨 FATAL: C-Core konnte den Binär-Stream nicht laden!\n");
        }

        // VETO-FIX: Puffer EINMALIG allokieren. Die FFI-Bridge wird nicht mehr im Loop belastet.
        $this->c_tokens_buffer = $this->ffi->new("int[{$this->ctx_size}]");
        $this->c_logits_buffer = $this->ffi->new("float[{$this->vocab_size}]");
    }

    public function __destruct() {
        if ($this->state !== null) {
            try {
                $this->ffi->cortex_free($this->state);
                $this->state = null;
            } catch (Throwable $e) {}
        }
    }

    public function reset_context() {
        $this->ffi->cortex_reset_cache($this->state);
    }

    /**
     * Führt den Forward Pass aus und wählt das nächste Token intelligent aus.
     */
    public function generate_next_token(array $tokens, float $temperature = 0.7, int $top_k = 40, float $top_p = 0.9, float $repetition_penalty = 1.1): int {
        // VETO-FIX: Context-Overflow absichern, bevor C crasht
        $num = count($tokens);
        if ($num > $this->ctx_size) {
            $tokens = array_slice($tokens, -$this->ctx_size);
            $num = $this->ctx_size;
        }
        if ($num === 0) return 0;

        // Pointer auf persistente Buffer holen
        $c_tokens = $this->c_tokens_buffer;
        $c_logits = $this->c_logits_buffer;

        // Token überschreiben (O(N) ist in PHP schnell genug für Arrays < CtxSize)
        foreach ($tokens as $i => $t) {
            $c_tokens[$i] = $t;
        }

        // Execute Inferenz
        $this->ffi->cortex_forward($this->state, $this->ffi->cast("int*", $c_tokens), $num, $this->ffi->cast("float*", $c_logits));

        // 1. REPETITION PENALTY
        if ($repetition_penalty !== 1.0) {
            $unique_tokens = array_unique($tokens);
            foreach ($unique_tokens as $t_id) {
                if ($t_id < $this->vocab_size) {
                    if ($c_logits[$t_id] < 0) {
                        $c_logits[$t_id] *= $repetition_penalty;
                    } else {
                        $c_logits[$t_id] /= $repetition_penalty;
                    }
                }
            }
        }

        // 2. ZERO-TEMPERATURE FALLBACK (Greedy)
        if ($temperature <= 0.0) {
            $best_id = 0;
            $max_val = -99999.0;
            for ($i = 0; $i < $this->vocab_size; $i++) {
                if ($c_logits[$i] > $max_val) {
                    $max_val = $c_logits[$i];
                    $best_id = $i;
                }
            }
            return $best_id;
        }

        // 3. TEMPERATURE SCALING & SOFTMAX (Optimierte Einzelschleife)
        $logits = [];
        $max_logit = -99999.0;
        for ($i = 0; $i < $this->vocab_size; $i++) {
            $val = $c_logits[$i] / $temperature;
            $logits[$i] = $val;
            if ($val > $max_logit) $max_logit = $val;
        }

        $sum_exp = 0.0;
        $probs = [];
        foreach ($logits as $i => $val) {
            $p = exp($val - $max_logit);
            $probs[$i] = $p;
            $sum_exp += $p;
        }
        
        // Direkte Normalisierung im Array
        foreach ($probs as $i => $p) {
            $probs[$i] = $p / $sum_exp;
        }

        arsort($probs, SORT_NUMERIC);

        // 4. TOP-K & TOP-P SAMPLING
        $filtered_probs = [];
        $cumulative_p = 0.0;
        $k = 0;

        foreach ($probs as $id => $p) {
            if ($k >= $top_k) break; 
            
            $filtered_probs[$id] = $p;
            $cumulative_p += $p;
            $k++;

            if ($cumulative_p >= $top_p) break;
        }

        $r = (mt_rand() / mt_getrandmax()) * $cumulative_p;
        $acc = 0.0;
        
        foreach ($filtered_probs as $id => $p) {
            $acc += $p;
            if ($r <= $acc) {
                return $id;
            }
        }

        return array_key_first($filtered_probs);
    }
}