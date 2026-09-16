<?php
// Dateiname: src/tensor_ffi_adapter.php
// Funktion: FFI Adapter & Cortex Bridge zur Nim-Engine (Memory Edition)

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: FFI Adapter & Cortex Bridge (Phase VIII - The Adam Persistence Update)
 * -------------------------------------------------------------------------
 * Info: Enthält OOM-Sicherheits-Checks & O(1) Speed-Boosts aus Audit-Phase.
 * AUDIT-FIX (V10): Adam-State (m, v) Extraktion vollständig implementiert!
 * AUDIT-FIX (V10.1): FFI::isNull() Guard verhindert stille Segfaults.
 * AUDIT-FIX (V10.2): Numerische Softmax-Stabilisierung gegen Float-Overflows.
 * AUDIT-FIX: OMP_WAIT_POLICY=PASSIVE gegen libgomp-Exit-Race (siehe snai_inference.php).
 *   Muss vor jedem FFI::cdef()-Laden von libcortex.so gesetzt sein, deshalb hier
 *   auf Datei-Ebene statt im Konstruktor - require_once laeuft garantiert zuerst.
 */

putenv('OMP_WAIT_POLICY=PASSIVE');

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

    // Caching für FFI Buffer (Verhindert GC Thrashing in der Generierungs-Schleife)
    private $cached_c_logits = null;

     /**
     * Lädt das Modell direkt über Nim von der Festplatte (OOM-Sicher).
     */
    public function __construct(string $filepath, int $num_heads = 1) {
        $lib_path = __DIR__ . '/../lib/libcortex.so';
        
        $this->ffi = FFI::cdef("
            void* cortex_init(int vocab_size, int hidden_dim, int ffn_dim, int ctx_size, int num_layers, int num_heads);
            void* cortex_load_stream(const char* filepath, int num_heads);
            float cortex_train_step(void* state, int* raw_inputs, int* raw_targets, int* raw_mask, int num_tokens, float lr);
            void cortex_free(void* state);
            void cortex_reset_cache(void* state);
            void cortex_forward(void* state, int* raw_tokens, int num_tokens, float* raw_logits);
            void cortex_extract_layer(void* state, int layer_idx, int tensor_type, float* raw_weights, int num_weights);
            void cortex_extract_adam_layer(void* state, int layer_idx, int tensor_type, bool is_variance, float* raw, int size);
            void cortex_load_layer(void* state, int layer_idx, int tensor_type, float* raw_weights, int num_weights);
            
            // --- NEU: Phase VI (OpenCL & Utils) ---
            void cortex_init_gpu();
            int get_vocab_size(void* state);
            int get_ctx_size(void* state);

            // --- NEU: Session-Gespraech 2026-08-09 ('Engine-/Framework-
            // Verbesserungen', Punkt 1) - adam_t writer-seitig persistieren ---
            int cortex_get_adam_t(void* state);

            // --- NEU: Phase XI (Diagnose & Guards, seit V11.2-V11.8) ---
            // Bislang in der .so vorhanden, aber aus PHP nicht aufrufbar -
            // Ursache der letzten drei Fehlmessungen in dieser Session.
            int   cortex_get_load_error();
            int   cortex_validate(void* state);
            float cortex_bench_gpu(void* state, int iters);
            void  cortex_profile_reset();
            float cortex_profile_get(int idx);
            int   cortex_get_threads();
            void  cortex_set_gpu_threshold(int elems);
            int   cortex_get_gpu_threshold();
            void  cortex_set_gain_guard(int enabled);

            // --- NEU: V11.9 (Etappe 1 - gebuendeltes w_out) ---
            void  cortex_set_batched_wout(int enabled);
            int   cortex_get_batched_wout();

            // --- AUDIT: gebuendelte Attention (layer-major statt position-major) ---
            void  cortex_set_batched_attention(int enabled);
            int   cortex_get_batched_attention();

            // --- AUDIT: LR-Boost fuer frisch Net2Net-expandierte Parameter ---
            void  cortex_set_growth_boundary(int old_dim, int old_ffn, int old_layers, float boost);

            // --- Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #8/#9):
            // Backend-/Gradient-Norm-Telemetrie. Kein aktiver Trainingslauf zum
            // Zeitpunkt dieser Ergaenzung - Deklaration und .so-Rebuild laufen
            // zusammen, kein Wiederholungsfall des Weight-Decay-Vorfalls von oben.
            int   cortex_get_gpu_backend();
            float cortex_get_last_grad_norm();

            // --- Update 2026-08-09: cortex_set_weight_decay()/
            // cortex_get_weight_decay() waren seit dem Vorfall vom 2026-08-07
            // (siehe Kommentar direkt unten) absichtlich NICHT deklariert -
            // die Wrapper-Methoden set_weight_decay()/get_weight_decay()
            // weiter unten in dieser Klasse existierten dadurch, waren aber
            // nie wirklich aufrufbar. libcortex.so wurde seither im Rahmen
            // dieser Session mehrfach neu gebaut (adam_t-Getter, GradNorm,
            // GPU-Backend, Dropout) - Deklaration und .so sind jetzt wieder
            // synchron, dieselbe Regel wie beim GradNorm-Update angewendet.
            void  cortex_set_weight_decay(float wd);
            float cortex_get_weight_decay();

            // --- Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #6
            // 'Dropout', umgesetzt im Rahmen 'Engine-/Framework-
            // Verbesserungen'): Deklaration und .so-Rebuild liefen wieder
            // zusammen, kein Trainingslauf aktiv zum Zeitpunkt dieser
            // Aenderung (Nutzer hat den laufenden Lauf explizit dafuer
            // beendet).
            void  cortex_set_dropout_rate(float rate);
            float cortex_get_dropout_rate();

            // --- Update 2026-08-15 (docs/ROADMAP.md, 'Naechste Schritte' Punkt 7
            // 'Label-Smoothing'): gleiches Muster wie Weight-Decay/Dropout oben -
            // Deklaration und .so-Rebuild laufen zusammen (kein aktiver
            // Trainingslauf zum Zeitpunkt dieser Aenderung, Nutzer hat den
            // laufenden Lauf explizit dafuer abgebrochen), kein Wiederholungsfall
            // des Weight-Decay-Vorfalls von oben.
            void  cortex_set_label_smoothing(float eps);
            float cortex_get_label_smoothing();

            // --- Update 2026-08-18 (docs/ROADMAP.md, 'Logit Lens'-Fund):
            // Symbol existiert im deployten libcortex.so (Stand 15.08.) schon
            // laenger unter dem internen Namen 'j_space' - bisher nie
            // deklariert/verdrahtet. Kein .so-Rebuild noetig, reine
            // PHP-seitige Ergaenzung, gleiche Regel wie beim adam_t-Getter
            // oben (Deklaration fuer ein bereits vorhandenes .so-Symbol).
            void  cortex_read_j_space(void* state, int layer_idx, float* raw_logits);
        ", $lib_path);
        // AUDIT-FIX (2026-08-07): cortex_set_weight_decay()/cortex_get_weight_decay()
        // NICHT in cdef() deklarieren, solange die deployte libcortex.so nicht neu
        // gebaut ist - FFI::cdef() resolved ALLE deklarierten Symbole SOFORT beim
        // Laden, auch wenn sie nie aufgerufen werden. Hat den laufenden 256-Dim-
        // Trainingslauf bei Epoche 11 mit "Failed resolving C function
        // 'cortex_set_weight_decay'" abstuerzen lassen. Siehe docs/ROADMAP.md,
        // Punkt 1 ("Deploy OFFEN") - erst nach echtem .so-Rebuild wieder aktivieren.

        require_once __DIR__ . '/tensor_io.php';
        $header = load_model_header($filepath);
        if (!$header) die("🚨 Fatal: Konnte Modell-Header nicht lesen.\n");

        $this->vocab_size = $header['vocab_size'];
        $this->hidden_dim = $header['hidden_dim'];
        $this->ctx_size   = $header['ctx_size'];
        $this->num_layers = $header['num_layers'];
        $this->ffn_dim    = $header['ffn_dim'];
        $this->num_heads  = $num_heads;

        $this->state_ptr = $this->ffi->cortex_load_stream($filepath, $this->num_heads);

        // AUDIT-FIX: Sauberer Null-Pointer Check via FFI::isNull
        if (FFI::isNull($this->state_ptr)) {
            die("🚨 Fatal: Nim C-Core konnte den Binär-Stream nicht laden (Pointer ist NULL)!\n");
        }

        // Update 2026-08-09 (docs/ROADMAP.md, "Engine-/Framework-Verbesserungen",
        // Punkt 2): cortex_validate() existierte bereits vollstaendig (prueft
        // tote Gain-Kanaele, NaN/Inf in Gewichten, adam_t=0 trotz warmem m/v),
        // wurde aber nirgends im PHP-Code aufgerufen - eine fertige Pruefung, die
        // niemand nutzte. Nicht fatal (gleiche Vorsicht wie beim Held-out-Eval):
        // ein auffaelliger Befund soll sichtbar werden, aber keinen laufenden
        // Trainings-/Inferenzlauf abwuergen, der trotzdem funktionsfaehig sein
        // kann (z.B. adam_t=0 bremst nur, sprengt nichts, siehe V11.2b-Kommentar
        // in cortex.nim).
        $validation_flags = $this->ffi->cortex_validate($this->state_ptr);
        if ($validation_flags > 0) {
            $issues = [];
            if ($validation_flags & 1) $issues[] = 'tote Gain-Kanaele';
            if ($validation_flags & 2) $issues[] = 'NaN/Inf in Gewichten';
            if ($validation_flags & 4) $issues[] = 'adam_t=0 trotz warmem m/v';
            fwrite(STDERR, "⚠️  [Validate] Checkpoint '$filepath' auffaellig (Flags=$validation_flags): " . implode(', ', $issues) . "\n");
        }

        // Buffer einmalig allokieren
        $this->cached_c_logits = $this->ffi->new("float[{$this->vocab_size}]");
    }

    public function init_gpu() {
        $this->ffi->cortex_init_gpu();
    }

    // Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #8 "Backend-/Hardware-
    // Telemetrie" und #9 "Gradient-Norm-Logging"): 0=keins, 1=OpenCL, 2=CUDA.
    public function get_gpu_backend(): int {
        return $this->ffi->cortex_get_gpu_backend();
    }

    public function get_last_grad_norm(): float {
        return $this->ffi->cortex_get_last_grad_norm();
    }

    // Update 2026-08-08 (docs/ROADMAP.md, "Held-out-Set + periodische Eval"):
    // reset_cache()/forward() waren als C-Symbole bereits seit Laengerem in
    // cdef() deklariert (siehe oben, cortex_reset_cache/cortex_forward -
    // genutzt intern von generate_next_token()), aber nie als eigene
    // oeffentliche Methoden freigelegt. Reine Ergaenzung, kein neues
    // cdef()-Symbol, kein .so-Rebuild-Risiko (siehe AUDIT-FIX-Kommentar beim
    // weight_decay-Vorfall oben).

    public function reset_cache(): void {
        $this->ffi->cortex_reset_cache($this->state_ptr);
    }

    /**
     * Reiner Forward-Pass ohne Training - fuettert $tokens sequentiell in den
     * bestehenden KV-Cache (siehe reset_cache()) und liefert die Logits fuer
     * die Vorhersage NACH dem letzten uebergebenen Token zurueck. Dieselbe
     * Semantik wie generate_next_token(), nur ohne Sampling/Penalty - der
     * Aufrufer bekommt den rohen FFI-Puffer (kein PHP-Array-Kopieren, siehe
     * "VETO-FIX" bei $cached_c_logits oben). Fuer eine ganze Sequenz: Token
     * fuer Token aufrufen (siehe eval_held_out.php), nicht das Array am
     * Stueck uebergeben - cortex_forward() liefert nur Logits fuer die
     * JEWEILS LETZTE Position.
     */
    public function forward(array $tokens): FFI\CData {
        $num = count($tokens);
        $c_in = $this->ffi->new("int[$num]");
        foreach ($tokens as $i => $v) $c_in[$i] = $v;
        $c_logits = $this->cached_c_logits;
        $this->ffi->cortex_forward($this->state_ptr, $this->ffi->cast("int*", $c_in), $num, $this->ffi->cast("float*", $c_logits));
        return $c_logits;
    }

    public function set_growth_boundary(int $old_dim, int $old_ffn, int $old_layers, float $boost): void {
        $this->ffi->cortex_set_growth_boundary($old_dim, $old_ffn, $old_layers, $boost);
    }

    // --- Diagnose-Wrapper (Phase XI) ---

    public function get_threads(): int {
        return $this->ffi->cortex_get_threads();
    }

    public function get_load_error(): int {
        return $this->ffi->cortex_get_load_error();
    }

    /** Bitmaske: 1=tote Gain-Kanaele 2=NaN/Inf 4=adam_t=0 trotz warmem m/v. -1=kein State. */
    public function validate(): int {
        return $this->ffi->cortex_validate($this->state_ptr);
    }

    /** Reine GPU-Roundtrip-Latenz in ms (Mittel ueber $iters Aufrufe). */
    public function bench_gpu(int $iters = 1000): float {
        return $this->ffi->cortex_bench_gpu($this->state_ptr, $iters);
    }

    public function profile_reset(): void {
        $this->ffi->cortex_profile_reset();
    }

    /** idx: 0 matVecMul-GPU 1 matVecMul-CPU 2 backwardAcc-GPU 3 backwardAcc-CPU
     *       4 Attention-Backward 5 Clip+Adam 6 Step gesamt. Rueckgabe in ms. */
    public function profile_get(int $idx): float {
        return $this->ffi->cortex_profile_get($idx);
    }

    public function set_gpu_threshold(int $elems): void {
        $this->ffi->cortex_set_gpu_threshold($elems);
    }

    public function get_gpu_threshold(): int {
        return $this->ffi->cortex_get_gpu_threshold();
    }

    public function set_gain_guard(bool $enabled): void {
        $this->ffi->cortex_set_gain_guard($enabled ? 1 : 0);
    }

    /** V11.9: schaltet den gebuendelten w_out-Pfad ein (Standard: aus). */
    public function set_batched_wout(bool $enabled): void {
        $this->ffi->cortex_set_batched_wout($enabled ? 1 : 0);
    }

    public function get_batched_wout(): int {
        return $this->ffi->cortex_get_batched_wout();
    }

    /** AUDIT: schaltet den gebuendelten (layer-major) Attention-Pfad ein (Standard: aus). */
    public function set_batched_attention(bool $enabled): void {
        $this->ffi->cortex_set_batched_attention($enabled ? 1 : 0);
    }

    /** AdamW-Style Weight-Decay, dekorreliert vom Adam-Update. Default 0.0
     *  (kein Aufruf) verhaelt sich identisch zum bisherigen reinen Adam. */
    public function set_weight_decay(float $wd): void {
        $this->ffi->cortex_set_weight_decay($wd);
    }

    public function get_weight_decay(): float {
        return $this->ffi->cortex_get_weight_decay();
    }

    /** Dropout-Rate fuer die FFN-Aktivierung (0.0 = No-Op, Default). Greift
     *  NUR in train_step(), nie in forward() - siehe cortex.nim-Kommentar zu
     *  applyDropoutForward(). Obergrenze 0.9 wird engine-seitig erzwungen. */
    public function set_dropout_rate(float $rate): void {
        $this->ffi->cortex_set_dropout_rate($rate);
    }

    public function get_dropout_rate(): float {
        return $this->ffi->cortex_get_dropout_rate();
    }

    public function set_label_smoothing(float $eps): void {
        $this->ffi->cortex_set_label_smoothing($eps);
    }

    public function get_label_smoothing(): float {
        return $this->ffi->cortex_get_label_smoothing();
    }

    /** Adam-Bias-Correction-Schrittzaehler, siehe extract_state()/adam_t. */
    public function get_adam_t(): int {
        return $this->ffi->cortex_get_adam_t($this->state_ptr);
    }

    public function get_batched_attention(): int {
        return $this->ffi->cortex_get_batched_attention();
    }

    /**
     * Logit Lens: Residual-Stream nach Layer $layer_idx (nur fuer die LETZTE
     * per forward()/cortex_forward() verarbeitete Token-Position, siehe
     * j_space_cache-Kommentar in cortex.nim) durch die finale RMSNorm + w_out
     * gejagt - liefert Logits, als haette das Netz an dieser Stelle
     * aufgehoert. layer_idx = num_layers-1 muss (bis auf Float-Rauschen)
     * dieselben Logits wie forward() liefern, siehe Sanity-Check in
     * diagnose/snai_logit_lens.php. Reiner Lesezugriff auf einen bereits
     * gefuellten Puffer, kein neuer Forward-Pass.
     */
    public function read_j_space(int $layer_idx): FFI\CData {
        $c_logits = $this->cached_c_logits;
        $this->ffi->cortex_read_j_space($this->state_ptr, $layer_idx, $this->ffi->cast("float*", $c_logits));
        return $c_logits;
    }

    public function __destruct() {
        if ($this->state_ptr !== null && !FFI::isNull($this->state_ptr)) {
            try { $this->ffi->cortex_free($this->state_ptr); } catch (Throwable $e) {}
            $this->state_ptr = null;
        }
    }

    public function extract_state(): array {
        $state = [
            'magic' => 'SNAI_V5',
            'vocab_size' => $this->vocab_size, 'hidden_dim' => $this->hidden_dim,
            'ffn_dim' => $this->ffn_dim,
            'ctx_size' => $this->ctx_size, 'num_layers' => $this->num_layers,
            'num_heads' => $this->num_heads,
            // Update 2026-08-09 ("Engine-/Framework-Verbesserungen", Punkt 1):
            // Adam-Bias-Correction-Schrittzaehler - vorher nie extrahiert,
            // ging bei jedem Speichern verloren. save_model_stream() schreibt
            // ihn als den vom Loader bereits erwarteten optionalen
            // int32-Trailer (siehe cortex.nim, "withT"-Logik).
            'adam_t' => $this->ffi->cortex_get_adam_t($this->state_ptr),
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

        // AUDIT-FIX: Extrahiere nun Gewichte (w), Momentum (m) UND Varianz (v) für Adam!
        foreach ($this->global_map as $type => $keys) {
            $c_array = $this->ffi->new("float[{$sizes[$type]}]");
            
            // 1. Gewichte
            $this->ffi->cortex_extract_layer($this->state_ptr, -1, $type, $this->ffi->cast("float*", $c_array), $sizes[$type]);
            $state['tensors']['global'][$keys['w']] = FFI::string($c_array, $sizes[$type] * 4);
            
            // 2. Adam Momentum (m)
            $this->ffi->cortex_extract_adam_layer($this->state_ptr, -1, $type, false, $this->ffi->cast("float*", $c_array), $sizes[$type]);
            $state['tensors']['global'][$keys['m']] = FFI::string($c_array, $sizes[$type] * 4);
            
            // 3. Adam Variance (v)
            $this->ffi->cortex_extract_adam_layer($this->state_ptr, -1, $type, true, $this->ffi->cast("float*", $c_array), $sizes[$type]);
            $state['tensors']['global'][$keys['v']] = FFI::string($c_array, $sizes[$type] * 4);
        }

        for ($l = 0; $l < $this->num_layers; $l++) {
            $layer = [];
            foreach ($this->layer_map as $type => $keys) {
                $c_array = $this->ffi->new("float[{$sizes[$type]}]");
                
                // 1. Gewichte
                $this->ffi->cortex_extract_layer($this->state_ptr, $l, $type, $this->ffi->cast("float*", $c_array), $sizes[$type]);
                $layer[$keys['w']] = FFI::string($c_array, $sizes[$type] * 4);
                
                // 2. Adam Momentum (m)
                $this->ffi->cortex_extract_adam_layer($this->state_ptr, $l, $type, false, $this->ffi->cast("float*", $c_array), $sizes[$type]);
                $layer[$keys['m']] = FFI::string($c_array, $sizes[$type] * 4);
                
                // 3. Adam Variance (v)
                $this->ffi->cortex_extract_adam_layer($this->state_ptr, $l, $type, true, $this->ffi->cast("float*", $c_array), $sizes[$type]);
                $layer[$keys['v']] = FFI::string($c_array, $sizes[$type] * 4);
            }
            $state['tensors']['layers'][$l] = $layer;
        }

        return $state;
    }

    public function train_step(array $inputs, array $targets, array $mask, float $lr): float {
        $num_tokens = count($inputs);
        if ($num_tokens > $this->ctx_size) {
            $num_tokens = $this->ctx_size;
            $inputs = array_slice($inputs, -$num_tokens);
            $targets = array_slice($targets, -$num_tokens);
            $mask = array_slice($mask, -$num_tokens);
        }

        if ($num_tokens === 0) return 0.0;

        $c_in = $this->ffi->new("int[$num_tokens]");
        $c_tg = $this->ffi->new("int[$num_tokens]");
        $c_mask = $this->ffi->new("int[$num_tokens]");
        foreach ($inputs as $i => $v) $c_in[$i] = $v;
        foreach ($targets as $i => $v) $c_tg[$i] = $v;
        foreach ($mask as $i => $v) $c_mask[$i] = $v ? 1 : 0;

        return $this->ffi->cortex_train_step($this->state_ptr, $this->ffi->cast("int*", $c_in), $this->ffi->cast("int*", $c_tg), $this->ffi->cast("int*", $c_mask), $num_tokens, $lr);
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
        
        // AUDIT-FIX: Softmax Stabilisierung (Verhindert Float-Overflow zu INF/NAN)
        $max_val = max($logits); 
        
        $probs = []; $sum = 0.0;
        foreach ($logits as $id => $val) {
            $p = exp(($val - $max_val) / $temp); 
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
        
        // VETO-FIX: $r skaliert mit der echten $cum-Summe!
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

        // AUDIT-FIX: pack("g*", ...$sum) entfaltet den GESAMTEN Array auf den
        // PHP-Argument-Stack. Bei den Embeddings sind das >4 Mio. Floats -
        // die Funktionsaufruf-Konvention ist dafuer nicht gebaut (Crash/extreme
        // Verlangsamung). train_cluster.php macht es an anderer Stelle bereits
        // korrekt in Bloecken - hier nachgezogen statt neu erfunden.
        $bin_out = '';
        $chunk_size = 65536; // Floats pro pack()-Aufruf, bewusst klein gehalten
        for ($off = 0; $off < $num_floats; $off += $chunk_size) {
            $slice = array_slice($sum, $off, $chunk_size);
            $bin_out .= pack("g*", ...$slice);
        }
        return $bin_out;
    }
}