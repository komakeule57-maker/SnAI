<?php
/**
 * Dateiname: src/snai_inference.php
 * Funktion: SYMBIO NANO-AI FRAMEWORK - API Inference Worker (Phase VIII)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Nackte Latenz. Umgeht PHP komplett beim Laden der Gewichte.
 * FIX: Repetition Penalty (Anti-Loop Korsett) hinzugefügt.
 * FIX: NaN-Tripwire. Bricht hart ab, wenn das Modell korrupte Gewichte hat.
 * UPDATE (Phase VIII): Volle SNAI_V5 Kompatibilität! Das Aegis-Shield weiß nun,
 * dass Trainings-Artefakte ihre Adam-States (m, v) mitschleppen (x3 Dateigröße).
 * UPDATE (2026-08-14): Optionaler Beam-Search-Modus (drittes CLI-Arg) als
 * Alternative zum reinen Temperature-Sampling gegen Wiederhol-/Spam-Kollaps.
 * Rein PHP-seitig via mehrfachem cortex_load_stream() (ein unabhaengiger
 * KV-Cache pro Beam) - kein cortex.nim-Eingriff, kein libcortex.so-Rebuild
 * noetig. Siehe beam_topk_logprobs() und den Beam-Zweig weiter unten.
 */

error_reporting(E_ALL & ~E_NOTICE & ~E_WARNING);
ini_set('display_errors', '0');
ini_set('memory_limit', '128M'); // OOM Safe, da PHP keine Arrays mehr hält!

// AUDIT-FIX: cortex.nim's matVecMul() nutzt OpenMP (`||`-Schleifen, -fopenmp).
// libgomp haelt danach einen idlen Worker-Thread-Pool offen. Beendet sich dieser
// Prozess (wie hier: ein Forward-Pass, dann sofortiges Skript-Ende) sehr kurz nach
// dem letzten Kernel-Aufruf, faengt der Prozess-Exit die Threads oft noch im
// Uebergang in den Idle-Zustand - Segfault in libgomp selbst (verifiziert per gdb-
// Backtrace, reproduzierbar 10/10 ohne diesen Fix). PASSIVE laesst die Worker
// per futex schlafen statt zu spinnen und vermeidet das Zeitfenster (10/10 sauber
// mit diesem Fix). Muss VOR dem FFI::cdef()-Laden von libcortex.so gesetzt werden.
putenv('OMP_WAIT_POLICY=PASSIVE');

require_once 'snai_tokenizer.php';
require_once 'head_config.php';

if ($argc < 3) {
    fwrite(STDERR, "Nutzung: php snai_inference.php <modell.snai> <prompt> [beam_width]\n");
    fwrite(STDERR, "  beam_width weggelassen oder <=1: normales Temperature-/Min-P-Sampling (Default).\n");
    fwrite(STDERR, "  beam_width >=2: Beam-Search ueber N parallele Kandidatenpfade.\n");
    exit(1);
}

$model_file = $argv[1];
$user_input = $argv[2];
$beam_width = isset($argv[3]) ? max(1, (int)$argv[3]) : 1; // >=2 aktiviert Beam-Search (siehe unten)
$temperature = 0.25;         // E>H Sweet-Spot für präzise SFT-Inferenz
$repetition_penalty = 1.15;  // E>H Anti-Loop Korsett (Zwingt das Modell, neue Worte zu finden)
$repetition_window = 64;     // Nur die letzten N Tokens penalisieren, nicht die ganze History
$min_p = 0.05;                // Kappt den Rausch-Tail relativ zum wahrscheinlichsten Token (Füllwort-/<unk>-Spam)
$max_new_tokens = 256;

/**
 * Reproduziert exakt die Temperature/Repetition-Penalty/Min-P-Pipeline des
 * Sampling-Pfads weiter unten, gibt aber Log-Wahrscheinlichkeiten statt eines
 * gezogenen Samples zurueck - Kandidatenbewertung fuer den Beam-Search-Zweig.
 * $top_k begrenzt die Rueckgabe auf die wahrscheinlichsten Kandidaten (staerker
 * gefiltert als min_p allein, haelt den Kandidatenpool pro Beam klein).
 */
function beam_topk_logprobs($c_logits, int $vocab_size, float $temperature, float $repetition_penalty, array $recent_history, float $min_p, int $top_k): array {
    $vals = [];
    for ($v = 0; $v < $vocab_size; $v++) {
        $vals[$v] = $c_logits[$v] / $temperature;
    }

    foreach (array_unique($recent_history) as $prev_tok) {
        if ($vals[$prev_tok] > 0) {
            $vals[$prev_tok] /= $repetition_penalty;
        } else {
            $vals[$prev_tok] *= $repetition_penalty;
        }
    }

    $max_val = max($vals);

    $unnorm = [];
    foreach ($vals as $v => $x) {
        $p = exp($x - $max_val);
        if ($min_p > 0.0 && $p < $min_p) continue; // Rausch-Tail raus, wie im Sampling-Pfad
        $unnorm[$v] = $p;
    }
    if (empty($unnorm)) {
        // Sicherheitsnetz: min_p hat alles gekappt (praktisch nie, da der Top-Token
        // exp(0)=1 immer >= min_p ueberlebt) - Top-Token roh zurueckgeben.
        $unnorm[array_search($max_val, $vals, true)] = 1.0;
    }

    $log_sum = log(array_sum($unnorm));
    $logprobs = [];
    foreach ($unnorm as $v => $p) {
        $logprobs[$v] = ($vals[$v] - $max_val) - $log_sum;
    }

    arsort($logprobs);
    return array_slice($logprobs, 0, $top_k, true);
}

if (!file_exists($model_file)) {
    fwrite(STDERR, "🚨 Fehler: Matrix nicht gefunden: $model_file\n");
    echo "System-Fehler: Cortex-Artefakt nicht gefunden.";
    exit(1);
}

// 1. DYNAMISCHE HEADER-ANALYSE (SNAI_V5 Binär-Format)
// -------------------------------------------------------------------------
$fp = fopen($model_file, 'rb');
$magic = fread($fp, 8);
if (trim($magic) !== 'SNAI_V5') {
    fwrite(STDERR, "🚨 Fatal: Invalider Header. Kein SNAI_V5 Artefakt!\n");
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
// AUDIT-FIX: Vorher eine vom Sidecar-System unabhaengige Heuristik (dim/64), die
// mit dem tatsaechlich beim Training verwendeten resolve_num_heads() auseinanderlaufen
// konnte -> falsche Head-Zahl beim Laden reinterpretiert die gelernten Attention-
// Gewichte komplett (bis hin zu Segfaults bei Modellen, die mit weniger Heads
// trainiert wurden). Jetzt dieselbe Quelle wie Training/Synthese.
$num_heads = resolve_num_heads($model_file, $hidden_dim);

fwrite(STDERR, "[System] V5 Binär-Artefakt decodiert: " . basename($model_file) . "\n");
fwrite(STDERR, "  > Architektur: $hidden_dim Dim | $num_layers Layers | $num_heads Heads\n");
if ($beam_width >= 2) {
    fwrite(STDERR, "  > Beam-Search aktiv: Breite $beam_width (statt Sampling)\n");
}

// =========================================================================
// 🚨 VETO-AUDIT (Phase VIII): Strikte V5 Dateigrößen-Validierung (Aegis-Shield)
// Jedes Array existiert jetzt im Triplet (Gewicht, Momentum, Varianz).
// =========================================================================
$expected_size = 28; // Header
$expected_size += ($vocab_size * $hidden_dim * 4) * 3; // embeddings
$expected_size += ($vocab_size * $hidden_dim * 4) * 3; // w_out
$expected_size += ($hidden_dim * 4) * 3;               // rms_final

$layer_size = ($hidden_dim * 4) * 3 * 2;                 // rms_att, rms_ffn
$layer_size += ($hidden_dim * $hidden_dim * 4) * 3 * 4;  // q, k, v, o
$layer_size += ($ffn_dim * $hidden_dim * 4) * 3 * 2;     // up, down

$expected_size += $layer_size * $num_layers;
$expected_size += 4; // Update 2026-08-09: Adam-Bias-Correction-Schrittzähler (adam_t) als int32-Trailer (siehe tensor_io.php::save_model_stream)

$actual_size = filesize($model_file);
if ($actual_size !== $expected_size) {
    fwrite(STDERR, "\n🚨 [SYMBIO VETO] KORRUPTION ENTDECKT!\n");
    fwrite(STDERR, "  > Artefakt ist unvollständig (Trainings-Abbruch / OOM beim Speichern).\n");
    fwrite(STDERR, "  > Ist-Größe:  " . number_format($actual_size) . " Bytes\n");
    fwrite(STDERR, "  > Soll-Größe: " . number_format($expected_size) . " Bytes (V5)\n\n");
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
    __DIR__ . '/../lib/libcortex.so'
);

// Wecke die GPU auf!
fwrite(STDERR, "  > Initiiere OpenCL JIT-Compiler...\n");
$ffi->cortex_init_gpu();

// Stream Load: Die C-Library zieht die Datei direkt in den VRAM/Heap.
// Beam-Search braucht pro Beam einen eigenen, unabhaengigen KV-Cache -> das
// Modell wird $beam_width mal geladen (bei Default-Breite 1: genau einmal,
// identisch zum bisherigen Verhalten). Fuer die Modellgroessen dieses Projekts
// (Edge-Skala, siehe CLAUDE.md) ist die doppelte/mehrfache Gewichtskopie im
// RAM/VRAM vernachlaessigbar gegenueber dem Nutzen paralleler Kandidatenpfade.
$states = [];
for ($b = 0; $b < $beam_width; $b++) {
    $states[$b] = $ffi->cortex_load_stream($model_file, $num_heads);
    if ($states[$b] == null) {
        fwrite(STDERR, "🚨 Fatal: C-Stream-Loader gescheitert (Unbekannter I/O Fehler, Beam $b).\n");
        echo "System-Fehler: Memory Allocation Error.";
        exit(1);
    }
}
$state_ptr = $states[0]; // Alias fuer den klassischen Sampling-Pfad (beam_width == 1)

// 3. TOKENIZER & PROMPT-ENGINEERING (SFT Korsett)
// -------------------------------------------------------------------------
$tokenizer = new SnaiTokenizer();

// Wir rahmen den Input des Dashboards strikt ein, damit die Drohne antwortet!
$formatted_prompt = "[USER] " . trim($user_input) . "\n[SYMBIO]";
$tokens = $tokenizer->encode($formatted_prompt);

if (count($tokens) > $ctx_size - $max_new_tokens) {
    $tokens = array_slice($tokens, -($ctx_size - $max_new_tokens)); // Context Window erzwingen
}

// AUDIT-FIX (2026-08-14): Hardcoded ID 2 war im aktuellen Vokabular <s> (BOS),
// nicht </s> (EOS) - stammte offenbar noch vom alten 16k-Vokabular, in dem ID 2
// tatsaechlich der Stop-Token war. Jetzt dynamisch aufgeloest wie [USER].
$stop_tokens = [];
if (isset($tokenizer->encode("</s>")[0])) $stop_tokens[] = $tokenizer->encode("</s>")[0]; // EOS Token
if (isset($tokenizer->encode("[USER]")[0])) $stop_tokens[] = $tokenizer->encode("[USER]")[0];

// 4. INFERENZ (Autoregressiver Loop)
// -------------------------------------------------------------------------
$start_time = microtime(true);

if ($beam_width <= 1) {
    // -------------------------------------------------------------------
    // 4a. KLASSISCHER PFAD: ein Kandidatenpfad, Temperature/Min-P-Sampling.
    // Unveraendert gegenueber dem bisherigen Verhalten (Default).
    // -------------------------------------------------------------------
    $c_tokens = $ffi->new("int[" . $ctx_size . "]");
    $c_logits = $ffi->new("float[" . $vocab_size . "]");
    $history = []; // Tracking für Repetition Penalty

    $ffi->cortex_reset_cache($state_ptr);
    foreach ($tokens as $i => $tok) {
        $c_tokens[$i] = $tok;
        $history[] = $tok; // Prompt-Tokens ebenfalls penalisieren
    }

    $ffi->cortex_forward($state_ptr, $c_tokens, count($tokens), $c_logits);
    $prompt_time = round((microtime(true) - $start_time) * 1000);
    fwrite(STDERR, "  > Prompt verdaut in {$prompt_time}ms. Beginne Synthese...\n");

    $generated_text = "";

    for ($i = 0; $i < $max_new_tokens; $i++) {
        $best_token = 0;

        // =========================================================================
        // 🚨 VETO-AUDIT: NaN Tripwire
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
        // 🛡️ REPETITION PENALTY (Verhindert das \n Koma)
        // Nur Sliding Window, sonst penalisiert eine lange Antwort irgendwann fast
        // den ganzen genutzten Vokab und die Verteilung rutscht in den Rausch-Tail.
        // =========================================================================
        $recent_history = array_slice($history, -$repetition_window);
        foreach (array_unique($recent_history) as $prev_tok) {
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

            // =====================================================================
            // 🎯 MIN-P SAMPLING (Kappt den Rausch-Tail: Füllwörter/<unk>/Interpunktion)
            // $max_val ist per Konstruktion (s.o.) das Logit-Maximum, daher hat der
            // wahrscheinlichste Token hier immer unnormalisiertes p = exp(0) = 1.
            // min_p wirkt also direkt als Schwelle relativ zum Top-Token, ohne dass
            // wir vorher normalisieren müssen.
            // =====================================================================
            if ($min_p > 0.0) {
                $sum = 0.0;
                foreach ($probs as $v => $p) {
                    if ($p < $min_p) {
                        $probs[$v] = 0.0;
                    } else {
                        $sum += $p;
                    }
                }
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

    fwrite(STDERR, "\n[⚙️ Metrik: " . mb_strlen($generated_text) . " Chars generiert]\n");

} else {
    // -------------------------------------------------------------------
    // 4b. BEAM SEARCH: $beam_width parallele Kandidatenpfade, je ein
    // physischer FFI-State (= eigener KV-Cache) pro Beam-"Slot". Ein Slot
    // repraesentiert ueber die Zeit nicht immer denselben Beam - wenn ein
    // Elternpfad mehrere Top-Kandidaten gleichzeitig stellt (Fork), wird ein
    // freier Slot per Replay (cortex_reset_cache + kompletter Tokenverlauf
    // neu durch cortex_forward) auf den neuen Pfad synchronisiert. Kein
    // natives Cache-Cloning noetig, daher keine cortex.nim-Aenderung - Preis
    // ist die O(Laenge)-Kosten eines Replays bei jedem Fork statt O(1).
    // -------------------------------------------------------------------
    $top_k = $beam_width; // Kandidaten pro aktivem Beam je Runde

    $c_tokens = [];
    $c_logits = [];
    for ($b = 0; $b < $beam_width; $b++) {
        $c_tokens[$b] = $ffi->new("int[" . $ctx_size . "]");
        $c_logits[$b] = $ffi->new("float[" . $vocab_size . "]");
    }

    // Initialer Forward Pass: Prompt auf jedem Beam-Slot identisch verdauen.
    foreach ($c_tokens as $b => $buf) {
        $ffi->cortex_reset_cache($states[$b]);
        foreach ($tokens as $i => $tok) {
            $buf[$i] = $tok;
        }
        $ffi->cortex_forward($states[$b], $buf, count($tokens), $c_logits[$b]);
    }
    $prompt_time = round((microtime(true) - $start_time) * 1000);
    fwrite(STDERR, "  > Prompt verdaut in {$prompt_time}ms (x$beam_width Beams). Beginne Beam-Synthese...\n");

    if (is_nan($c_logits[0][0])) {
        fwrite(STDERR, "\n🚨 [SYMBIO VETO] KERNEL PANIC: Cortex spuckt NaNs! Die Matriarchin hat korrupte Gewichte (Explodierte Gradienten beim Merge).\n");
        $beam_gen = [0 => []];
        $active = [];
        $finished = [['tokens' => [], 'score' => 0.0, 'len' => 0]];
    } else {
        $beam_gen = array_fill(0, $beam_width, []);   // generierte Tokens je Slot
        $beam_score = array_fill(0, $beam_width, 0.0); // kumulierte Log-Wahrscheinlichkeit je Slot
        $active = range(0, $beam_width - 1);
        $finished = []; // abgeschlossene Kandidaten: ['tokens'=>[], 'score'=>float, 'len'=>int]

        for ($step = 0; $step < $max_new_tokens; $step++) {
            if (empty($active)) break;

            $step_len = count($tokens) + count($beam_gen[$active[0]]); // alle aktiven Beams gleich lang
            if ($step_len >= $ctx_size - 1) break; // Kontextfenster voll -> Rest als "aktiv am Cutoff" unten uebernehmen

            // Runde 0: alle Slots tragen noch den identischen Prompt-Forward-Pass ->
            // nur EINEN Slot als Quelle nehmen, sonst B identische Kandidatensaetze.
            $sources = ($step === 0) ? [$active[0]] : $active;

            $candidates = [];
            foreach ($sources as $slot) {
                if (is_nan($c_logits[$slot][0])) {
                    fwrite(STDERR, "\n🚨 [SYMBIO VETO] KERNEL PANIC: Cortex spuckt NaNs (Beam-Slot $slot)!\n");
                    continue;
                }
                $recent = array_slice(array_merge($tokens, $beam_gen[$slot]), -$repetition_window);
                $logprobs = beam_topk_logprobs($c_logits[$slot], $vocab_size, $temperature, $repetition_penalty, $recent, $min_p, $top_k);
                foreach ($logprobs as $tok => $lp) {
                    $candidates[] = ['src' => $slot, 'tok' => $tok, 'score' => $beam_score[$slot] + $lp];
                }
            }
            if (empty($candidates)) break;

            usort($candidates, fn($a, $b) => $b['score'] <=> $a['score']);
            $picks = array_slice($candidates, 0, min(count($active), count($candidates)));

            // Kandidaten in "beendet" (Stop-Token / Halluzinations-Fang) und
            // "wird fortgesetzt" trennen. Ein Stop braucht keinen physischen
            // Slot-Umbau - der Tokenverlauf des Elternpfads reicht zur Bewertung.
            $continue_picks = [];
            foreach ($picks as $p) {
                $is_stop = in_array($p['tok'], $stop_tokens, true);
                if (!$is_stop) {
                    $word = $tokenizer->decode([$p['tok']]);
                    if (strpos($word, "[USER]") !== false) $is_stop = true;
                }
                if ($is_stop) {
                    $finished[] = [
                        'tokens' => $beam_gen[$p['src']],
                        'score' => $p['score'],
                        'len' => count($beam_gen[$p['src']]),
                    ];
                } else {
                    $continue_picks[] = $p;
                }
            }

            if (empty($continue_picks)) { $active = []; break; }

            // Physische Slot-Zuordnung: erste Wiederverwendung eines Quell-Slots
            // laeuft in-place (Cache schon synchron, nur 1 Token anhaengen).
            // Jede weitere Verwendung derselben Quelle (Fork) braucht einen
            // freien Slot + vollstaendigen Replay ihres Tokenverlaufs.
            $reuse_idx_for_src = [];
            foreach ($continue_picks as $idx => $p) {
                if (!isset($reuse_idx_for_src[$p['src']])) {
                    $reuse_idx_for_src[$p['src']] = $idx;
                }
            }
            $used_as_src = array_values(array_unique(array_column($continue_picks, 'src')));
            $free_pool = array_values(array_diff($active, $used_as_src));

            $new_beam_gen = [];
            $new_beam_score = [];
            $new_active = [];

            foreach ($continue_picks as $idx => $p) {
                $tok = $p['tok'];
                if ($reuse_idx_for_src[$p['src']] === $idx) {
                    // In-Place: Cache ist bereits exakt auf $p['src']'s Verlauf synchron.
                    $slot = $p['src'];
                    $c_tokens[$slot][0] = $tok;
                    $ffi->cortex_forward($states[$slot], $c_tokens[$slot], 1, $c_logits[$slot]);
                } else {
                    // Fork: freien Slot per Replay auf den neuen Pfad synchronisieren.
                    $slot = array_shift($free_pool);
                    $ffi->cortex_reset_cache($states[$slot]);
                    $replay = array_merge($tokens, $beam_gen[$p['src']], [$tok]);
                    foreach ($replay as $ri => $rt) {
                        $c_tokens[$slot][$ri] = $rt;
                    }
                    $ffi->cortex_forward($states[$slot], $c_tokens[$slot], count($replay), $c_logits[$slot]);
                }

                $new_beam_gen[$slot] = array_merge($beam_gen[$p['src']], [$tok]);
                $new_beam_score[$slot] = $p['score'];
                $new_active[] = $slot;
            }

            $beam_gen = $new_beam_gen;
            $beam_score = $new_beam_score;
            $active = $new_active;
        }

        // Beams, die bis zum Token-/Kontextlimit durchliefen ohne Stop-Token,
        // zaehlen ebenfalls als Abschluss-Kandidaten (nur eben unbeendet).
        foreach ($active as $slot) {
            $finished[] = ['tokens' => $beam_gen[$slot], 'score' => $beam_score[$slot], 'len' => count($beam_gen[$slot])];
        }
    }

    // Gewinner: hoechste laengennormalisierte Log-Wahrscheinlichkeit (verhindert,
    // dass kurze, frueh gestoppte Pfade laengere allein durch Kuerze schlagen).
    $winner = null;
    $winner_norm = -INF;
    foreach ($finished as $f) {
        $norm = $f['score'] / max(1, $f['len']);
        if ($norm > $winner_norm) {
            $winner_norm = $norm;
            $winner = $f;
        }
    }

    $generated_text = $winner ? $tokenizer->decode($winner['tokens']) : "";
    echo $generated_text;

    fwrite(STDERR, "\n[⚙️ Metrik: " . mb_strlen($generated_text) . " Chars generiert, " . count($finished) . " Beam-Kandidaten verglichen, Score/Token=" . round($winner_norm, 4) . "]\n");
}

foreach ($states as $s) {
    $ffi->cortex_free($s);
}
