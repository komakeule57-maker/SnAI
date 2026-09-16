<?php
/**
 * Datei: train_worker.php
 * Funktion: Lokaler CLI-Worker für das Training der SNAI-Modelle. 
 * Liest Text-Chunks, tokenisiert sie, füttert die C-Bridge 
 * und speichert die Gewichte per Stream.
 * * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Worker Training Node (Phase V - OOM Zero Edition)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Verarbeitet Chunks dynamischer Länge. 
 * AUDIT-V9.4: Array-Slice GC-Shredder in der Hot-Loop eliminiert.
 * AUDIT-FIX: UTF-8 Notfall-Cut repariert, Header-Validierung hinzugefügt.
 * VERSION: V2.1 - Initialisierungsreihenfolge korrigiert (num_heads vor Konstruktor).
 */

// AUDIT-FIX (2026-08-11): siehe beehive_core.php - date.timezone-Ini steht auf
// UTC statt Europe/Berlin (betrifft hier tensor_ffi_adapter.php's date()-Aufrufe).
date_default_timezone_set('Europe/Berlin');
// Limit auf 2048M hochgesetzt, da extract_state() am Ende kurzzeitig RAM für das gesamte Modell benötigt (E>H Sweetspot)
ini_set('memory_limit', '2048M');
require_once 'snai_tokenizer.php';
require_once 'tensor_io.php';
require_once 'tensor_ffi_adapter.php';
require_once __DIR__ . '/head_config.php';
require_once __DIR__ . '/growth_config.php';

// LR-Boost fuer frisch Net2Net-expandierte Parameter (siehe docs/ROADMAP.md).
// Faktor bewusst moderat gewaehlt als Startwert - Feintuning folgt empirisch.
define('GROWTH_LR_BOOST', 3.0);

/**
 * AUDIT-FIX: fread() liefert auf Netzwerk-Mounts (GVFS/SMB/NFS) regelmaessig
 * weniger Bytes als angefordert, ohne Fehler zu melden. Der Header-Read lief
 * dadurch in "Basismodell ist korrupt", obwohl die Datei in Ordnung war.
 */
function worker_robust_fread($fp, int $length): string {
    $data = '';
    while (strlen($data) < $length && !feof($fp)) {
        $chunk = fread($fp, $length - strlen($data));
        if ($chunk === false || $chunk === '') break;
        $data .= $chunk;
    }
    return $data;
}

// Konstanten für Magic Numbers (Wartbarkeit)
define('MIN_FILE_SIZE', 10);
define('DEFAULT_LR', 0.0005);
// Update 2026-08-09 (Session-Gespraech): Fenstergroesse fuer den "Loss der
// letzten N Schritte" (siehe RECENT_LOSS= unten) - bewusst deutlich kleiner
// als typische Epochenlaengen (haeufig mehrere hundert bis tausend Schritte),
// damit der Wert den Endstand der Epoche widerspiegelt statt vom
// Epochenanfang verwaessert zu werden.
define('RECENT_LOSS_WINDOW', 20);

if ($argc < 6) {
    die("Nutzung: php train_worker.php <input.txt> <output.snai> <epochs> <base_model> <dtype> [custom_lr] [global_epoch] [base_lr] [weight_decay] [dropout_rate] [label_smoothing]\n");
}

$input_file  = $argv[1];
$output_file = $argv[2];
$epochs      = (int)$argv[3];
$base_model  = $argv[4];
$dtype       = (int)$argv[5];
$custom_lr   = isset($argv[6]) ? (float)$argv[6] : DEFAULT_LR;
$global_epoch = isset($argv[7]) ? (int)$argv[7] : 1;
$base_lr     = isset($argv[8]) ? (float)$argv[8] : DEFAULT_LR;
// Optionale Regularisierung (Engine-Hooks existierten bereits in cortex.nim,
// wurden aber nirgends aufgerufen - siehe Session-Notiz zu Punkt 2 der
// Memorierungs-Gegenmassnahmen). Default 0.0 = No-Op, damit bestehende
// Aufrufe ohne diese Argumente exakt gleich verhalten wie bisher.
$weight_decay = isset($argv[9]) ? (float)$argv[9] : 0.0;
$dropout_rate = isset($argv[10]) ? (float)$argv[10] : 0.0;
// Update 2026-08-15 (docs/ROADMAP.md, "Naechste Schritte" Punkt 7): als
// letzter positioneller Arg angehaengt statt mittendrin eingefuegt - sonst
// verschieben sich alle nachfolgenden argv[N]-Indizes.
$label_smoothing = isset($argv[11]) ? (float)$argv[11] : 0.0;

echo "=== [WORKER " . getmypid() . "] INITIALISIERE SCHMIEDE ===\n";

// Sanity Check Lernrate
if ($custom_lr <= 0) $custom_lr = DEFAULT_LR;
if ($base_lr <= 0.00001) {
    echo "  > [Worker " . getmypid() . "] 🌙 Traum-Modus aktiv (Experience Replay).\n";
}

if (!file_exists($base_model)) {
    die("🚨 Fatal: Basismodell konnte nicht gefunden werden -> $base_model\n");
}

if (!file_exists($input_file) || filesize($input_file) < MIN_FILE_SIZE) {
    echo "  > 🚨 Warnung: Chunk ist leer oder fehlt. Kopiere Basismodell als Fallback...\n";
    if (!copy($base_model, $output_file)) {
        die("🚨 Fatal: Konnte Basismodell nicht kopieren!\n");
    }
    // AUDIT-FIX: Ohne die Sidecar faellt der naechste Ladevorgang auf den
    // Legacy-Wert 1 Head zurueck und deutet die Attention-Gewichte um.
    $fallback_sidecar = head_sidecar_path($base_model);
    if (file_exists($fallback_sidecar)) {
        @copy($fallback_sidecar, head_sidecar_path($output_file));
    }
    propagate_growth_sidecar($base_model, $output_file);
    exit(0);
}

$tokenizer = new SnaiTokenizer();
$vocab_limit = $tokenizer->getVocabSize();

// Loss-Masking (docs/ROADMAP.md, "Loss-Masking auf [USER]-Tokens"): [USER]
// und [SYMBIO] sind registrierte Einzel-Tokens (siehe snai_tokenizer.php),
// hier einmalig aufgeloest statt pro Chunk neu zu encoden.
$user_token_ids = $tokenizer->encode('[USER]');
$symbio_token_ids = $tokenizer->encode('[SYMBIO]');
$user_token_id = $user_token_ids[0];
$symbio_token_id = $symbio_token_ids[0];

// ============================================================================
// FIX V2.1: Header UND Head-Zahl muessen VOR dem Konstruktor feststehen.
// In V2 stand der Header-Read weiterhin hinter "new SnaiCortexBridge(...)",
// waehrend der Konstruktor bereits $num_heads erwartete -> "Undefined variable"
// und sofortiger Fatal Error in jedem Worker.
// ============================================================================

// --- 1. Header lesen (vor allem anderen) ---
$fp = fopen($base_model, 'rb');
worker_robust_fread($fp, 8); // Skip Magic
$h_data = worker_robust_fread($fp, 20);
fclose($fp);

// AUDIT-FIX: Harte Header-Validierung, um undefinierte Arrays und falsche CPU-Fallbacks zu verhindern
if ($h_data === false || strlen($h_data) !== 20) {
    die("🚨 [Fatal] Basismodell ist korrupt oder unvollständig (Header < 28 Bytes)!\n");
}

$p = unpack('Vvocab/Vdim/Vffn/Vctx/Vlayers', $h_data);
$model_dim = $p['dim'];

// --- 2. Head-Zahl aufloesen ---
// num_heads ist NICHT im .cortex-Header persistiert. Ein bereits trainierter
// Checkpoint hat headSize implizit in seinen Gewichten "eingebrannt" - wird er
// mit einer anderen Head-Zahl geladen, bleiben die Zahlen gleich, ihre Bedeutung
// (RoPE-Rotation, Attention-Segmentierung) aendert sich komplett. Deshalb liegt
// die Head-Zahl in einer Sidecar-Datei neben dem Modell und wird NIE aus der
// Modellgroesse neu geraten, sobald sie einmal feststeht.
require_once __DIR__ . '/head_config.php';
$num_heads = resolve_num_heads($base_model, $model_dim);
echo "  > [Heads] $num_heads Attention-Heads (headSize=" . intdiv($model_dim, $num_heads) . ") fuer dieses Modell.\n";

// --- 3. DER INJEKTOR (jetzt mit gueltigem $num_heads) ---
$cortex = new SnaiCortexBridge($base_model, $num_heads);

// --- 3b. Net2Net-Wachstumsgrenze aufloesen (LR-Boost fuer frische Parameter) ---
// Sidecar existiert nur, wenn $base_model aus einer Expansion in dieser
// Synthese-Session stammt (siehe growth_config.php) - sonst kein Boost,
// identisch zum bisherigen Verhalten.
$growth = resolve_growth_boundary($base_model);
if ($growth !== null && method_exists($cortex, 'set_growth_boundary')) {
    [$g_old_dim, $g_old_ffn, $g_old_layers] = $growth;
    $cortex->set_growth_boundary($g_old_dim, $g_old_ffn, $g_old_layers, GROWTH_LR_BOOST);
    echo "  > [Growth] Net2Net-Grenze erkannt (alt: dim=$g_old_dim, ffn=$g_old_ffn, layers=$g_old_layers) -> LR-Boost " . GROWTH_LR_BOOST . "x fuer frische Parameter aktiv.\n";
}

// --- 3c. Regularisierung (Weight-Decay/Dropout), nur falls explizit > 0 ---
if ($weight_decay > 0.0 && method_exists($cortex, 'set_weight_decay')) {
    $cortex->set_weight_decay($weight_decay);
    echo "  > [Regularisierung] AdamW-Weight-Decay aktiv: $weight_decay\n";
}
if ($dropout_rate > 0.0 && method_exists($cortex, 'set_dropout_rate')) {
    $cortex->set_dropout_rate($dropout_rate);
    echo "  > [Regularisierung] FFN-Dropout aktiv: $dropout_rate\n";
}
if ($label_smoothing > 0.0 && method_exists($cortex, 'set_label_smoothing')) {
    $cortex->set_label_smoothing($label_smoothing);
    echo "  > [Regularisierung] Label-Smoothing aktiv: $label_smoothing\n";
}

// --- 4. ⚡ PHASE VI: OpenCL JIT (NUR AB 512 DIM) ---
if (method_exists($cortex, 'init_gpu') && $model_dim >= 512) {
    echo "  > [VRAM] Modell hat {$model_dim} Dim. Aktiviere GPU-Beschleunigung...\n";
    $cortex->init_gpu();
} else {
    // AUDIT-FIX: Meldung sagte "(< 256)", die tatsaechliche Schwelle oben ist 512 -
    // Copy-Paste-Rest einer alten Version, hat nichts mit der Gate-Logik zu tun.
    echo "  > [CPU] Modell hat {$model_dim} Dim (< 512). Bleibe auf CPU (PCIe-Overhead Schutz)...\n";
}

// Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #8 "Backend-/Hardware-
// Telemetrie"): parsebare Zeile fuer train_cluster.php, unabhaengig davon, ob
// oben ueberhaupt init_gpu() gerufen wurde (dann bleibt es beim Default 0/NONE).
if (method_exists($cortex, 'get_gpu_backend')) {
    $backend_names = [0 => 'NONE', 1 => 'OPENCL', 2 => 'CUDA'];
    $backend_id = $cortex->get_gpu_backend();
    echo "GPU_BACKEND=" . ($backend_names[$backend_id] ?? 'UNKNOWN') . "\n";
}


$ctx = property_exists($cortex, 'ctx_size') ? $cortex->ctx_size : 512;
if ($ctx <= 0) die("🚨 Fatal: Ungültige ctx_size ($ctx) vom Cortex gemeldet!\n");

// HINWEIS: $initial_lr ist die vom Master vorgegebene, bereits decay'te Rate für DIESE Epoche!
$initial_lr = $custom_lr; 
$global_step = 0;

for ($e = 1; $e <= $epochs; $e++) {
    $loss_sum = 0;
    $steps = 0;
    // Update 2026-08-09 (Session-Gespraech): rollierender Puffer der letzten
    // RECENT_LOSS_WINDOW Einzelschritt-Losses - siehe RECENT_LOSS= am Ende der
    // Mini-Batch-Schleife unten. Pro Epoche zurueckgesetzt (kein
    // epochenuebergreifender Puffer), damit der Wert immer den Endstand
    // GENAU DIESER Epoche zeigt.
    $recent_losses = [];

    // AUDIT-FIX (docs/ROADMAP.md, "Batch-Shuffling zwischen Epochen"): die
    // Chunk-Datei wird jetzt einmalig komplett gelesen und am Stueck
    // tokenisiert statt in 1-MB-Leseputtern - dadurch entfaellt der bisherige
    // Whitespace-/UTF-8-Notfall-Cut an Pufferrgrenzen (unnoetig, sobald keine
    // Puffergrenzen mehr existieren) UND es steht die vollstaendige
    // Fenster-Liste vor dem ersten Trainingsschritt fest, sodass ihre
    // Reihenfolge gemischt werden kann. Chunk-Dateien sind bereits von
    // train_cluster.php pro Worker vorab aufgeteilt (typischerweise
    // niedrige einstellige MB) - unproblematisch fuer den 2048M-Limit oben.
    $full_text = file_get_contents($input_file);
    if ($full_text === false) {
        die("🚨 Fatal: Konnte Input-File nicht lesen: $input_file\n");
    }

    if (strlen(trim($full_text)) >= 2) {
        $tokens = $tokenizer->encode($full_text);
        unset($full_text);
        $total_tokens = count($tokens);
        $batch_size = $ctx;

        // Loss-Masking: Rollen-Scan ueber den kompletten Token-Stream, EINMAL
        // pro Chunk statt pro Fenster. Default true (trainierbar) - GEN-Chunks
        // ohne [USER]/[SYMBIO]-Marker bleiben dadurch automatisch komplett
        // trainierbar, keine Sonderbehandlung nach Tier noetig. Das Marker-
        // Token selbst bekommt die NEUE Rolle (der Rollenwechsel ist selbst
        // ein lernbares Signal).
        $role = [];
        $current_role = true;
        foreach ($tokens as $idx => $tok) {
            if ($tok === $user_token_id) $current_role = false;
            elseif ($tok === $symbio_token_id) $current_role = true;
            $role[$idx] = $current_role;
        }

        // Fenster-Startpositionen (0, ctx, 2*ctx, ...) sammeln, dann
        // deterministisch pro (globaler Epoche, interner Epoche, Chunk-Datei)
        // mischen - deterministisch statt echt zufaellig, damit ein Neustart
        // mit demselben global_epoch dieselbe Reihenfolge reproduziert (es
        // gibt keinen Intra-Epochen-Fortschritts-State, ein Neustart
        // wiederholt die Epoche ohnehin komplett).
        $offsets = [];
        for ($i = 0; $i < $total_tokens - 1; $i += $batch_size) {
            $offsets[] = $i;
        }
        mt_srand(crc32($global_epoch . ':' . $e . ':' . $input_file));
        shuffle($offsets);

        // === MINI-BATCH SCHLEIFE (VETO-FIX: Array Slice GC-Shredder behoben) ===
        foreach ($offsets as $i) {
            // AUDIT-FIX: Vorher wurde erst ein voller Batch geschnitten und dann
            // per "$i + $actual_len >= $total_tokens" abgebrochen. Bei
            // $total_tokens <= $batch_size galt das schon im ERSTEN Durchlauf
            // (0 + total >= total) -> null Trainingsschritte, das Modell wurde
            // trotzdem als fertiger Experte gespeichert. Zusaetzlich fiel bei
            // langen Texten immer der letzte Teil-Batch unter den Tisch.
            //
            // Jetzt wird die Laenge vorab begrenzt: fuer ein Paar (in, out) mit
            // out = in um 1 verschoben stehen ab Position $i genau
            // ($total_tokens - 1 - $i) Tokens zur Verfuegung.
            $actual_len = min($batch_size, $total_tokens - 1 - $i);
            if ($actual_len < 1) continue;

            $batch_in  = array_slice($tokens, $i, $actual_len);
            $batch_out = array_slice($tokens, $i + 1, $actual_len);
            // Maske beschreibt die Rolle des ZIEL-Tokens ($batch_out), nicht
            // des Input-Tokens - genau das wird vorhergesagt und soll bei
            // [USER]-Text nicht in den Loss/Gradienten einfliessen.
            $batch_mask = array_slice($role, $i + 1, $actual_len);

            // AUDIT-FIX 2026-08-18 (docs/ROADMAP.md, Engine-Audit Befund 8 + 7):
            // Ein Fenster, dessen Ziel-Tokens KOMPLETT maskiert sind (reiner
            // [USER]-Text), liefert per Definition keinen Gradienten:
            // cortex_train_step() setzt alle d_logits auf 0, die globale Norm
            // wird 0, adamStep() ueberspringt jeden Parameter wegen scale == 0.
            // Das Fenster kostete also einen vollen Forward+Backward-Pass fuer
            // exakt nichts - UND es gab 0.0 als Loss zurueck, was hier
            // ungeprueft in $loss_sum landete waehrend $steps hochzaehlte: der
            // geloggte "Avg Loss" wurde dadurch nach unten verwaessert, ein
            // Messfehler, der wie eine Verbesserung aussieht. Vorab
            // ueberspringen ist trainingsmathematisch exakt aequivalent (kein
            // Gewicht aendert sich in so einem Schritt), spart den Pass und
            // haelt zusaetzlich den Adam-Schrittzaehler ehrlich.
            $has_trainable = false;
            foreach ($batch_mask as $m) {
                if ($m) { $has_trainable = true; break; }
            }
            if (!$has_trainable) continue;

            $global_step++;
            $current_lr = $initial_lr;

            // Bounds Check
            $max_id = max($batch_in);
            $min_id = min($batch_in);
            if ($min_id < 0 || $max_id >= $vocab_limit) {
                echo "🚨 [Worker Debug] ILLEGALE TOKENS! Min: $min_id, Max: $max_id (Limit: $vocab_limit)\n";
                exit(1);
            }

            // Inferenz / Training in der C-Engine
            if (method_exists($cortex, 'train_step')) {
                $loss = $cortex->train_step($batch_in, $batch_out, $batch_mask, $current_lr);
                if ($loss === null) $loss = 0.001;
            } else {
                echo "⚠️ [Worker " . getmypid() . "] train_step() fehlt! Simuliere...\n";
                $loss = max(0.001, 0.99 * (1.0 / (($global_step * 0.01) + 1)));
            }

            // AUDIT-FIX: Off-by-one Logging Bug behoben
            if (is_nan($loss) || is_infinite($loss)) {
                $err_step = $steps + 1;
                echo "🚨 [Fatal] Worker " . getmypid() . " | Exploding Gradients (NaN) bei Step $err_step! Breche ab.\n";
                exit(1);
            }

            $loss_sum += $loss;
            $steps++;
            $recent_losses[] = $loss;

            if ($steps <= 5 || $steps % 5 === 0) {
                // Update 2026-08-09 (docs/ROADMAP.md, Nachruest-Paket #9
                // "Gradient-Norm-Logging"): GradNorm haengt an dieselbe Zeile wie
                // Avg Loss an - die bestehende "Avg Loss:"-Regex in
                // train_cluster.php bleibt dadurch unveraendert gueltig, ein
                // neuer Regex greift zusaetzlich das Feld danach ab.
                // AUDIT-FIX (Session-Gespraech 2026-08-09): number_format() ohne
                // explizite Trenner-Argumente setzt ab 1000 ein Tausender-Komma
                // (z.B. "1,042.7545") - der Parse-Regex in train_cluster.php
                // ([0-9.]+) brach dort ab und lieferte nur "1" statt des echten
                // Werts. GradNorm liegt in der Praxis oft > 1000. Explizit ohne
                // Trenner formatieren, wie $active_lr in train_cluster.php.
                $grad_norm_str = method_exists($cortex, 'get_last_grad_norm')
                    ? number_format($cortex->get_last_grad_norm(), 4, '.', '')
                    : 'n/a';
                echo "Worker " . str_pad((string)getmypid(), 5, " ", STR_PAD_LEFT) . " | Epoch $global_epoch | Step " . str_pad((string)$steps, 3, " ", STR_PAD_LEFT) . " | Avg Loss: " . number_format($loss_sum / $steps, 6) . " | GradNorm: $grad_norm_str\n";
            }

            // GC explizit entlasten
            unset($batch_in, $batch_out);
        }

        // Update 2026-08-09 (Session-Gespraech): Loss der letzten
        // RECENT_LOSS_WINDOW Schritte statt des Epochendurchschnitts -
        // $loss_sum/$steps (siehe "Avg Loss" oben) ist ueber die GANZE Epoche
        // gemittelt, verwaesert also von den anfangs noch schlechteren
        // Schritten. Fuer einen direkten Vergleich mit dem Held-out-Loss
        // (der immer den FERTIGEN Epochenstand misst) ist dieser Wert die
        // passendere Grundlage. Eigene KEY=VALUE-Zeile (Muster wie
        // GPU_BACKEND= oben), garantiert am tatsaechlichen Ende der
        // Mini-Batch-Schleife berechnet - unabhaengig davon, ob der letzte
        // Schritt zufaellig auf eine "Step % 5 === 0"-Log-Zeile fiel.
        if (count($recent_losses) > 0) {
            $recent_window = array_slice($recent_losses, -RECENT_LOSS_WINDOW);
            $recent_avg = array_sum($recent_window) / count($recent_window);
            echo "RECENT_LOSS=" . number_format($recent_avg, 6, '.', '') . " RECENT_LOSS_N=" . count($recent_window) . "\n";
        }
        unset($tokens, $offsets, $recent_losses);
    }
}

// AUDIT-FIX: Ohne einen einzigen Schritt waere das Ergebnis eine unveraenderte
// Kopie des Basismodells - die Pipeline hat solche Klone bisher als vollwertige
// Experten verbucht (Loss-Platzhalter 999). Lieber laut scheitern.
if ($global_step === 0) {
    echo "🚨 [Fatal] Worker " . getmypid() . ": kein einziger Trainingsschritt ausgefuehrt.\n";
    echo "  > Der Chunk enthaelt zu wenige Tokens (mindestens 2 noetig).\n";
    exit(1);
}

echo "  > [Worker " . getmypid() . "] Training abgeschlossen. Extrahiere aktualisierte Gewichte...\n";

// Hier passiert der große RAM-Spike (Die OOM-Illusion), ist aber für E>H aktuell akzeptabel
if (method_exists($cortex, 'extract_state')) {
    $state_array = $cortex->extract_state();
    if ($state_array === false || $state_array === null) {
        die("🚨 [Worker " . getmypid() . "] FATAL: extract_state() lieferte ungültiges Ergebnis!\n");
    }
} else {
    die("🚨 [Worker " . getmypid() . "] FATAL: Konnte Gewichte nicht aus C-RAM extrahieren (Methode fehlt)!\n");
}

if (!function_exists('save_model_stream')) {
    die("🚨 Fatal: 'save_model_stream' nicht in tensor_io.php gefunden!\n");
}

try {
    save_model_stream($state_array, $output_file);
    // Head-Zahl fuer den NAECHSTEN Ladevorgang festschreiben, damit sie nie
    // wieder aus $model_dim neu geraten werden kann/muss.
    write_num_heads($output_file, $num_heads);
    echo "  > ✅ [Erfolg] Arbeiter-Gewichte via Stream gesichert unter $output_file\n";
} catch (Exception $e) {
    die("🚨 Fatal: Fehlschlag beim Speichern des Modells: " . $e->getMessage() . "\n");
}

exit(0);