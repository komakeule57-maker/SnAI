<?php
// Dateiname: src/royal_distillery.php
// Funktion: Die 'Königin'. Synthetisiert kleine Worker-Tensoren iterativ bis zur Goddess (Phase VII).

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Royal Distillery (Phase VIII - The Adam Persistence Edition)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Stream-basierte Synthese & Network Morphism.
 * UPDATE (Phase VIII): Volle Unterstützung für SNAI_V5. Skaliert nun nicht nur
 * die Gewichte, sondern auch das Adam-Gedächtnis (m, v) in OOM-Zero Streams mit!
 * AUDIT-FIX V11: Exception-Handling. `die()` entfernt, damit der Daemon nicht crasht.
 */

ini_set('memory_limit', '1024M');

require_once 'tensor_io.php';
require_once 'tensor_ffi_adapter.php';
require_once 'snai_caste_profiler.php';
require_once __DIR__ . '/head_config.php';
require_once __DIR__ . '/growth_config.php';
require_once __DIR__ . '/tensor_utils_ffi.php';
require_once __DIR__ . '/ram_checkpoint_config.php';

if (!defined('SNAI_TYPE_FP32')) define('SNAI_TYPE_FP32', 1);

class RoyalDistillery {
    
    private const CHUNK_SIZE_BYTES = 1048576; // 1 MB für optimalen SSD Durchsatz
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

    /**
     * Gruppiert Dateien nach ihrer TATSAECHLICHEN Head-Zahl (via resolve_num_heads,
     * dieselbe Quelle, die auch Training/Inferenz nutzen) und gibt die groesste
     * Gruppe zurueck, die mindestens $min_count Mitglieder hat (leer, wenn keine
     * Gruppe gross genug ist).
     *
     * WARUM: merge_tributes() mittelt Tribute rein byteweise, ganz ohne Ruecksicht
     * auf Attention-Struktur. Mischt man dabei Modelle mit unterschiedlicher
     * Head-Zahl (z.B. ein per Smart-Forge frisch mit headSize=64 erzeugtes
     * Modell mit zwei Legacy-Modellen mit headSize=voller-Dimension), ergibt
     * das gemittelte Ergebnis weder ein gueltiges altes noch ein gueltiges neues
     * Attention-Layout - unabhaengig davon, welche Head-Zahl man hinterher fuer
     * das Ergebnis festschreibt. Deshalb muss die Kompatibilitaet VOR dem Mergen
     * geprueft werden, nicht erst bei der Frage "wie soll das Ergebnis skalieren".
     */
    // AUDIT-FIX: public, damit beehive_core.php::action_distill() Tribute schon
    // VOR dem Kopieren nach tributes/ auf Head-Kompatibilitaet filtern kann -
    // sonst wiederholt die Auswahl dort stur dieselben (inkompatiblen) Top-3-
    // nach mtime, und diese Methode lehnt sie hier jedes Mal erneut ab
    // (Endlosschleife aus wiederholten, garantiert scheiternden Merge-Versuchen).
    public function largest_head_compatible_group(array $files, int $min_count): array {
        $groups = [];
        foreach ($files as $f) {
            $header = load_model_header($f);
            if (!$header) continue;
            $heads = resolve_num_heads($f, $header['hidden_dim']);
            $groups[$heads][] = $f;
        }
        $best = [];
        foreach ($groups as $heads => $group_files) {
            if (count($group_files) >= $min_count && count($group_files) > count($best)) {
                $best = $group_files;
            }
        }
        return $best;
    }

    private function validate_tensor_file(string $file): bool {
        $fp = @fopen($file, 'rb');
        if (!$fp) return false;
        
        // AUDIT-FIX: Hardcoded auf SNAI_V5 und robust_fread für Netzwerk-Mounts
        $magic = $this->robust_fread($fp, 8);
        if ($magic !== str_pad("SNAI_V5", 8, "\0")) { fclose($fp); return false; }
        
        $h_data = $this->robust_fread($fp, 20);
        if (strlen($h_data) < 20) { fclose($fp); return false; }
        
        $p = unpack('Vvocab/Vdim/Vffn/Vctx/Vlayers', $h_data);
        $vocab = $p['vocab']; $dim = $p['dim']; $ffn = $p['ffn']; $ctx = $p['ctx']; $layers = $p['layers'];
        fclose($fp);

        // V5 Size Calculation: Jedes Feld (w) hat nun ein (m) und (v) Array -> Faktor 3
        $expected_size = 28; 
        $expected_size += ($vocab * $dim * 4) * 3; // embeddings
        $expected_size += ($vocab * $dim * 4) * 3; // w_out
        $expected_size += ($dim * 4) * 3;          // rms_final
        
        $layer_size = ($dim * 4) * 3 * 2;                 // rms_att, rms_ffn
        $layer_size += ($dim * $dim * 4) * 3 * 4;         // w_q, w_k, w_v, w_o
        $layer_size += ($ffn * $dim * 4) * 3 * 2;         // w_up, w_down
        
        $expected_size += $layer_size * $layers;

        clearstatcache();
        $actual_size = filesize($file);

        // AUDIT-FIX (Session-Gespraech 2026-08-09, Projekt-Review): der
        // Nim-Loader akzeptiert seit jeher ZWEI gueltige Groessen - mit und
        // ohne den optionalen int32-Trailer fuer adam_t (siehe cortex.nim,
        // "withT": payload == wBytes*3 ODER wBytes*3 + 4). Diese Pruefung hier
        // kannte nur die Variante OHNE Trailer. Seit save_model_stream() den
        // Trailer tatsaechlich schreibt (adam_t-Persistenz, selbe Session),
        // war damit JEDER frisch trainierte Checkpoint exakt 4 Bytes "zu
        // gross" -> wurde hier als KORRUPT eingestuft und von den Aufrufern
        // (distill_princess/distill_apex_entity) per @unlink() GELOESCHT.
        // Empirisch reproduziert: 1.576.736 statt 1.576.732 Bytes -> abgelehnt.
        // Auswirkung waere gewesen: der Schwarm haette jedes neu trainierte
        // Modell beim naechsten Synthese-Versuch stillschweigend vernichtet.
        // Fix: dieselbe Toleranz wie der Loader, nicht mehr und nicht weniger.
        if ($actual_size !== $expected_size && $actual_size !== $expected_size + 4) {
            echo "[Queen] 🚨 KORRUPTION ENTDECKT in " . basename($file) . "!\n";
            echo "  > Soll-Größe: " . number_format($expected_size) . " Bytes (V5, optional +4 fuer adam_t)\n";
            echo "  > Ist-Größe:  " . number_format($actual_size) . " Bytes\n";
            return false;
        }
        return true;
    }

    // Update 2026-08-14 (Session-Entscheidung, docs/ROADMAP.md): Epochenzahl ab
    // Matriarchin aufwaerts angehoben (3 -> 6 bei Matriarchin, 3 -> 5 bei
    // Empress), UNTER DER BEDINGUNG, dass execute_calibration() ab dieser
    // Stufe nicht mehr dieselbe Kalibrierungsdatei fuer alle Epochen
    // wiederverwendet, sondern jede Epoche frisches, noch nicht in diesem
    // Lauf gezogenes Material bekommt (siehe execute_calibration() weiter
    // unten). Grund: Parameterzahl waechst pro Stufe staerker als die bisher
    // gleichbleibende 'nectar'-Dateizahl (Matriarchin ~21 Mio. Parameter,
    // Empress ~117 Mio. - Faktor ~5,6x - waehrend 'nectar' nur 10->20, Faktor
    // 2x, waechst). Ohne mehr *einzigartige* Tokens verschiebt mehr Kapazitaet
    // die Balance Richtung Memorierung statt Generalisierung (Chinchilla-
    // Ueberlegung, Session-Diskussion). Reine Epochen-Erhoehung OHNE den
    // Frisch-Nektar-Fix waere kontraproduktiv gewesen (nur mehr Wiederholung
    // derselben kleinen Datei) - deshalb beide Aenderungen nur gemeinsam.
    // Prinzessin bewusst unveraendert (dort nicht angefragt, Pipeline dort
    // seit laengerem stabil eingespielt).
    private const EVOLUTION_LADDER = [
        // Layer-Zahlen ab Prinzessin (Stand 2026-08-16, Session-Diskussion "512-Dim/32-
        // Layer-Gedankenexperiment"): vorher ueberall konstantes Verhaeltnis Dim/Layer=128
        // ueber 16x Groessenspanne (256..4096 Dim) - das traf nur fuer Goddess/Demi Goddess
        // tatsaechlich echte Referenzarchitekturen (LLaMA-7B: 4096/32=128). Bei kleiner
        // Skala nutzen real durchgetunte Modelle ein niedrigeres Verhaeltnis - konkret
        // verifiziert an Pythia-70M (EleutherAI): exakt 512 Dim, exakt headSize=64 wie hier,
        // aber 6 statt 4 Layer (Verhaeltnis 85 statt 128). Prinzessin/Matriarchin/Empress
        // linear auf Pythia-70Ms eigenes Verhaeltnis (512/6) skaliert - ergibt exakte
        // Ganzzahlen ohne Rundung. Demi Goddess/Goddess bewusst UNVERAENDERT (Verhaeltnis
        // 128 dort bereits die richtige Referenzklasse).
        'Drohne'       => ['src_prefix' => 'none',        'src_ext' => '',        'dim' => 128,  'ctx' => 512,  'layers' => 2,  'nectar' => 0,   'epochs' => 0,   'target_loss' => 0.0],
        'Prinzessin'   => ['src_prefix' => 'expert',      'src_ext' => '.snai',   'dim' => 256,  'ctx' => 1024, 'layers' => 3,  'nectar' => 5,   'epochs' => 3,   'target_loss' => 0.0],
        // 2026-08-27: Epochen 6 -> 13. Der Grund ist NICHT "mehr ist besser",
        // sondern eine gemessene Schwelle: Prompt-Bindung setzte bei 256 Dim
        // erst bei rund 200 MB KUMULIERTEM SFT ein (Sprung topic_own 3/30 ->
        // 29/30 zwischen Epoche 6 und 8 des princess_sftmix-Arms). Mit dem
        // neuen Byte-Ziel liefert die Matriarchin ~16 MB SFT je Epoche; 6
        // Epochen ergaeben 96 MB und damit garantiert wieder einen Nullbefund,
        // 13 Epochen ergeben ~208 MB. Der abgebrochene Lauf f50702 hatte 42 MB
        // (docs/inferenz/INFERENZ_MATRIARCHIN_F50702_REFERENZ.md).
        // Kosten: ~15 h je Epoche bei 20 MB, also ~8 Tage fuer die Stufe.
        // Rueckbau ist diese eine Zahl. Die uebrigen Tiers bleiben unangetastet -
        // fuer sie ist die Schwelle nicht gemessen, und ein Raten waere hier
        // besonders teuer.
        'Matriarchin'  => ['src_prefix' => 'princess',    'src_ext' => '.cortex', 'dim' => 512,  'ctx' => 2048, 'layers' => 6,  'nectar' => 10,  'epochs' => 13,  'target_loss' => 0.0],
        'Empress'      => ['src_prefix' => 'matriarchin', 'src_ext' => '.apex',   'dim' => 1024, 'ctx' => 4096, 'layers' => 12, 'nectar' => 20,  'epochs' => 5,   'target_loss' => 0.0],
        'Demi Goddess' => ['src_prefix' => 'empress',     'src_ext' => '.apex',   'dim' => 2048, 'ctx' => 8192, 'layers' => 16, 'nectar' => 40,  'epochs' => 3,   'target_loss' => 0.0],
        'Goddess'      => ['src_prefix' => 'demigoddess', 'src_ext' => '.apex',   'dim' => 4096, 'ctx' => 8192, 'layers' => 32, 'nectar' => 80,  'epochs' => 200, 'target_loss' => 3.0]
    ];

    private $expert_dir = './factory/experts/';
    private $tribute_dir = './factory/tributes/';
    private $royal_dir = './factory/royal/';
    private $fragments_dir = './factory/royal_fragments/';
    private $checkpoint_dir = './factory/checkpoints/';
    private $archive_dir = './factory/archive/';
    private $apex_dir = './factory/apex/'; 

    public function __construct() {
        foreach ([$this->expert_dir, $this->tribute_dir, $this->royal_dir, $this->fragments_dir, $this->checkpoint_dir, $this->archive_dir, $this->apex_dir] as $d) {
            if (!is_dir($d)) @mkdir($d, 0777, true);
        }
    }

    private function log_synthesis(string $entity_type, string $caste_tag, string $uid, string $loss) {
        $log_file = './factory/royal_synthesis.log';
        $timestamp = date('Y-m-d H:i:s');
        $entry = sprintf("[%s] KASTE: %-8s | TYP: %-12s | UID: %-8s | END-LOSS: %s\n", $timestamp, $caste_tag, $entity_type, $uid, $loss);
        file_put_contents($log_file, $entry, FILE_APPEND | LOCK_EX);
    }

    public function distill_princess(string $uid) {
        $tributes = glob($this->tribute_dir . '*.snai');
        
        $valid_tributes = [];
        foreach ($tributes as $t) {
            if ($this->validate_tensor_file($t)) {
                $valid_tributes[] = $t;
            } else {
                @unlink($t); 
                @unlink($this->expert_dir . basename($t)); 
            }
        }

        $groups = [];
        foreach ($valid_tributes as $t) {
            $tag = 'GEN';
            if (preg_match('/expert_([A-Z]+)_/', basename($t), $m)) $tag = $m[1];
            $groups[$tag][] = $t;
        }

        $target_tag = null;
        $selected_tributes = [];

        foreach ($groups as $tag => $files) {
            if (count($files) < 3) continue;
            // AUDIT-FIX: erst auf Head-Struktur-Kompatibilitaet filtern, dann erst
            // die 3 nehmen - sonst mittelt merge_tributes() ggf. Modelle mit
            // unterschiedlicher Attention-Struktur zusammen (siehe Kommentar an
            // largest_head_compatible_group()).
            $compatible = $this->largest_head_compatible_group($files, 3);
            if (count($compatible) < 3) continue;
            usort($compatible, function($a, $b) { return filemtime($b) <=> filemtime($a); });
            $target_tag = $tag;
            $selected_tributes = array_slice($compatible, 0, 3);
            break;
        }

        if (!$target_tag) {
            echo "[Queen] Abbruch: Keine 3 sauberen, validen UND head-strukturkompatiblen Tribute im VIP-Raum gefunden.\n";
            return false;
        }

        $uid = str_replace('gen_', '', $uid); 
        $session_id = uniqid('royal_');
        
        $proto_cortex = $this->checkpoint_dir . "proto_{$session_id}.snai";
        $expanded_cortex = $this->checkpoint_dir . "expanded_{$session_id}.snai";
        $final_cortex = $this->royal_dir . "princess_{$target_tag}_{$uid}.cortex";
        
        echo "[Queen] Extrahiere Essenzen aus 3 '{$target_tag}'-Tributen...\n";
        
        try {
            $this->merge_tributes($selected_tributes, $proto_cortex);
        } catch (Exception $e) {
            echo "  > 🚨 [Queen] Synthese-Fehler: " . $e->getMessage() . "\n";
            @unlink($proto_cortex);
            return false;
        }

        if (!file_exists($proto_cortex)) return false;

        $conf = self::EVOLUTION_LADDER['Prinzessin'];
        
        try {
            if (!$this->upscale_cortex($proto_cortex, $expanded_cortex, $conf['dim'], $conf['ctx'], $conf['layers'])) {
                @unlink($proto_cortex);
                return false;
            }
        } catch (Exception $e) {
             echo "  > 🚨 [Queen] Expansion-Fehler: " . $e->getMessage() . "\n";
             @unlink($proto_cortex);
             @unlink($expanded_cortex);
             return false;
        }

        // MULTI-HEAD SCALING: die Head-Zahl skaliert mit der Net2Net-Synthese statt
        // eingefroren zu bleiben - ABER NUR, wenn die Quelle bereits headSize=64-
        // konform ist. suggest_num_heads() zielt auf headSize=64, und weil
        // upscale_matrix_stream() alte Gewichte als reinen Top-Links-Block erhaelt
        // (Zeilen [0:old_dim) bleiben exakt an ihrer Stelle), fallen bei einer
        // bereits headSize=64-strukturierten Quelle die alten Head-Grenzen exakt
        // mit einem Praefix der neuen Heads zusammen - die neuen Heads sind reine
        // Zusatzkapazitaet, nichts Bestehendes wird umgedeutet.
        // Bei einer NICHT so strukturierten Quelle (z.B. Legacy-Drohnen mit
        // num_heads=1/headSize=voller-dim, die es auf diesem System noch aktiv
        // gibt) waere ein Hochskalieren auf suggest_num_heads($conf['dim']) genau
        // der Bug, den wir eigentlich verhindern wollen: eine einzelne globale
        // Attention wuerde faelschlich als mehrere unabhaengige 64er-Heads
        // umgedeutet. Deshalb: nur wachsen, wenn die Quelle schon standardkonform
        // war, sonst deren Head-Zahl unveraendert uebernehmen (nicht neu raten).
        $source_header = load_model_header($selected_tributes[0]);
        $source_heads = resolve_num_heads($selected_tributes[0], $source_header['hidden_dim']);
        $expected_source_heads = suggest_num_heads($source_header['hidden_dim']);
        if ($source_heads === $expected_source_heads) {
            $new_heads = suggest_num_heads($conf['dim']);
            echo "[Queen] Head-Skalierung: Quelle bereits headSize=64-konform ($source_heads Heads bei dim={$source_header['hidden_dim']}) -> wachse auf $new_heads Heads (dim={$conf['dim']}).\n";
        } else {
            $new_heads = $source_heads;
            echo "[Queen] Head-Skalierung uebersprungen: Quelle hat $source_heads Heads (Legacy/Non-Standard, headSize=64 erwartet $expected_source_heads) -> Struktur unveraendert uebernommen, um gelernte Attention nicht umzudeuten.\n";
        }
        write_num_heads($expanded_cortex, $new_heads);

        if (!$this->execute_calibration($session_id, $expanded_cortex, $final_cortex, $uid, "Prinzessin", $target_tag)) {
            echo "  > [Rollback] Synthese fehlgeschlagen. Original-Experten bleiben unangetastet!\n";
            @unlink($proto_cortex);
            @unlink($expanded_cortex);
            return false;
        }
        
        @unlink($proto_cortex);
        @unlink($expanded_cortex);
        
        foreach ($selected_tributes as $tribute) {
            $base_name = basename($tribute);
            $src = $this->expert_dir . $base_name;
            $dst = $this->fragments_dir . $base_name;
            @rename($src, $dst);
            // AUDIT-FIX: .heads-Sidecar blieb beim Verbrauchen der Tribut-Drohne
            // als Waise in factory/experts/ zurueck (nur die .snai wanderte nach
            // royal_fragments/) - dieselbe Fehlerklasse wie beim REFLECT-/DIGEST-
            // Rename und dem Tributes-Kopieren, hier nur uebersehen.
            @rename(head_sidecar_path($src), head_sidecar_path($dst));
            @unlink($tribute);
        }
        return true;
    }

    public function distill_apex_entity(string $uid, string $target_tier) {
        if (!isset(self::EVOLUTION_LADDER[$target_tier])) return false;

        $config = self::EVOLUTION_LADDER[$target_tier];
        $src_prefix = $config['src_prefix'];
        $src_ext = $config['src_ext'];
        
        $search_dir = ($target_tier === 'Matriarchin') ? $this->royal_dir : $this->apex_dir;
        $candidates = glob($search_dir . $src_prefix . '_*' . $src_ext);

        $valid_candidates = [];
        foreach ($candidates as $c) {
            if ($this->validate_tensor_file($c)) $valid_candidates[] = $c;
            else @unlink($c);
        }

        if (count($valid_candidates) < 3) return false;

        // AUDIT-FIX: siehe distill_princess() - erst auf Head-Struktur-
        // Kompatibilitaet filtern, dann erst in SFT/General aufteilen. Sonst
        // koennten sft[0]/general[0]/general[1] aus unterschiedlichen Head-
        // Strukturen stammen und merge_tributes() wuerde sie trotzdem mitteln.
        $compatible_candidates = $this->largest_head_compatible_group($valid_candidates, 3);
        if (count($compatible_candidates) < 3) return false;

        $sft_models = [];
        $general_models = [];

        foreach ($compatible_candidates as $c) {
            if (preg_match('/' . $src_prefix . '_SFT_/', basename($c))) {
                $sft_models[] = $c;
            } else {
                $general_models[] = $c;
            }
        }

        if (count($sft_models) < 1 || count($general_models) < 2) return false;

        $selected_for_merge = [
            $sft_models[0],     
            $general_models[0], 
            $general_models[0], 
            $general_models[1], 
            $general_models[1]  
        ];

        // AUDIT-FIX: Entfernt array_unique, damit die gewollte Duplikats-Gewichtung 
        // auch bei der Wahl des Namens-Tags ('dominant_tag') berücksichtigt wird.
        $tags = [];
        foreach ($selected_for_merge as $f) {
            if (preg_match('/' . $src_prefix . '_([A-Z]+)_/', basename($f), $m)) $tags[] = $m[1];
        }
        
        $counts = array_count_values($tags);
        arsort($counts);
        $dominant_tag = array_key_first($counts); 

        $session_id = uniqid('apex_');
        $proto_apex = $this->checkpoint_dir . "proto_{$target_tier}_{$session_id}.cortex";
        $expanded_apex = $this->checkpoint_dir . "expanded_{$target_tier}_{$session_id}.apex";
        $clean_tier_name = str_replace(' ', '', strtolower($target_tier));
        $final_apex = $this->apex_dir . "{$clean_tier_name}_{$dominant_tag}_{$uid}.apex";
        
        echo "\n====================================================\n";
        echo "[Queen] APEX-SYNTHESE: Erschaffe eine '{$dominant_tag}' {$target_tier}...\n";
        echo "====================================================\n";
        
        try {
            $this->merge_tributes($selected_for_merge, $proto_apex);
        } catch (Exception $e) {
            echo "  > 🚨 [Queen] Synthese-Fehler: " . $e->getMessage() . "\n";
            @unlink($proto_apex);
            return false;
        }

        if (!file_exists($proto_apex)) return false;
        
        try {
            if (!$this->upscale_cortex($proto_apex, $expanded_apex, $config['dim'], $config['ctx'], $config['layers'])) {
                @unlink($proto_apex);
                return false;
            }
        } catch (Exception $e) {
             echo "  > 🚨 [Queen] Expansion-Fehler: " . $e->getMessage() . "\n";
             @unlink($proto_apex);
             @unlink($expanded_apex);
             return false;
        }

        // MULTI-HEAD SCALING: siehe distill_princess() fuer die volle Begruendung.
        // Nur wachsen, wenn die Quelle schon headSize=64-konform ist, sonst
        // deren Head-Zahl unveraendert uebernehmen.
        $source_header = load_model_header($selected_for_merge[0]);
        $source_heads = resolve_num_heads($selected_for_merge[0], $source_header['hidden_dim']);
        $expected_source_heads = suggest_num_heads($source_header['hidden_dim']);
        if ($source_heads === $expected_source_heads) {
            $new_heads = suggest_num_heads($config['dim']);
            echo "[Queen] Head-Skalierung: Quelle bereits headSize=64-konform ($source_heads Heads bei dim={$source_header['hidden_dim']}) -> wachse auf $new_heads Heads (dim={$config['dim']}).\n";
        } else {
            $new_heads = $source_heads;
            echo "[Queen] Head-Skalierung uebersprungen: Quelle hat $source_heads Heads (Legacy/Non-Standard, headSize=64 erwartet $expected_source_heads) -> Struktur unveraendert uebernommen, um gelernte Attention nicht umzudeuten.\n";
        }
        write_num_heads($expanded_apex, $new_heads);

        if (!$this->execute_calibration($session_id, $expanded_apex, $final_apex, $uid, $target_tier, $dominant_tag)) {
            echo "  > [Rollback] Synthese fehlgeschlagen. Vorgänger-Modelle bleiben unangetastet!\n";
            @unlink($proto_apex);
            @unlink($expanded_apex);
            return false;
        }
        
        @unlink($proto_apex);
        @unlink($expanded_apex);

        // AUDIT-FIX: $unique_selected war nirgends definiert (die array_unique()-Zeile
        // wurde beim Tags-Fix entfernt, dieser Loop hing noch daran) - verbrauchte
        // Vorgaenger-Modelle wanderten dadurch nie nach royal_fragments/ und blieben
        // fuer immer als Kandidaten sichtbar (Endlosschleife auf derselben Generation).
        $unique_selected = array_unique($selected_for_merge);
        foreach ($unique_selected as $p) {
            $dst = $this->fragments_dir . basename($p);
            @rename($p, $dst);
            // AUDIT-FIX: siehe distill_princess() - .heads-Sidecar muss mit
            // umziehen, sonst bleibt sie als Waise zurueck.
            @rename(head_sidecar_path($p), head_sidecar_path($dst));
        }
        return true;
    }

    private function execute_calibration($session_id, $expanded_path, $final_path, $uid, $title, $tag): bool {
        $config = self::EVOLUTION_LADDER[$title];

        $nectar_files = $config['nectar'];
        $epochs = $config['epochs'];
        $target_loss = $config['target_loss'];
        // AUDIT-FIX (2026-08-16, RAM-Checkpoints, siehe ram_checkpoint_config.php):
        // Prinzessin (256 Dim) und darunter bekommen ihre Worker-/Epochen-
        // Checkpoints auf tmpfs statt SSD, Matriarchin aufwaerts unveraendert.
        $use_ram_checkpoints = snai_should_use_ram_checkpoints($config['dim']);

        $archives = glob($this->archive_dir . '*.txt');
        $inputs = glob('./factory/input/*.txt');
        $all_data = array_merge($archives, $inputs);

        // Lang-Material (2026-08-24). factory/input_long/ enthaelt SFT-Paare mit
        // Turns bis 4000 Zeichen (~1140 Token) statt der sonst geltenden 1500
        // (~430). Das ist noetig, weil die grossen deutschen RAG-Datensaetze
        // durchweg darueber liegen - avemio extraction-recall 11,2 KB/Zeile,
        // ocr-correction 7,8 KB - und am alten Limit vollstaendig verworfen
        // wurden.
        // Gate ueber ctx statt ueber den Tier-Namen: 4000 Zeichen passen in
        // Ctx 2048 (Matriarchin aufwaerts) und liessen der Antwort noch Platz,
        // wuerden das 1024er-Fenster der Prinzessin aber sprengen. Waechst der
        // Ladder, gilt die Regel automatisch weiter.
        if (($config['ctx'] ?? 0) >= 2048) {
            $long = glob('./factory/input_long/*.txt') ?: [];
            if ($long) {
                $all_data = array_merge($all_data, $long);
                echo "[Queen] Lang-Material zugelassen (Ctx {$config['ctx']}): " . count($long) . " zusaetzliche Dateien.\n";
            }
        }

        // 2026-08-23: von 0.20 auf 0.50 angehoben. Der Chinchilla-Lauf
        // (~11 % SFT nach Bytes) entwickelte ueber eine Halbierung der
        // Perplexitaet hinweg KEINE Prompt-Konditionierung, der Kontrolllauf
        // mit ~69 % SFT schon - Belege in
        // docs/inferenz/INFERENZ_SFTMIX_KONTROLLLAUF_E1_E3.md. Der alte Wert
        // war damit die gemessene Ursache dafuer, dass Modelle auf ALLEN
        // Tiers am Prompt vorbeischreiben.
        // Warum 0.50 und nicht knapp darueber: eine Chinchilla-Epoche lief
        // zufaellig mit 29,3 % SFT und blieb trotzdem beim Nullbefund
        // (topic_lift 0,67) - eine Verdopplung reicht nachweislich nicht.
        // ACHTUNG: die Quote zaehlt DATEIEN, nicht Bytes. SFT-Dateien sind mit
        // ~360 KB deutlich kleiner als GEN-Dateien mit ~2 MB, der Byte-Anteil
        // liegt also klar unter dem Dateianteil (bei 0.20 waren es ~11 %).
        // 2026-08-27: Quote von einer DATEI- auf eine BYTE-Zielgroesse umgestellt.
        //
        // Anlass ist die Messung vom 27.08.: Prompt-Bindung setzt bei 256 Dim
        // schwellwertartig ein (topic_own 3/30 -> 29/30 zwischen Epoche 6 und 8),
        // und der Arm, der sie erreicht hat, lief mit **77,8 % SFT nach Bytes**.
        // Das ist damit die Zielgroesse - nicht ein Dateianteil.
        //
        // Warum die alte Datei-Quote das nicht treffen konnte: derselbe Wert 0.50
        // ergab je Tier einen voellig anderen Byte-Anteil, weil die Pools
        // unterschiedlich grosse Dateien enthalten (gemessen 27.08.):
        //   Prinzessin (nur factory/input/):   SFT-Median 1,11 MB, GEN 2,00 MB -> ~36 %
        //   Matriarchin (+ factory/input_long/): SFT-Median 2,00 MB, GEN 2,00 MB -> ~47 %
        // Der Lang-Ordner enthaelt 435 SFT-Dateien und KEINE GEN-Datei; ab Ctx 2048
        // verschiebt er den Pool also einseitig. Eine feste Dateizahl ist damit
        // keine steuerbare Groesse, sondern ein Nebenprodukt der Ordnerbelegung.
        //
        // Die Dateizahl wird jetzt aus dem Byte-Ziel zurueckgerechnet, mit den
        // TATSAECHLICHEN mittleren Dateigroessen beider Pools:
        //   n_sft = N * t * b / ( s*(1-t) + t*b )
        // (t = Byte-Ziel, s/b = mittlere SFT-/GEN-Dateigroesse, N = Dateien/Epoche)
        // Mindestens eine Datei je Pool bleibt erhalten - ein reiner SFT-Korpus
        // ist ausdruecklich NICHT das Ziel, der erfolgreiche Arm hatte 22 % GEN.
        $sft_byte_target = 0.78;

        $sft_pool = [];
        $base_pool = [];

        foreach ($all_data as $file) {
            if (strpos(basename($file), "hf_SFT_") !== false) $sft_pool[] = $file;
            else $base_pool[] = $file;
        }

        $mean_size = function (array $files): float {
            if (!$files) return 0.0;
            // Stichprobe statt aller Dateien: bei 14 000 GEN-Dateien waeren das
            // ebenso viele stat()-Aufrufe pro Kalibrierung, ohne Mehrwert.
            $sample = $files;
            shuffle($sample);
            $sample = array_slice($sample, 0, 50);
            $sum = 0;
            foreach ($sample as $f) $sum += (int)@filesize($f);
            return $sum / max(1, count($sample));
        };

        $s = $mean_size($sft_pool);
        $b = $mean_size($base_pool);

        if ($s > 0 && $b > 0) {
            $t = $sft_byte_target;
            $sft_quota = (int)round($nectar_files * $t * $b / ($s * (1 - $t) + $t * $b));
            $sft_quota = max(1, min($nectar_files - 1, $sft_quota));
        } else {
            // Ein Pool leer: nicht rechnen, sondern auf die alte Haelfte-Regel
            // zurueckfallen. draw_epoch_slice() kommt mit einem leeren Pool klar.
            $sft_quota = max(1, (int)round($nectar_files * 0.50));
        }
        $base_quota = $nectar_files - $sft_quota;

        $expected_share = ($s * $sft_quota + $b * $base_quota) > 0
            ? 100 * $s * $sft_quota / ($s * $sft_quota + $b * $base_quota)
            : 0;
        printf("[Queen] SFT-Mix: %d/%d Dateien = %.1f %% nach Bytes (Ziel %.0f %%; "
             . "SFT-Datei ~%.2f MB, GEN-Datei ~%.2f MB)\n",
               $sft_quota, $nectar_files, $expected_share, $sft_byte_target * 100,
               $s / 1048576, $b / 1048576);

        shuffle($sft_pool);
        shuffle($base_pool);

        echo "[Queen] Bereite Kalibrierungs-Nektar für $title '$tag' vor ($nectar_files Artefakte/Epoche, $epochs Epochen)...\n";

        // Update 2026-08-10 (Session-Gespraech, Folge auf den 2026-08-09-Versuch):
        // Prinzessin zurueck auf 1 Worker. Der 2-Worker-Versuch (roadmap-notiert
        // als "braucht einen zweiten Vergleichslauf") hat sich mit dem zweiten
        // unabhaengigen Lauf erhaertet statt aufgeloest: zwei 2-Worker-Prinzessinnen
        // (unterschiedlicher Nektar, unterschiedliche Startzeit) landeten bei
        // Held-out-Loss 5.9640 bzw. 5.9644 nach Epoche 3 - eine Uebereinstimmung
        // auf vier Nachkommastellen, waehrend die beiden 1-Worker-Vorgaenger bei
        // 5.6613/5.6991 lagen. Gleichzeitig wuchs die Generalisierungsluecke
        // (Held-out minus Train-Loss) von ~0.36 auf ~0.53, obwohl der Train-Loss
        // selbst besser wurde - klassisches Overfitting-Muster. Plausibelste
        // Ursache unveraendert: die FedAvg-artige Korpus-Aufteilung auf 2 Worker
        // bei nur 3 Epochen. Siehe docs/ROADMAP.md, Punkt 4 der
        // "Engine-/Framework-Verbesserungen" fuer die volle Herleitung.
        $workers = 1;
        // AUDIT: vorher bekamen ALLE Tiers ab Matriarchin dieselbe LR (0.0002),
        // egal ob 512 oder 4096 Dim - Empress/DemiGoddess/Goddess hatten also nie
        // eine eigens abgestimmte Rate, sondern liefen blind im "else"-Zweig mit.
        // Prinzessins 0.0001 ist ein bereits eingespielter, separat gepflegter
        // Wert und bleibt bewusst unangetastet. Fuer die uebrigen Tiers wird die
        // Rate jetzt mit 1/sqrt(dim) skaliert (dieselbe Skalierungsfamilie wie
        // die Gewichts-Initialisierung in init_base.php), verankert an
        // Matriarchins bisherigem Wert 0.0002 bei 512 Dim - siehe
        // docs/ROADMAP.md, Abschnitt "Learning-Rate-Skalierung nach Tier-Groesse".
        if ($title === "Prinzessin") {
            $custom_lr = "0.0001";
        } else {
            $anchor_dim = self::EVOLUTION_LADDER['Matriarchin']['dim']; // 512
            $anchor_lr  = 0.0002;
            $custom_lr  = (string)($anchor_lr * sqrt($anchor_dim / $config['dim']));
        }
        $dtype = SNAI_TYPE_FP32;

        // Update 2026-08-11 (Session-Entscheidung): Weight-Decay/Dropout waren seit
        // dem CLI-Wiring (siehe docs/ROADMAP.md) nur ueber einen gezielten manuellen
        // train_cluster.php-Aufruf aktivierbar, hier bewusst auf 0.0 (No-Op) belassen.
        // Auf Nutzerwunsch jetzt fuer alle Tiers im automatischen Pfad scharf
        // geschaltet, konservative Startwerte (0.01/0.1).
        $weight_decay = "0.01";
        $dropout_rate = "0.1";
        // Update 2026-08-15 (Session-Entscheidung, docs/ROADMAP.md, "Naechste
        // Schritte" Punkt 7): analog Weight-Decay/Dropout fuer alle Tiers im
        // automatischen Pfad scharf geschaltet, ueblicher Startwert eps=0.1.
        $label_smoothing = "0.1";

        $combined_output = '';

        // Update 2026-08-16 (Session-Entscheidung, docs/ROADMAP.md): Prinzessin
        // war beim Frisch-Nektar-Umbau vom 2026-08-14 bewusst ausgenommen und
        // trainierte seitdem als einziger Tier 3 Epochen lang auf derselben
        // einen ~10-MB-Kalibrierungsdatei statt frischem Material pro Epoche.
        // Bei ~6-8 Mio. Parametern (256 Dim, 3 Layer) sind das grob 1-2% des
        // Chinchilla-Richtwerts (~20 Tokens/Parameter) - vermutlich der
        // dominante Grund, warum die Regularisierungs-/Korpus-Fixes bei der
        // Prinzessin bisher keine sichtbare Verbesserung zeigten (Inferenztest
        // 2026-08-16, docs/inferenz/INFERENZ_PRINZESSIN_6A817AD0DC1D1_FINAL.md).
        // Sonderfall entfernt, Prinzessin nutzt jetzt denselben
        // Frisch-Nektar-pro-Epoche-Mechanismus wie Matriarchin aufwaerts: jede
        // Epoche sieht frisches, in diesem Lauf noch nicht gezogenes Material
        // statt derselben einen Datei fuer alle Epochen. Verkettet ueber den
        // vorhandenen Checkpoint-Resume-Mechanismus (start_epoch/
        // train_cluster.php benennt seinen Epochen-Checkpoint am Ende direkt in
        // $final_path um, siehe train_cluster.php - der naechste Aufruf kann
        // ihn also direkt als eigenes base_model wiederverwenden).
        // $lr_schedule_epochs haelt die Warmup+Cosine-Decay-Kurve dabei ueber
        // den GESAMTEN verketteten Lauf glatt, obwohl jeder Einzelaufruf nur
        // eine Epoche durchlaeuft (start_epoch == max_epochs fuer diesen Call).
        //
        // Bekannte Nebenwirkung (nicht behoben, da kosmetisch): jeder
        // Einzelaufruf haengt eine eigene "epoch":"TOTAL"-Zeile an
        // train_<session>.jsonl an, statt einer einzigen fuer den ganzen Lauf -
        // metrics_report.php nimmt beim Parsen ohnehin die jeweils letzte
        // TOTAL-Zeile, zeigt danach also korrekt die letzte Epoche, aber mit
        // der Laufzeit/epochs_completed NUR dieser letzten Einzel-Epoche statt
        // der Summe ueber den ganzen Lauf.
        $current_base = $expanded_path;
        for ($epoch = 1; $epoch <= $epochs; $epoch++) {
            $sft_slice = $this->draw_epoch_slice($sft_pool, $sft_quota, $epoch);
            $base_slice = $this->draw_epoch_slice($base_pool, $base_quota, $epoch);
            $selected = array_merge($sft_slice, $base_slice);

            $calib_file = $this->checkpoint_dir . "calib_{$session_id}_e{$epoch}.txt";
            if (!$this->write_calib_file($calib_file, $selected)) return false;

            $mb_size = round(filesize($calib_file) / 1024 / 1024, 2);
            echo "[Queen] Epoche $epoch/$epochs | frischer Nektar: {$mb_size} MB (" . count($selected) . " Dateien)...\n";

            $result = $this->run_calibration_epoch(
                $calib_file, $final_path, $current_base,
                $epoch, $epoch, $epochs, $workers, $dtype, $custom_lr, $target_loss,
                $weight_decay, $dropout_rate, $label_smoothing, $use_ram_checkpoints
            );
            @unlink($calib_file);

            if (!$result['success']) {
                echo "🚨 [Queen] FEHLSCHLAG beim Fine-Tuning in Epoche $epoch. Worker-Cluster crashte mit Code {$result['exit_code']}.\n";
                @unlink($final_path);
                return false;
            }
            $combined_output .= $result['output'];
            $current_base = $final_path; // naechste Epoche startet vom eben geschriebenen Checkpoint
        }

        if (file_exists($final_path)) {
            echo "[Queen] ERFOLG: $title '{$uid}' wurde geboren!\n";
            if (preg_match_all('/\[Metrik\] Globaler Avg Loss:\s*([0-9\.]+)/', $combined_output, $matches)) {
                $loss_value = end($matches[1]);
                echo "  > [Kalibrierung] Nektar verdaut. Post-Expansion Loss: " . $loss_value . "\n";
                $this->log_synthesis($title, $tag, $uid, $loss_value);
            } else {
                $this->log_synthesis($title, $tag, $uid, "PARSE_ERROR");
            }
            return true;
        } else {
            echo "🚨 [Queen] FEHLSCHLAG. Kalibrierung beendet, aber Modelldatei fehlt.\n";
            return false;
        }
    }

    /**
     * Zieht ein Epochen-Segment aus $pool, ohne Wiederholung innerhalb eines
     * Laufs, solange der (vorab gemischte) Pool reicht - darueber hinaus
     * zyklisch (Modulo), mit einer einmaligen Warnung statt hartem Abbruch.
     */
    private function draw_epoch_slice(array $pool, int $quota, int $epoch): array {
        $pool_size = count($pool);
        if ($pool_size === 0 || $quota <= 0) return [];
        $start = ($epoch - 1) * $quota;
        if ($start >= $pool_size && $start % $pool_size < $quota) {
            echo "  > ⚠️ [Queen] Nektar-Pool erschöpft, wiederhole Dateien ab Epoche $epoch.\n";
        }
        $slice = [];
        for ($i = 0; $i < $quota; $i++) {
            $slice[] = $pool[($start + $i) % $pool_size];
        }
        return $slice;
    }

    private function write_calib_file(string $calib_file, array $files): bool {
        $out_fp = @fopen($calib_file, 'wb');
        if (!$out_fp) return false;

        foreach ($files as $file) {
            $in_fp = @fopen($file, 'rb');
            if ($in_fp) {
                stream_copy_to_stream($in_fp, $out_fp);
                fwrite($out_fp, "\n");
                fclose($in_fp);
            }
        }
        fclose($out_fp);

        if (filesize($calib_file) < 10) file_put_contents($calib_file, "Symbio: Ich bin erwacht.</s>");
        return true;
    }

    /**
     * Ein einzelner train_cluster.php-Aufruf fuer die Epochen-Spanne
     * [$start_epoch..$max_epochs_call] DIESES Prozesses. $lr_schedule_total
     * ist der Gesamt-Epochenwert des ganzen (ggf. verketteten) Laufs, fuer
     * die Warmup/Cosine-Kurve (siehe train_cluster.php, $lr_schedule_epochs).
     */
    private function run_calibration_epoch(
        string $calib_file, string $final_path, string $base_model,
        int $start_epoch, int $max_epochs_call, int $lr_schedule_total,
        int $workers, int $dtype, string $custom_lr, float $target_loss,
        string $weight_decay, string $dropout_rate, string $label_smoothing,
        bool $use_ram_checkpoints = false
    ): array {
        $arg_calib = escapeshellarg($calib_file);
        $arg_final = escapeshellarg($final_path);
        $arg_base  = escapeshellarg($base_model);
        $arg_loss  = escapeshellarg((string)$target_loss);

        // Update 2026-08-15 (docs/ROADMAP.md, "Naechste Schritte" Punkt 7):
        // label_smoothing als LETZTER positioneller Arg angehaengt, nach
        // $lr_schedule_total (Position 12) - sonst verschieben sich alle
        // train_cluster.php-argv[N]-Indizes.
        //
        // AUDIT-FIX (2026-08-16, RAM-Checkpoints, siehe ram_checkpoint_config.php):
        // ueber Env-Var statt weiterem Positions-Arg gesteuert, genau aus
        // demselben Grund wie oben (keine argv[N]-Verschiebung).
        $ckpt_env = $use_ram_checkpoints
            ? 'SNAI_CHECKPOINT_DIR=' . escapeshellarg(SNAI_RAM_CHECKPOINT_DIR) . ' '
            : '';
        $cmd = $ckpt_env . "php " . escapeshellarg(__DIR__ . '/train_cluster.php') . " $arg_calib $arg_final $workers $max_epochs_call $arg_base $dtype $custom_lr $arg_loss $start_epoch $weight_decay $dropout_rate $lr_schedule_total $label_smoothing 2>&1";

        $output = '';
        $exit_code = 1;
        $process = proc_open($cmd, [1 => ["pipe", "w"], 2 => ["pipe", "w"]], $pipes);
        if (is_resource($process)) {
            while ($line = fgets($pipes[1])) {
                $clean_line = trim($line);
                if ($clean_line !== '') echo "    > $clean_line\n";
                $output .= $line;
                flush();
            }
            fclose($pipes[1]);
            fclose($pipes[2]);
            $exit_code = proc_close($process);
        }

        return ['success' => $exit_code === 0, 'exit_code' => $exit_code, 'output' => $output];
    }

    private function merge_tributes(array $files, string $output_path) {
        $num_models = count($files);
        if ($num_models === 0) return;

        $handles = [];
        foreach ($files as $file) {
            $fp = @fopen($file, 'rb');
            if (!$fp) throw new Exception("Kann Tribut nicht lesen: $file");
            $handles[] = $fp;
        }

        $out_fp = @fopen($output_path, 'wb');
        if (!$out_fp) {
             foreach ($handles as $h) fclose($h);
             throw new Exception("Kann Output-Datei nicht öffnen: $output_path");
        }
        
        flock($out_fp, LOCK_EX);

        // AUDIT-FIX: robust_fread für den initialen Header (Netzwerk-Mount Schutz)
        $header = $this->robust_fread($handles[0], 28);
        fwrite($out_fp, $header);

        for ($i = 1; $i < $num_models; $i++) {
            if ($this->robust_fread($handles[$i], 28) !== $header) {
                foreach ($handles as $h) fclose($h);
                flock($out_fp, LOCK_UN);
                fclose($out_fp);
                throw new Exception("Inkompatible Header während der Verschmelzung.");
            }
        }

        echo "  > Verschmelze Multi-Layer Schichten (V5 Stream-Modus, nativ)...\n";

        // AUDIT: vorher PHP-seitiges unpack()/Schleifen-Mitteln pro einzelnem
        // Float (siehe docs/ROADMAP.md, "Native FP16/FP32-Tensor-Bibliothek").
        // Gleiche Chunk-Granularitaet (CHUNK_SIZE_BYTES) wie vorher, also
        // identisches Speicherprofil - nur die eigentliche Mittelung laeuft
        // jetzt als ein nativer Aufruf pro Chunk statt einer PHP-Schleife
        // ueber Hunderttausende Einzelwerte.
        while (!feof($handles[0])) {
            $chunks = [];
            for ($i = 0; $i < $num_models; $i++) {
                $chunks[] = $this->robust_fread($handles[$i], self::CHUNK_SIZE_BYTES);
            }

            if (strlen($chunks[0]) === 0) break;

            // AUDIT-FIX: Chunks am Dateiende koennten (bei korrupten/
            // ungleich langen Quellen) unterschiedlich lang sein - nur
            // volle Float-Elemente an die native Mittelung geben, sonst
            // liest sie ueber das Ende eines kuerzeren Puffers hinaus.
            // Im Normalfall (alle Dateien schon oben header-/laengengeprueft)
            // sind alle Chunks exakt gleich lang, das hier ist nur ein Netz.
            $min_len = min(array_map('strlen', $chunks));
            $usable_len = $min_len - ($min_len % 4);
            if ($usable_len === 0) break;
            if ($usable_len !== strlen($chunks[0])) {
                $chunks = array_map(fn($c) => substr($c, 0, $usable_len), $chunks);
            }

            fwrite($out_fp, SnaiTensorUtils::merge($chunks));
        }

        foreach ($handles as $h) fclose($h);
        flock($out_fp, LOCK_UN);
        fclose($out_fp);
    }

    /**
     * V5: Der Upscaler muss unterscheiden zwischen Gewichten (erhalten Noise) 
     * und Adam-States (erhalten strickt 0.0).
     */
    private function upscale_matrix_stream($in_fp, $out_fp, int $old_r, int $old_c, int $new_r, int $new_c, float $scale = 1.0, bool $is_adam_state = false, float $pad_center = 0.0) {
        // AUDIT: vorher eine PHP-Schleife mit einem fread()+unpack() PRO
        // EINZELNEM Float (siehe docs/ROADMAP.md, "Native FP16/FP32-Tensor-
        // Bibliothek") - bei alten/DemiGoddess-Groessenordnungen Millionen
        // Einzelaufrufe. Jetzt: alter Block in EINEM Rutsch einlesen, native
        // Bibliothek (snai_tensor_utils.nim) macht Skalierung + Rauschen/
        // Fuellwert fuer den gesamten neuen Tensor in einem Aufruf.
        $old_elements = $old_r * $old_c;
        $old_bytes = $old_elements * 4;

        $old_buffer = $old_bytes > 0 ? $this->robust_fread($in_fp, $old_bytes) : '';
        if (strlen($old_buffer) !== $old_bytes) {
            // AUDIT-FIX V5.1 (unveraendert uebernommen): vorher wurde hier
            // still 0.0 geschrieben. Die Ausgabedatei hatte dann die
            // korrekte Groesse und kam durch validate_tensor_file - gefuellt
            // mit Nullen.
            throw new Exception("upscale_matrix_stream: Quelldatei bricht ab (erwartet {$old_bytes} Bytes, erhalten " . strlen($old_buffer) . ").");
        }

        $new_buffer = SnaiTensorUtils::upscale(
            $old_buffer, $old_r, $old_c, $new_r, $new_c,
            $scale, self::NOISE_FACTOR, $pad_center, $is_adam_state
        );

        fwrite($out_fp, $new_buffer);
    }

    // Makro für V5: Schreibt immer w, m, und v nacheinander
    private function upscale_tensor_triplet($in_fp, $out_fp, $old_r, $old_c, $new_r, $new_c, $scale, $pad_center = 0.0) {
        $this->upscale_matrix_stream($in_fp, $out_fp, $old_r, $old_c, $new_r, $new_c, $scale, false, $pad_center); // Gewichte (w)
        $this->upscale_matrix_stream($in_fp, $out_fp, $old_r, $old_c, $new_r, $new_c, 1.0, true);    // Momentum (m) - Zero padded
        $this->upscale_matrix_stream($in_fp, $out_fp, $old_r, $old_c, $new_r, $new_c, 1.0, true);    // Varianz (v) - Zero padded
    }

    private function upscale_cortex(string $proto_path, string $expanded_path, int $new_dim, int $new_ctx, int $new_layers): bool {
        $in_fp = @fopen($proto_path, 'rb');
        $out_fp = @fopen($expanded_path, 'wb');
        if (!$in_fp || !$out_fp) {
            if ($in_fp) fclose($in_fp);
            if ($out_fp) fclose($out_fp);
            throw new Exception("I/O Fehler beim Öffnen der Upscale-Pfade.");
        }

        // AUDIT-FIX: Hardcoded auf SNAI_V5
        $magic = $this->robust_fread($in_fp, 8);
        $h_data = $this->robust_fread($in_fp, 20);
        $p = unpack('Vvocab/Vdim/Vffn/Vctx/Vlayers', $h_data);
        
        $vocab = $p['vocab'];
        $old_dim = $p['dim'];
        $old_ffn = $p['ffn'];
        $old_ctx = $p['ctx'];
        $old_layers = $p['layers'];

        // AUDIT-FIX: Dynamische Skalierung des FFN-Verhältnisses (Kein hardcodiertes * 4 mehr!)
        $new_ffn = (int)round($new_dim * ($old_ffn / $old_dim));

        echo "[Queen] Zelluläre Expansion bestätigt: Vocab $vocab | $old_dim->$new_dim Dim | $old_ctx->$new_ctx Ctx | $old_layers->$new_layers Layers...\n";

        fwrite($out_fp, $magic);
        fwrite($out_fp, pack('VVVVV', $vocab, $new_dim, $new_ffn, $new_ctx, $new_layers));

        // Globale Tensoren (w, m, v interleaved)
        $this->upscale_tensor_triplet($in_fp, $out_fp, $vocab, $old_dim, $vocab, $new_dim, 1.0); // embeddings
        $this->upscale_tensor_triplet($in_fp, $out_fp, $vocab, $old_dim, $vocab, $new_dim, 1.0); // w_out
        $this->upscale_tensor_triplet($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0, 1.0);      // rms_final  (Gain -> pad 1.0)

        // Offset-Berechnung für V5 (x3 für m und v)
        $layer_0_offset = ftell($in_fp);
        $layer_byte_size = ((2 * $old_dim) + (4 * ($old_dim * $old_dim)) + (2 * ($old_dim * $old_ffn))) * 4 * 3;

        for ($l = 0; $l < $new_layers; $l++) {
            $src_l = $l % $old_layers;
            fseek($in_fp, $layer_0_offset + ($src_l * $layer_byte_size));

            $is_new_layer = ($l >= $old_layers);
            // Tiefenskalierte Daempfung statt fixer Magic Number (Session-Diskussion
            // 2026-08-16, GPT-2/DeepNet-Prinzip: Residual-Stream-Varianz waechst mit der
            // Zahl beitragender Layer, also Anfangs-Beitrag frischer Layer ~1/sqrt(Tiefe)
            // skalieren statt konstant). Anker bei 6 Layern (aktuelles Matriarchin-Ziel)
            // bewusst so gewaehlt, dass der bisher bewaehrte Wert 0.01 dort unveraendert
            // bleibt - nur fuer tiefere kuenftige Tiers wird automatisch vorsichtiger
            // gedaempft, statt bei jedem Wachstumssprung erneut von Hand nachjustiert
            // werden zu muessen.
            $out_scale = $is_new_layer ? (0.01 * sqrt(6.0 / $new_layers)) : 1.0;

            $this->upscale_tensor_triplet($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0, 1.0);     // rms_att   (Gain -> pad 1.0)
            $this->upscale_tensor_triplet($in_fp, $out_fp, 1, $old_dim, 1, $new_dim, 1.0, 1.0);     // rms_ffn   (Gain -> pad 1.0)

            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); // w_q
            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); // w_k
            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, 1.0); // w_v
            
            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_dim, $old_dim, $new_dim, $new_dim, $out_scale); // w_o
            
            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_ffn, $old_dim, $new_ffn, $new_dim, 1.0); // w_up
            $this->upscale_tensor_triplet($in_fp, $out_fp, $old_dim, $old_ffn, $new_dim, $new_ffn, $out_scale); // w_down
        }
        
        fclose($in_fp);
        
        fflush($out_fp); 
        if (function_exists('fsync')) {
            @fsync($out_fp); 
        }
        fclose($out_fp);
        
        clearstatcache();
        
        if (!$this->validate_tensor_file($expanded_path)) {
            throw new Exception("upscale_cortex hat eine korrupte/unvollständige V5-Datei generiert.");
        }

        // AUDIT: Net2Net-Expansionsgrenze persistieren, damit der Adam-
        // Optimizer die frisch verrauschten Bereiche (siehe growth_config.php)
        // ueber alle Epochen dieser Synthese hinweg mit einer hoeheren
        // effektiven Lernrate versorgen kann.
        write_growth_sidecar($expanded_path, $old_dim, $old_ffn, $old_layers);

        echo "[Queen] Modell erfolgreich rekonstruiert.\n";
        return true;
    }
}