<?php
// Dateiname: src/snai_tokenizer.php
// Funktion: BPE Tokenizer Engine. Wandelt Text in Token-IDs um und umgekehrt. Nutzt Metaspaces für exakte Leerzeichen-Rekonstruktion.

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: BPE Tokenizer (Phase V.2 - Data Integrity Patch)
 * -------------------------------------------------------------------------
 * MAXIME: Kompromisslose Präzision & Dynamische Skalierbarkeit. 
 * FIX: Trim-Bug in der decode() Funktion behoben. 
 * FIX: SFT-Korsett Isolation. Special-Tokens werden vor dem BPE-Split extrahiert.
 * AUDIT-FIX (V9.6): UTF-8 Aegis Patch gegen Zerschnittene Bytes.
 * AUDIT-FIX (V10): Stiller Text-Verlust! Regex um String-Anfang (`^`) erweitert, 
 * damit Wörter nach Special-Tokens (wie \n) nicht mehr aus dem Vokabular gelöscht werden.
 */

class SnaiTokenizer {
    private $vocab = [];
    private $inverse_vocab = [];
    private $bpe_ranks = [];
    private $cache = []; 
    private $byte_ids = null;
    private $byte_of_id = [];
    
    // Wir nutzen die Hex-Repräsentation von U+2581 für absolute Sicherheit in PHP
    private const META_SPACE = "\xE2\x96\x81"; 

    public function __construct(string $vocab_file = __DIR__ . '/../lib/symbio_vocab.json') {
        if (!file_exists($vocab_file)) {
            die("🚨 FATAL: Tokenizer-Datei '$vocab_file' fehlt. Erzeuge sie mit tools/bpe_init/bpe_forge.\n");
        }

        $data = json_decode(file_get_contents($vocab_file), true);
        
        // AUDIT-FIX: Struktur-Validierung
        if (!isset($data['model']['vocab']) || !is_array($data['model']['vocab'])) {
            die("🚨 FATAL: '$vocab_file' ist korrupt. 'model.vocab' fehlt oder ist ungültig.\n");
        }
        
        $this->vocab = $data['model']['vocab'];
        $this->inverse_vocab = array_flip($this->vocab);
        
        // Byte-Rueckfall nur, wenn das Vokabular alle 256 Byte-Token traegt.
        if (isset($this->vocab['<0x00>'])) {
            for ($b = 0; $b < 256; $b++) {
                $key = sprintf('<0x%02X>', $b);
                if (!isset($this->vocab[$key])) {
                    die("🚨 FATAL: '$vocab_file' hat <0x00>, aber $key fehlt - Byte-Rueckfall unvollstaendig.\n");
                }
                $this->byte_ids[$b] = $this->vocab[$key];
                $this->byte_of_id[$this->vocab[$key]] = chr($b);
            }
        }

        if (isset($data['model']['merges'])) {
            foreach ($data['model']['merges'] as $i => $merge) {
                $merge_str = is_array($merge) ? implode(' ', $merge) : (string)$merge;
                $this->bpe_ranks[$merge_str] = $i;
            }
        }
    }

    public function getVocabSize(): int {
        return count($this->vocab);
    }

    private function get_pairs(array $word): array {
        $pairs = [];
        $prev_char = $word[0];
        for ($i = 1; $i < count($word); $i++) {
            $char = $word[$i];
            $pairs[] = [$prev_char, $char];
            $prev_char = $char;
        }
        return $pairs;
    }

    private function bpe(string $token): string {
        if (isset($this->cache[$token])) return $this->cache[$token];

        $word = preg_split('//u', $token, -1, PREG_SPLIT_NO_EMPTY);
        if ($word === false) return $token; 

        $pairs = $this->get_pairs($word);

        if (empty($pairs)) return $token;

        while (true) {
            $min_rank = PHP_INT_MAX;
            $bigram = null;

            foreach ($pairs as $pair) {
                $pair_str = $pair[0] . ' ' . $pair[1];
                if (isset($this->bpe_ranks[$pair_str])) {
                    $rank = $this->bpe_ranks[$pair_str];
                    if ($rank < $min_rank) {
                        $min_rank = $rank;
                        $bigram = $pair;
                    }
                }
            }

            if ($bigram === null) break;

            $first = $bigram[0];
            $second = $bigram[1];
            $new_word = [];
            $i = 0;
            
            while ($i < count($word)) {
                $j = array_search($first, array_slice($word, $i), true);
                if ($j !== false) {
                    $j += $i;
                    for ($k = $i; $k < $j; $k++) $new_word[] = $word[$k];
                    $i = $j;
                } else {
                    for ($k = $i; $k < count($word); $k++) $new_word[] = $word[$k];
                    break;
                }

                if ($i < count($word) - 1 && $word[$i] === $first && $word[$i + 1] === $second) {
                    $new_word[] = $first . $second;
                    $i += 2;
                } else {
                    $new_word[] = $word[$i];
                    $i += 1;
                }
            }
            
            $word = $new_word;
            if (count($word) === 1) break;
            else $pairs = $this->get_pairs($word);
        }

        $result = implode(' ', $word);
        $this->cache[$token] = $result;
        return $result;
    }

    public function encode(string $text): array {
        $bpe_tokens = [];
        
        $text = mb_convert_encoding($text, 'UTF-8', 'UTF-8');
        
        if (class_exists('Normalizer')) {
            $text = Normalizer::normalize($text, Normalizer::FORM_KC);
        }

        if ($text !== '' && $text[0] !== ' ' && $text[0] !== '[' && $text[0] !== '<') {
            $text = ' ' . $text;
        }

        $special_tokens = ['<unk>', '<pad>', '<s>', '</s>', "\n", '[USER]', '[SYMBIO]', '[SYSTEM]'];
        $special_map = [];
        
        foreach ($special_tokens as $st) {
            if (isset($this->vocab[$st])) {
                $special_map[$st] = $this->vocab[$st];
            }
        }

        if (!empty($special_map)) {
            $escaped_specials = array_map(function($st) { return preg_quote($st, '/'); }, array_keys($special_map));
            $regex = '/(' . implode('|', $escaped_specials) . ')/u';
            
            $parts = @preg_split($regex, $text, -1, PREG_SPLIT_DELIM_CAPTURE | PREG_SPLIT_NO_EMPTY);
            
            if ($parts === false) {
                $parts = [$text];
            }
        } else {
            $parts = [$text];
        }

        foreach ($parts as $part) {
            if (isset($special_map[$part])) {
                $bpe_tokens[] = $special_map[$part];
                continue;
            }

            if ($part === '') continue;

            $work_text = str_replace(' ', self::META_SPACE, $part);

            // AUDIT-FIX: Erweitertes Pattern, das auch String-Anfänge ohne Meta-Space matchen kann!
            // Verhindert das spurlos Verschwinden von Wörtern direkt nach einem Newline/Special-Token.
            $pattern = '/(?:' . self::META_SPACE . '|\n)[^' . self::META_SPACE . '\n]*|^[^' . self::META_SPACE . '\n]*/u';
            $matched = @preg_match_all($pattern, $work_text, $matches);
            
            if ($matched === false || empty($matches[0])) {
                continue;
            }
            
            foreach ($matches[0] as $chunk) {
                if ($chunk === '') continue;
                $bpe_str = $this->bpe($chunk);
                $bpe_parts = explode(' ', $bpe_str);
                foreach ($bpe_parts as $bpe_part) {
                    if (isset($this->vocab[$bpe_part])) {
                        $bpe_tokens[] = $this->vocab[$bpe_part];
                    } elseif ($this->byte_ids !== null) {
                        foreach (unpack('C*', $bpe_part) as $b) $bpe_tokens[] = $this->byte_ids[$b];
                    } else {
                        $bpe_tokens[] = $this->vocab['<unk>'] ?? 0;
                    }
                }
            }
        }
        
        return $bpe_tokens;
    }

    public function decode(array $token_ids, bool $do_trim = false): string {
        $text = "";
        
        // Byte-Token werden als rohes Byte ausgegeben, auch einzeln: die
        // Inferenz dekodiert Token fuer Token und setzt den Strom so wieder
        // zu gueltigem UTF-8 zusammen.
        foreach ($token_ids as $id) {
            if (isset($this->byte_of_id[$id])) {
                $text .= $this->byte_of_id[$id];
            } elseif (isset($this->inverse_vocab[$id])) {
                $text .= $this->inverse_vocab[$id];
            }
        }
        
        $text = str_replace(self::META_SPACE, ' ', $text);
        
        return $do_trim ? trim($text) : $text;
    }
}

// === QUICK TEST ===
if (basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    if (file_exists(__DIR__ . '/../lib/symbio_vocab.json')) {
        $tok = new SnaiTokenizer();
        
        echo "\n[System] Tokenizer geladen. Dynamische Vocab-Size: " . $tok->getVocabSize() . "\n";
        
        // Erweiterter Test-String ohne führendes Leerzeichen am Zeilenanfang
        $test_string = "[USER] Definiere eine Python Funktion:\n[SYMBIO] def hallo():\nprint('Test')</s>";
        echo "\n[Test String]:\n$test_string\n\n";
        
        $start = microtime(true);
        $ids = $tok->encode($test_string);
        $time = round((microtime(true) - $start) * 1000, 2);
        
        echo "[Encoded IDs]: " . implode(", ", $ids) . " ({$time}ms)\n";
        
        $decoded = $tok->decode($ids, false); 
        echo "\n[Decoded String]:\n$decoded\n";
        
        if ($test_string === $decoded) {
            echo "\n✅ ROUNDTRIP TEST BESTANDEN! Exakte Rekonstruktion.\n";
        } else {
            echo "\n❌ ROUNDTRIP TEST FEHLGESCHLAGEN!\n";
        }
    }
}