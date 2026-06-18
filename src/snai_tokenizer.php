<?php
/** Dateiname: snai_tokenizer.php
  * Funktion: BPE Tokenizer Engine. Wandelt Text in Token-IDs um und umgekehrt. Nutzt Metaspaces für exakte Leerzeichen-Rekonstruktion.
  * 
  * Modul: Base Model Initializer
  * This file is part of SnAI.
  *
  * SnAI is free software: you can redistribute it and/or modify
  * it under the terms of the GNU Affero General Public License as published
  * by the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
*/

class SnaiTokenizer {
    private $vocab = [];
    private $inverse_vocab = [];
    private $bpe_ranks = [];
    private $cache = []; 
    
    // Hex-Repräsentation von U+2581 für absolute Sicherheit in PHP
    private const META_SPACE = "\xE2\x96\x81"; 

    public function __construct(string $vocab_file = 'lib/symbio_vocab.json') {
        if (!file_exists($vocab_file)) {
            die("FATAL: Tokenizer-Datei '$vocab_file' fehlt. Führe erst 'tools/bpe_init/bpe_gen.py' aus oder lade das Release-Paket herunter.\n");
        }

        $data = json_decode(file_get_contents($vocab_file), true);
        
        $this->vocab = $data['model']['vocab'];
        $this->inverse_vocab = array_flip($this->vocab);
        
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
        if ($word === false) return $token; // Fallback bei PCRE-Limits

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
        
        // Eliminiert zerschnittene UTF-8 Bytes (z.B. halbe Emojis), die beim 
        // byte-basierten Chunking der Textdateien entstehen und die PCRE-Engine crashen lassen.
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
            
            // @ unterdrückt PCRE-Warnings.
            $parts = @preg_split($regex, $text, -1, PREG_SPLIT_DELIM_CAPTURE | PREG_SPLIT_NO_EMPTY);
            
            // Fallback, falls die Regex-Engine trotzdem versagt (z.B. Memory/Backtrack Limits)
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

            $matched = @preg_match_all('/(' . self::META_SPACE . '|\n)[^' . self::META_SPACE . '\n]*/u', $work_text, $matches);
            
            // Abfangen von Engine-Crashes. Verhindert "false given in foreach"
            if ($matched === false || empty($matches[0])) {
                continue;
            }
            
            foreach ($matches[0] as $chunk) {
                if ($chunk === '') continue;
                $bpe_str = $this->bpe($chunk);
                $bpe_parts = explode(' ', $bpe_str);
                foreach ($bpe_parts as $bpe_part) {
                    $bpe_tokens[] = $this->vocab[$bpe_part] ?? ($this->vocab['<unk>'] ?? 0);
                }
            }
        }
        
        return $bpe_tokens;
    }

    public function decode(array $token_ids, bool $do_trim = false): string {
        $text = "";
        
        foreach ($token_ids as $id) {
            if (isset($this->inverse_vocab[$id])) {
                $text .= $this->inverse_vocab[$id];
            }
        }
        
        $text = str_replace(self::META_SPACE, ' ', $text);
        
        return $do_trim ? trim($text) : $text;
    }
}

// === QUICK TEST ===
if (basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    if (file_exists('symbio_vocab.json')) {
        $tok = new SnaiTokenizer('symbio_vocab.json');
        
        echo "\n[System] Tokenizer geladen. Dynamische Vocab-Size: " . $tok->getVocabSize() . "\n";
        
        $test_string = "[USER] Definiere eine Python Funktion:\n[SYMBIO] def hallo():\n    print('Test')</s>";
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
