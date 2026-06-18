<?php
/** Dateiname: snai_gossip_node.php
  *Funktion: Dezentraler P2P-Daemon. Empfängt UDP-Pheromone und synchronisiert Tensoren (Base Engine Edition).
  *
  * SYMBIO NANO-AI FRAMEWORK
  * Modul: Base Model Initializer
  * This file is part of SnAI.
  *
  * SnAI is free software: you can redistribute it and/or modify
  * it under the terms of the GNU Affero General Public License as published
  * by the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
*/
// Polyfill für PHP < 8.0
if (!function_exists('str_ends_with')) {
    function str_ends_with(string $haystack, string $needle): bool {
        $needle_len = strlen($needle);
        return ($needle_len === 0 || 0 === substr_compare($haystack, $needle, - $needle_len));
    }
}
chdir(__DIR__ . '/..');
require_once __DIR__ . '/tensor_io.php';

define('GOSSIP_PORT', 43720);
define('TENSOR_PORT', 43721);
define('GOSSIP_PHEROMONE_TTL', 86400); 
define('CENSUS_EVAPORATION_TIME', 300);

class SnaiGossipNode {
    private $local_ip;
    private $uuid;
    
    // Sockets
    private $udp_sock;
    private $tcp_sock;
    
    // Async I/O Queues
    private $active_downloads = [];
    private $active_uploads = [];
    
    // Taktgeber
    private $last_broadcast = 0;
    private $last_evaporation = 0;
    
    public function __construct() {
        $this->local_ip = $this->get_local_lan_ip();
        $this->init_uuid();
        $this->init_sockets();
        
        echo "\e[32m====================================================\e[0m\n";
        echo "\e[32m=== [P2P DAEMON] SYMBIO BASE ENGINE INIT         ===\e[0m\n";
        echo "\e[32m====================================================\e[0m\n";
        echo "\e[36m[System]\e[0m IP: {$this->local_ip} | UUID: {$this->uuid}\n";
        echo "\e[36m[Daemon]\e[0m Sensorik auf UDP " . GOSSIP_PORT . " | Arterie auf TCP " . TENSOR_PORT . "\n\n";
    }

    private function init_uuid() {
        $uuid_file = './factory/node_uuid.txt';
        if (!file_exists($uuid_file)) {
            @mkdir('./factory', 0777, true);
            file_put_contents($uuid_file, uniqid('snai_'));
        }
        $this->uuid = trim(file_get_contents($uuid_file));
    }

    private function get_local_lan_ip() {
        $sock = @socket_create(AF_INET, SOCK_DGRAM, SOL_UDP);
        if ($sock) {
            @socket_connect($sock, '8.8.8.8', 53);
            @socket_getsockname($sock, $name);
            @socket_close($sock);
            if ($name && $name !== '0.0.0.0') return $name;
        }

        $ip = gethostbyname(gethostname());
        if (strpos($ip, '127.') === 0 && function_exists('shell_exec')) {
            $out = @shell_exec("hostname -I | awk '{print $1}'");
            if ($out) $ip = trim($out);
        }
        return $ip;
    }

    private function init_sockets() {
        $opts = ['socket' => ['so_reuseaddr' => true]];
        $ctx = stream_context_create($opts);
        
        $this->udp_sock = stream_socket_server("udp://0.0.0.0:" . GOSSIP_PORT, $errno, $errstr, STREAM_SERVER_BIND, $ctx);
        stream_set_blocking($this->udp_sock, false);
        
        $this->tcp_sock = stream_socket_server("tcp://0.0.0.0:" . TENSOR_PORT, $errno, $errstr, STREAM_SERVER_BIND | STREAM_SERVER_LISTEN, $ctx);
        stream_set_blocking($this->tcp_sock, false);
    }

    // =========================================================================
    // DIE EVENT-LOOP
    // =========================================================================

    public function pulse() {
        while (true) {
            $read = [$this->udp_sock, $this->tcp_sock];
            $write = [];
            $except = null;
            
            foreach ($this->active_downloads as $id => $job) $read[] = $job['sock'];
            foreach ($this->active_uploads as $id => $job) $write[] = $job['sock'];

            if (stream_select($read, $write, $except, 0, 50000) > 0) {
                
                // 1. Neues Pheromon gehört?
                if (in_array($this->udp_sock, $read)) {
                    while ($payload = @stream_socket_recvfrom($this->udp_sock, 2048, 0, $peer)) {
                        $this->handle_udp_payload($payload);
                    }
                }
                
                // 2. Jemand will ein Artefakt laden?
                if (in_array($this->tcp_sock, $read)) {
                    $this->accept_tcp_client();
                }
                
                // 3. Eigene Downloads weiter saugen
                $this->process_downloads($read);
            }
            
            // 4. Eigene Uploads weiter pumpen
            $this->process_uploads($write);
            
            // 5. Taktgesteuerte Events
            $now = time();
            if ($now - $this->last_broadcast >= 10) {
                $this->broadcast_local_state();
                $this->print_telemetry_heartbeat(); 
                $this->last_broadcast = $now;
            }
            
            if ($now - $this->last_evaporation >= 60) {
                $this->evaporate_census();
                $this->last_evaporation = $now;
            }
        }
    }

    // TERMINAL TELEMETRIE
    
    private function print_telemetry_heartbeat() {
        $dl_count = count($this->active_downloads);
        $ul_count = count($this->active_uploads);
        
        $census_file = './factory/swarm_census.json';
        $node_count = 0;
        if (file_exists($census_file)) {
            $census = json_decode(file_get_contents($census_file), true);
            $node_count = isset($census['nodes']) ? count($census['nodes']) : 0;
        }

        $dl_details = "";
        foreach ($this->active_downloads as $job) {
            $pct = $job['expected'] > 0 ? round(($job['downloaded'] / $job['expected']) * 100) : 0;
            $dl_details .= " [↓ {$job['filename']} {$pct}%]";
        }

        $ul_details = "";
        foreach ($this->active_uploads as $job) {
            $ul_details .= " [↑ {$job['filename']}]";
        }

        $time = date('H:i:s');
        echo "\r\e[K\e[90m[{$time}] [PULSE] Nodes: {$node_count} | DL: {$dl_count} | UL: {$ul_count}\e[0m{$dl_details}{$ul_details}";
    }

    // SENSORIK & GEDÄCHTNIS

    private function handle_udp_payload($payload) {
        if (!$payload) return;
        
        $data = json_decode($payload, true);
        if (!$data || !isset($data['origin'])) return;
        
        $sender_ip = $data['origin'];
        if ($sender_ip === $this->local_ip) return; 
        if (time() - $data['timestamp'] > GOSSIP_PHEROMONE_TTL) return; 
        
        $this->update_census($sender_ip);
        
        foreach (($data['entities'] ?? []) as $type => $filename) {
            $local_path = $this->get_path_for_type($type, $filename);
            
            if ($local_path && !file_exists($local_path)) {
                $is_downloading = false;
                foreach ($this->active_downloads as $job) {
                    if ($job['final_path'] === $local_path) $is_downloading = true;
                }
                
                if (!$is_downloading) {
                    $this->start_async_download($sender_ip, $filename, $local_path);
                }
            }
        }
    }

    private function update_census($ip) {
        $file = './factory/swarm_census.json';
        $census = file_exists($file) ? json_decode(file_get_contents($file), true) : ['nodes' => []];
        if (!is_array($census)) $census = ['nodes' => []];

        $census['nodes'][$ip] = [
            'last_seen' => time()
        ];
        
        @file_put_contents($file, json_encode($census), LOCK_EX);
    }

    private function evaporate_census() {
        $file = './factory/swarm_census.json';
        if (!file_exists($file)) return;
        
        $census = json_decode(file_get_contents($file), true);
        if (!is_array($census) || !isset($census['nodes'])) return;

        $changed = false;
        $now = time();
        
        foreach ($census['nodes'] as $ip => $data) {
            if ($now - $data['last_seen'] > CENSUS_EVAPORATION_TIME) {
                unset($census['nodes'][$ip]);
                $this->log_event('EVAPORATE', $ip, "Knoten aus Topologie verdunstet.");
                $changed = true;
            }
        }
        
        if ($changed) {
            @file_put_contents($file, json_encode($census), LOCK_EX);
        }
    }

    private function broadcast_local_state() {
        $sock = @socket_create(AF_INET, SOCK_DGRAM, SOL_UDP);
        if (!$sock) return;

        socket_set_option($sock, SOL_SOCKET, SO_BROADCAST, 1);
        
        $entities = [];
        
        $apex = glob('./factory/apex/*.apex');
        if (!empty($apex)) $entities['APEX'] = basename($apex[0]);
        
        $cortex = glob('./factory/royal/*.cortex');
        if (!empty($cortex)) $entities['CORTEX'] = basename($cortex[0]);
        
        $drones = glob('./factory/experts/*.snai');
        if (!empty($drones)) {
            usort($drones, function($a, $b) {
                preg_match('/_L([0-9\.]+)\.snai$/', $a, $mA);
                preg_match('/_L([0-9\.]+)\.snai$/', $b, $mB);
                $lA = isset($mA[1]) ? (float)$mA[1] : 9.9;
                $lB = isset($mB[1]) ? (float)$mB[1] : 9.9;
                return $lA <=> $lB;
            });
            $entities['DRONE'] = basename($drones[0]);
        }
        
        $sft_data = glob('./factory/input/*.txt');
        if (!empty($sft_data)) {
            shuffle($sft_data); 
            $entities['NEKTAR'] = basename($sft_data[0]); 
        }

        $payload = json_encode([
            'origin' => $this->local_ip,
            'timestamp' => time(),
            'entities' => $entities
        ]);

        $parts = explode('.', $this->local_ip);
        if (count($parts) === 4) {
            $bcast = $parts[0] . '.' . $parts[1] . '.' . $parts[2] . '.255';
            if ($bcast !== '127.0.0.255') {
                @socket_sendto($sock, $payload, strlen($payload), 0, $bcast, GOSSIP_PORT);
            }
        }
        socket_close($sock);
    }


    // I/O: ARTERIE & MAGEN (Non-Blocking Transfers)


    private function get_path_for_type($type, $filename) {
        $safe_name = basename($filename);
        if ($type === 'APEX') return './factory/apex/' . $safe_name;
        if ($type === 'CORTEX') return './factory/royal/' . $safe_name;
        if ($type === 'DRONE') return './factory/experts/' . $safe_name;
        if ($type === 'NEKTAR') return './factory/input/' . $safe_name;
        return null;
    }

    private function accept_tcp_client() {
        $client = @stream_socket_accept($this->tcp_sock, 0);
        if (!$client) return;
        
        stream_set_blocking($client, true);
        stream_set_timeout($client, 0, 500000); 
        
        $req = basename(trim(fgets($client, 256))); 
        
        stream_set_blocking($client, false);
        
        $file_path = null;
        if (str_ends_with($req, '.apex')) $file_path = './factory/apex/' . $req;
        elseif (str_ends_with($req, '.cortex')) $file_path = './factory/royal/' . $req;
        elseif (str_ends_with($req, '.snai')) $file_path = './factory/experts/' . $req;
        elseif (str_ends_with($req, '.txt')) $file_path = './factory/input/' . $req;

        if ($file_path && file_exists($file_path)) {
            $size = filesize($file_path);
            @fwrite($client, "OK:$size\n");
            
            $job_id = uniqid();
            $this->active_uploads[$job_id] = [
                'sock' => $client,
                'fp' => fopen($file_path, 'rb'),
                'filename' => $req
            ];
            $this->log_event('UPLOAD_START', stream_socket_get_name($client, true), "Sende $req...");
        } else {
            @fwrite($client, "ERROR:Ghost-Pheromon\n");
            fclose($client);
        }
    }

    private function process_uploads($write_ready_sockets) {
        foreach ($this->active_uploads as $id => &$job) {
            if (in_array($job['sock'], $write_ready_sockets)) {
                $chunk = fread($job['fp'], 8192); 
                
                if ($chunk === false || strlen($chunk) === 0) {
                    fclose($job['fp']);
                    fclose($job['sock']);
                    unset($this->active_uploads[$id]);
                    $this->log_event('UPLOAD_DONE', 'local', "Artefakt versendet: {$job['filename']}");
                } else {
                    $written = @fwrite($job['sock'], $chunk);
                    
                    if ($written === false || $written === 0) {
                        fseek($job['fp'], -strlen($chunk), SEEK_CUR);
                    } elseif ($written < strlen($chunk)) {
                        fseek($job['fp'], - (strlen($chunk) - $written), SEEK_CUR);
                    }
                }
            }
        }
    }

    private function start_async_download($ip, $filename, $final_path) {
        $sock = @stream_socket_client("tcp://$ip:" . TENSOR_PORT, $errno, $errstr, 0.2);
        if (!$sock) {
            return;
        }
        
        stream_set_timeout($sock, 1);
        @fwrite($sock, basename($filename) . "\n");
        $header = trim(fgets($sock, 128));
        
        if (strpos($header, "OK:") === 0) {
            $expected_size = (int)str_replace("OK:", "", $header);
            
            stream_set_blocking($sock, false); 
            
            $tmp_path = $final_path . ".tmp"; 
            
            $job_id = uniqid();
            $this->active_downloads[$job_id] = [
                'sock' => $sock,
                'fp' => fopen($tmp_path, 'wb'),
                'final_path' => $final_path,
                'tmp_path' => $tmp_path,
                'downloaded' => 0,
                'expected' => $expected_size,
                'filename' => $filename,
                'ip' => $ip
            ];
            $this->log_event('DOWNLOAD_START', $ip, "Sauge $filename ($expected_size Bytes)...");
        } else {
            fclose($sock);
        }
    }

    private function process_downloads($read_ready_sockets) {
        foreach ($this->active_downloads as $id => &$job) {
            if (in_array($job['sock'], $read_ready_sockets)) {
                $chunk = fread($job['sock'], 16384);
                
                if ($chunk !== false && strlen($chunk) > 0) {
                    fwrite($job['fp'], $chunk);
                    $job['downloaded'] += strlen($chunk);
                }
                
                // INSTANT CLOSE
                if ($job['downloaded'] >= $job['expected']) {
                    fclose($job['fp']);
                    fclose($job['sock']);
                    
                    if ($job['downloaded'] === $job['expected']) { 
                        rename($job['tmp_path'], $job['final_path']);
                        $this->log_event('SUCCESS', $job['ip'], "Artefakt assimiliert (Instant-Close): {$job['filename']}");
                    } else {
                        @unlink($job['tmp_path']);
                        $this->log_event('ERROR', $job['ip'], "Artefakt verseucht (Oversized): {$job['filename']}");
                    }
                    unset($this->active_downloads[$id]);
                    continue; 
                }
                
                if ($chunk === false || feof($job['sock'])) {
                    fclose($job['fp']);
                    fclose($job['sock']);
                    @unlink($job['tmp_path']);
                    $this->log_event('ERROR', $job['ip'], "Verbindung tot (Fehlende Bytes): {$job['filename']} ({$job['downloaded']}/{$job['expected']})");
                    unset($this->active_downloads[$id]);
                }
            }
        }
    }

    // =========================================================================
    // ANSI TERMINAL LOGGING (JSON entfernt für Base Engine)
    // =========================================================================

    private function log_event($type, $node_ip, $message) {
        $time = date('H:i:s');
        
        echo "\n"; 
        
        $color = "\e[0m"; // Default (Reset)
        if ($type === 'SUCCESS' || $type === 'UPLOAD_DONE') $color = "\e[32m"; // Grün
        elseif ($type === 'ERROR' || $type === 'EVAPORATE') $color = "\e[31m"; // Rot
        elseif ($type === 'DOWNLOAD_START' || $type === 'UPLOAD_START') $color = "\e[36m"; // Cyan
        
        echo "{$color}[{$time}] [{$type}] Node: {$node_ip} | {$message}\e[0m\n";
    }
}

// CLI Bootstrap
if (php_sapi_name() === 'cli' && basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    $daemon = new SnaiGossipNode();
    $daemon->pulse();
}
