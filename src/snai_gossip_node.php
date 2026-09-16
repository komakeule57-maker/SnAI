<?php
// DISTRO: Wurzel setzen, bevor irgendein './factory/...'-Pfad benutzt wird.
chdir(dirname(__DIR__));
require_once __DIR__ . '/korpus_sanitize.php';
// Polyfill für PHP < 8.0
if (!function_exists('str_ends_with')) {
    function str_ends_with(string $haystack, string $needle): bool {
        $needle_len = strlen($needle);
        return ($needle_len === 0 || 0 === substr_compare($haystack, $needle, - $needle_len));
    }
}
// Dateiname: src/snai_gossip_node.php
// Funktion: Dezentraler P2P-Daemon (Myzelium). Empfängt UDP-Pheromone, synchronisiert Tensoren und bedient den Hive-Fingerprint.

/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Dezentraler Gossip Node (The Mycelium) - PHASE VI.3 (SECURITY PATCH)
 * -------------------------------------------------------------------------
 * MAXIME: E > H. Non-Blocking I/O, Objektorientiert, Kasten-Bewusstsein.
 * FIX: Der "Last Chunk Drop" Bug ist behoben.
 * FIX: Half-Open TCP Prävention (Instant-Close) ist aktiv.
 * FEATURE (VI.2): Real-Time Terminal Telemetrie. Farbcodierte Logs und ein 
 * 10-Sekunden Heartbeat zeigen dir exakt, wer was saugt oder sendet.
 * AUDIT-FIX (VI.3): Hard-Crash bei belegten Sockets, Payload-Timestamp-Guard
 * und absolute Pfade für die State-Erkennung integriert.
 */

require_once 'tensor_io.php';
require_once 'snai_caste_profiler.php'; // Für die Hardware-Kaste

define('GOSSIP_PORT', 43720);
define('TENSOR_PORT', 43721);
define('IDENTIFY_PORT', 4372); 
define('GOSSIP_PHEROMONE_TTL', 86400); 
define('CENSUS_EVAPORATION_TIME', 300);

class SnaiGossipNode {
    private $local_ip;
    private $profile;
    private $uuid;
    
    // Sockets
    private $udp_sock;
    private $tcp_sock;
    private $identify_sock; 
    
    // Async I/O Queues
    private $active_downloads = [];
    private $active_uploads = [];
    
    // Taktgeber
    private $last_broadcast = 0;
    private $last_evaporation = 0;
    
    public function __construct() {
        $this->local_ip = $this->get_local_lan_ip();
        $this->profile = SnaiHardwareProfiler::get_profile();
        $this->init_uuid();
        
        $this->init_sockets();
        
        echo "\e[32m====================================================\e[0m\n";
        echo "\e[32m=== [GOSSIP NODE] MYZELIUM INITIALISIERT         ===\e[0m\n";
        echo "\e[32m====================================================\e[0m\n";
        echo "\e[36m[System]\e[0m IP: {$this->local_ip} | Kaste: {$this->profile['caste']} | UUID: {$this->uuid}\n";
        echo "\e[36m[Daemon]\e[0m Ohr auf UDP " . GOSSIP_PORT . " | Arterie auf TCP " . TENSOR_PORT . "\n";
        echo "\e[36m[Daemon]\e[0m Hive-Identifikation aktiv auf TCP " . IDENTIFY_PORT . "\n\n";
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
        
        // AUDIT-FIX: Hard-Crash bei belegten Sockets statt Endlosschleife
        $this->udp_sock = stream_socket_server("udp://0.0.0.0:" . GOSSIP_PORT, $errno, $errstr, STREAM_SERVER_BIND, $ctx);
        if (!$this->udp_sock) die("🚨 [Fatal] UDP Port " . GOSSIP_PORT . " ist belegt oder blockiert: $errstr\n");
        stream_set_blocking($this->udp_sock, false);
        
        $this->tcp_sock = stream_socket_server("tcp://0.0.0.0:" . TENSOR_PORT, $errno, $errstr, STREAM_SERVER_BIND | STREAM_SERVER_LISTEN, $ctx);
        if (!$this->tcp_sock) die("🚨 [Fatal] TCP Port " . TENSOR_PORT . " ist belegt oder blockiert: $errstr\n");
        stream_set_blocking($this->tcp_sock, false);

        $this->identify_sock = stream_socket_server("tcp://0.0.0.0:" . IDENTIFY_PORT, $errno, $errstr, STREAM_SERVER_BIND | STREAM_SERVER_LISTEN, $ctx);
        if (!$this->identify_sock) die("🚨 [Fatal] TCP Port " . IDENTIFY_PORT . " ist belegt oder blockiert: $errstr\n");
        stream_set_blocking($this->identify_sock, false);
    }

    // =========================================================================
    // DIE EVENT-LOOP (Das schlagende Herz)
    // =========================================================================

    public function pulse() {
        while (true) {
            $read = [$this->udp_sock, $this->tcp_sock, $this->identify_sock];
            $write = [];
            $except = null;
            
            foreach ($this->active_downloads as $id => $job) $read[] = $job['sock'];
            foreach ($this->active_uploads as $id => $job) $write[] = $job['sock'];

            if (stream_select($read, $write, $except, 0, 50000) > 0) {
                
                if (in_array($this->udp_sock, $read)) {
                    while ($payload = @stream_socket_recvfrom($this->udp_sock, 2048, 0, $peer)) {
                        $this->handle_udp_payload($payload);
                    }
                }
                
                if (in_array($this->tcp_sock, $read)) {
                    $this->accept_tcp_client();
                }

                if (in_array($this->identify_sock, $read)) {
                    $this->accept_identify_client();
                }
                
                $this->process_downloads($read);
            }
            
            $this->process_uploads($write);
            
            // Taktgesteuerte Events
            $now = time();
            if ($now - $this->last_broadcast >= 10) {
                $this->broadcast_local_state();
                $this->print_telemetry_heartbeat(); // NEU: Terminal Heartbeat
                $this->last_broadcast = $now;
            }
            
            if ($now - $this->last_evaporation >= 60) {
                $this->evaporate_census();
                $this->last_evaporation = $now;
            }
        }
    }

    // =========================================================================
    // TERMINAL TELEMETRIE (NEU)
    // =========================================================================
    
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

        // Überschreibt die Zeile sanft, wenn nichts los ist (verhindert Terminal-Spam)
        $time = date('H:i:s');
        echo "\r\e[K\e[90m[{$time}] [PULSE] Nodes: {$node_count} | DL: {$dl_count} | UL: {$ul_count}\e[0m{$dl_details}{$ul_details}";
    }

    // =========================================================================
    // HIVE-FINGERPRINTING (Anti-Spam-Loop)
    // =========================================================================

    private function accept_identify_client() {
        $client = @stream_socket_accept($this->identify_sock, 0);
        if (!$client) return;

        $atp = 100.0;
        // AUDIT-FIX: Absoluter Pfad für Klarheit
        $state_file = __DIR__ . '/beehive_state.json';
        if (file_exists($state_file)) {
            $state = json_decode(file_get_contents($state_file), true);
            $atp = $state['atp'] ?? 100.0;
        }

        $response = json_encode([
            'sys' => 'SNAI',
            'uuid' => $this->uuid,
            'kaste' => $this->profile['caste'],
            'atp' => $atp
        ]) . "\n";

        @fwrite($client, $response);
        @fclose($client); 
    }

    // =========================================================================
    // SENSORIK & GEDÄCHTNIS
    // =========================================================================

    private function handle_udp_payload($payload) {
        if (!$payload) return;
        
        $data = json_decode($payload, true);
        if (!$data || !isset($data['origin'])) return;
        
        $sender_ip = $data['origin'];
        if ($sender_ip === $this->local_ip) return; 
        // AUDIT-FIX: ?? 0 schützt vor "Undefined array key" Spam
        if (time() - ($data['timestamp'] ?? 0) > GOSSIP_PHEROMONE_TTL) return; 
        
        $caste = $data['caste'] ?? 'UNKNOWN';
        $this->update_census($sender_ip, $caste);
        
        foreach (($data['entities'] ?? []) as $type => $filename) {
            $local_path = $this->get_path_for_type($type, $filename);

            if ($local_path && !file_exists($local_path)) {
                // AUDIT-FIX: NEKTAR-Dateien wandern nach erfolgreichem Digest nach
                // factory/archive/ (bzw. factory/trash/ bei Ablehnung) und verschwinden
                // damit aus factory/input/. Ein reiner file_exists()-Check auf input/
                // sah eine bereits verarbeitete Datei danach wieder als "fehlt noch"
                // an -> endloses Re-Download+Re-Digest identischer Inhalte zwischen
                // zwei Gossip-Peers (live beobachtet: hf_GEN_6d781588.txt und
                // hf_SFT_6a8cff0b.txt liefen mehrfach im Kreis).
                // Gilt fuer beide Nektar-Sorten: auch Lang-Material wandert nach
                // dem Digest nach factory/archive/ und saehe sonst wieder wie
                // "fehlt noch" aus.
                $is_nektar = ($type === 'NEKTAR' || $type === 'NEKTAR_LONG');
                if ($is_nektar && $this->is_nektar_already_processed($filename)) {
                    continue;
                }

                $is_downloading = false;
                foreach ($this->active_downloads as $job) {
                    if ($job['final_path'] === $local_path) $is_downloading = true;
                }

                if (!$is_downloading) {
                    // Der Anfrage-Marker sagt der Gegenseite, aus welchem der
                    // beiden Nektar-Ordner sie ausliefern soll - am Dateinamen
                    // allein ist das nicht zu erkennen.
                    $this->start_async_download(
                        $sender_ip, $filename, $local_path,
                        $type === 'NEKTAR_LONG' ? 'LONG:' : ''
                    );
                }
            }
        }
    }

    private function is_nektar_already_processed($filename) {
        $safe_name = basename($filename);
        return file_exists('./factory/archive/' . $safe_name)
            || file_exists('./factory/trash/' . $safe_name)
            // AUDIT-FIX (2026-08-19): 154 fehlformatierte hf_SFT_*.txt (bare
            // [SYMBIO], kein [USER]-Tag, siehe docs/ROADMAP.md Update
            // 2026-08-19) nach factory/quarantine_sft_bare_symbio_20260819/
            // verschoben. Ohne diesen Zweig haette der Laptop (haelt 156
            // fast-identische Originale weiterhin in seinem factory/input/,
            // broadcastet zufaellig eine davon pro Zyklus) sie frueher oder
            // spaeter unbemerkt zurueckgesynct - derselbe Reinfektionsweg wie
            // beim "[Gekuerzt]"-Fund vom 15.08., nur ohne dieses Verzeichnis
            // im bisherigen Check.
            || file_exists('./factory/quarantine_sft_bare_symbio_20260819/' . $safe_name);
    }

    private function update_census($ip, $caste) {
        $file = './factory/swarm_census.json';
        $census = file_exists($file) ? json_decode(file_get_contents($file), true) : ['castes' => [], 'nodes' => []];
        if (!is_array($census)) $census = ['castes' => [], 'nodes' => []];

        $census['nodes'][$ip] = [
            'caste' => $caste,
            'last_seen' => time()
        ];
        
        $unique_castes = [];
        foreach ($census['nodes'] as $node) $unique_castes[] = $node['caste'];
        $census['castes'] = array_values(array_unique($unique_castes));
        
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
                $this->log_event('EVAPORATE', $ip, "Knoten aus Gedächtnis verdunstet.");
                $changed = true;
            }
        }
        
        if ($changed) {
            $unique_castes = [];
            foreach ($census['nodes'] as $node) $unique_castes[] = $node['caste'];
            $census['castes'] = array_values(array_unique($unique_castes));
            @file_put_contents($file, json_encode($census), LOCK_EX);
        }
    }

    private function broadcast_local_state() {
        $sock = @socket_create(AF_INET, SOCK_DGRAM, SOL_UDP);
        if (!$sock) return;

        socket_set_option($sock, SOL_SOCKET, SO_BROADCAST, 1);
        
        $entities = [];

        // AUDIT-FIX: glob() sortiert alphabetisch, nicht nach Erstellzeit. Ohne
        // explizite Sortierung bewarb der Broadcast dauerhaft nur EINE Datei pro
        // Kategorie (die alphabetisch erste) - bei mehreren Prinzessinnen/Apex-
        // Modellen blieb eine neu entstandene Datei fuer Gossip strukturell
        // unsichtbar, wenn eine aeltere (dem Peer schon bekannte) Datei
        // alphabetisch davor einsortiert (z.B. "princess_GEN_..." vor jedem
        // "princess_SFT_..."). Live beobachtet: eine fertige GEN-Prinzessin kam
        // nie an, weil der Broadcast weiter die alte GEN-Prinzessin bewarb.
        // Neueste Datei zuerst, analog zur bewussten Best-Loss-Sortierung bei
        // $drones weiter unten.
        $apex = glob('./factory/apex/*.apex') ?: [];
        if (!empty($apex)) {
            usort($apex, fn($a, $b) => filemtime($b) <=> filemtime($a));
            $entities['APEX'] = basename($apex[0]);
        }

        $cortex = glob('./factory/royal/*.cortex') ?: [];
        if (!empty($cortex)) {
            usort($cortex, fn($a, $b) => filemtime($b) <=> filemtime($a));
            $entities['CORTEX'] = basename($cortex[0]);
        }
        
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

        // AUDIT-FIX (2026-08-26): factory/input_long/ wurde ueberhaupt nicht
        // gesynct. Seit dem SFT-Nachschub-Fix vom selben Tag laufen 50 % der
        // SFT-Jagden im Lang-Modus, und der Laptop-Scout produzierte damit
        // Material, das er als reiner Scout nie selbst destilliert - 49 MB lagen
        // dort fest, waehrend royal_distillery.php:468 auf dem Hauptrechner
        // genau darauf wartet (Gate ueber Ctx >= 2048).
        //
        // Eigener Entity-Typ statt einfach mitgeglobbt: die Routung auf der
        // Serverseite (accept_tcp_client) laeuft ueber die DATEIENDUNG, und die
        // ist bei kurzem und langem Nektar identisch ".txt". Ohne eigenen Typ
        // landete Lang-Material in factory/input/ - dort zieht es die Prinzessin
        // mit Ctx 1024 und bekommt 4000-Zeichen-Paare, also genau das, was der
        // getrennte Ordner verhindern soll.
        //
        // Rueckwaertskompatibel in beide Richtungen: ein Node mit altem Stand
        // bekommt fuer 'NEKTAR_LONG' aus get_path_for_type() null zurueck und
        // ueberspringt den Eintrag; und er bietet den Typ selbst nie an, also
        // fragt ein neuer Node bei ihm auch nie danach.
        $long_data = glob('./factory/input_long/*.txt');
        if (!empty($long_data)) {
            shuffle($long_data);
            $entities['NEKTAR_LONG'] = basename($long_data[0]);
        }

        $payload = json_encode([
            'origin' => $this->local_ip,
            'timestamp' => time(),
            'caste' => $this->profile['caste'],
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

    // =========================================================================
    // I/O: ARTERIE & MAGEN (Non-Blocking Transfers)
    // =========================================================================

    private function get_path_for_type($type, $filename) {
        $safe_name = basename($filename);
        if ($type === 'APEX') return './factory/apex/' . $safe_name;
        if ($type === 'CORTEX') return './factory/royal/' . $safe_name;
        if ($type === 'DRONE') return './factory/experts/' . $safe_name;
        if ($type === 'NEKTAR') return './factory/input/' . $safe_name;
        if ($type === 'NEKTAR_LONG') return './factory/input_long/' . $safe_name;
        return null;
    }

    private function accept_tcp_client() {
        $client = @stream_socket_accept($this->tcp_sock, 0);
        if (!$client) return;
        
        stream_set_blocking($client, true);
        stream_set_timeout($client, 0, 500000); 
        
        // Routing-Marker vor dem basename() abtrennen, aber basename() als
        // LETZTEN Schritt behalten - es ist der Schutz gegen Pfad-Traversal.
        // "LONG:../../etc/passwd" verliert im basename() den Marker gleich mit
        // und faellt danach durch die Endungs-Routung, landet also nirgends.
        $raw_req    = trim(fgets($client, 256));
        $is_long_req = str_starts_with($raw_req, 'LONG:');
        $req = basename($is_long_req ? substr($raw_req, strlen('LONG:')) : $raw_req);

        stream_set_blocking($client, false);

        // AUDIT-FIX: .heads-Sidecar-Anfragen (z.B. "expert_xxx.snai.heads") werden
        // ueber das Verzeichnis der ZUGRUNDELIEGENDEN Datei aufgeloest - dieselbe
        // Endungs-Routung wie fuer die Hauptdatei, nur mit ".heads" wieder angehaengt.
        $is_heads_req = str_ends_with($req, '.heads');
        $base_req = $is_heads_req ? substr($req, 0, -strlen('.heads')) : $req;

        $file_path = null;
        if (str_ends_with($base_req, '.apex')) $file_path = './factory/apex/' . $base_req;
        elseif (str_ends_with($base_req, '.cortex')) $file_path = './factory/royal/' . $base_req;
        elseif (str_ends_with($base_req, '.snai')) $file_path = './factory/experts/' . $base_req;
        elseif (str_ends_with($base_req, '.txt')) {
            $file_path = ($is_long_req ? './factory/input_long/' : './factory/input/') . $base_req;
        }

        if ($file_path && $is_heads_req) $file_path .= '.heads';

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

    // $req_prefix: optionaler Routing-Marker vor dem Dateinamen ('LONG:' fuer
    // factory/input_long/). Leer fuer alles andere - dann ist die Anfrage
    // Byte-fuer-Byte dieselbe wie vor 2026-08-26, ein Node mit altem Stand
    // versteht sie also unveraendert.
    private function start_async_download($ip, $filename, $final_path, $req_prefix = '') {
        $sock = @stream_socket_client("tcp://$ip:" . TENSOR_PORT, $errno, $errstr, 0.2);
        if (!$sock) {
            return;
        }

        stream_set_timeout($sock, 1);
        @fwrite($sock, $req_prefix . basename($filename) . "\n");
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
                        // AUDIT-FIX (2026-08-15): eingehender Text-Nektar (.txt)
                        // kann von einem Peer mit aelterem/ungepatchtem
                        // Collector-Code stammen (der Laptop laeuft bewusst auf
                        // altem Codestand, siehe CLAUDE.md) - ohne Pruefung
                        // wuerde ein bereits hier lokal gefixter Korpus-Bug
                        // (z.B. der "[Gekuerzt]"-Trunkierungsmarker aus
                        // collect_sft.php, siehe docs/ROADMAP.md Update
                        // 2026-08-15) per Gossip unbemerkt zurueck in den
                        // lokalen Trainingskorpus geschleust werden, obwohl
                        // Collector UND Bestand hier bereits bereinigt sind.
                        // Modell-Dateien (.snai/.cortex/.apex) sind davon nicht
                        // betroffen, nur reiner Text-Nektar wird geprueft.
                        $is_model_file = !str_ends_with($job['filename'], '.txt');
                        if (!$is_model_file) {
                            $this->sanitize_incoming_nectar($job['tmp_path'], $job['filename'], $job['ip']);
                        }

                        rename($job['tmp_path'], $job['final_path']);
                        $this->log_event('SUCCESS', $job['ip'], "Artefakt assimiliert (Instant-Close): {$job['filename']}");

                        // AUDIT-FIX: .heads-Sidecar der Quelle nachziehen. Ohne das
                        // faellt resolve_num_heads() beim naechsten Laden mangels
                        // Sidecar auf Legacy-1 zurueck und deutet die per Gossip
                        // empfangenen (ggf. mit >1 Head trainierten) Attention-
                        // Gewichte still um. Text-Nektar (.txt) hat keine Heads.
                        if ($is_model_file && !file_exists($job['final_path'] . '.heads')) {
                            $this->start_async_download($job['ip'], $job['filename'] . '.heads', $job['final_path'] . '.heads');
                        }
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

    /**
     * Entfernt bekannte, hier bereits lokal gefixte Korpus-Bug-Marker aus
     * frisch per Gossip empfangenem Text-Nektar, BEVOR die Datei unter
     * ihrem finalen Namen live geht (Aufrufstelle: process_downloads(),
     * nur fuer .txt-Dateien). Verteidigungslinie gegen Peers mit
     * aelterem/ungepatchtem Collector-Code (z.B. der Laptop, siehe
     * CLAUDE.md) - ohne diese Pruefung koennte ein hier bereits gefixter
     * und aus dem Bestand entfernter Bug jederzeit ueber einen Gossip-Sync
     * zurueckkommen. Liste waechst bei Bedarf um weitere bekannte
     * Artefakte, analog zu retrofit_gekuerzt_marker.php fuer den bereits
     * vorhandenen lokalen Bestand.
     */
    private function sanitize_incoming_nectar(string $tmp_path, string $filename, string $ip): void {
        static $known_markers = ["... [Gekürzt]"];

        // Die Tabelle steht in korpus_sanitize.php - EINMAL fuer alle drei
        // Schreiber nach factory/input (dieser Node, collect_definitions.php,
        // collect_gemma.php). Sie hier inline zu halten hat am 01.09. schon
        // eine zweite Kopie erzwungen; eine dritte waere garantiert
        // auseinandergelaufen. Begruendung und Historie: korpus_sanitize.php.

        $content = file_get_contents($tmp_path);
        if ($content === false) return;

        $cleaned = str_replace($known_markers, '', $content);
        $cleaned = korpus_bereinigen($cleaned);

        if ($cleaned !== $content) {
            file_put_contents($tmp_path, $cleaned);
            $this->log_event('SANITIZED', $ip,
                sprintf("%s bereinigt (%+d Bytes).", $filename, strlen($cleaned) - strlen($content)));
        }
    }

    // =========================================================================
    // DUAL-LOGGING (JSON + ANSI TERMINAL)
    // =========================================================================

    private function log_event($type, $node_ip, $message) {
        $time = date('H:i:s');
        
        // --- 1. Terminal Ausgabe (ANSI Colors) ---
        // Verhindert das Zerhacken der Pulse-Line durch einen Zeilenumbruch vorher
        echo "\n"; 
        
        $color = "\e[0m"; // Default (Reset)
        if ($type === 'SUCCESS' || $type === 'UPLOAD_DONE') $color = "\e[32m"; // Grün
        elseif ($type === 'ERROR' || $type === 'EVAPORATE') $color = "\e[31m"; // Rot
        elseif ($type === 'DOWNLOAD_START' || $type === 'UPLOAD_START') $color = "\e[36m"; // Cyan
        
        echo "{$color}[{$time}] [{$type}] Node: {$node_ip} | {$message}\e[0m\n";

        // --- 2. JSON Logging fürs Dashboard ---
        $log_file = './factory/pheromone_log.json';
        $logs = file_exists($log_file) ? json_decode(file_get_contents($log_file), true) : [];
        if (!is_array($logs)) $logs = [];
        
        array_unshift($logs, [
            'time' => $time,
            'type' => $type,
            'node' => $node_ip,
            'msg' => $message
        ]);
        
        if (count($logs) > 30) array_pop($logs);
        @file_put_contents($log_file, json_encode($logs), LOCK_EX);
    }
}

function register_expert_in_index($id, $loc, $offset, $fp, $idx_path) {
    $entry = pack('A12', $id) . pack('C', 0) . pack('A16', $loc) . pack('L', $offset);
    foreach ($fp as $val) $entry .= pack('c', $val);
    file_put_contents($idx_path, $entry, FILE_APPEND | LOCK_EX);
}

// CLI Bootstrap
if (php_sapi_name() === 'cli' && basename(__FILE__) === basename($_SERVER['SCRIPT_FILENAME'])) {
    $daemon = new SnaiGossipNode();
    $daemon->pulse();
}