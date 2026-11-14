# SnAI — Bugfix-Dokumentation

## 1. GPU-Erkennung & Performance

### 1.1 GPU wurde auf dem Laptop nicht erkannt
**Datei:** `cortex_opencl.nim` (`init_opencl_engine()`)
**Symptom:** OpenCL initialisierte nie eine GPU, obwohl eine AMD-APU mit Mesa/RustiCL
vorhanden war.
**Ursache:** RustiCL meldet dieses Gerät als `CL_DEVICE_TYPE_CUSTOM`, nicht als
`CL_DEVICE_TYPE_GPU`. Die Geräteauswahl suchte hart nach `CL_DEVICE_TYPE_GPU` und fand
nichts.
**Fix:** Fallback ergänzt — wird kein `CL_DEVICE_TYPE_GPU` gefunden, wird das erste
Nicht-CPU-Gerät akzeptiert.

### 1.2 GPU-Trainingspfad war 20–40x langsamer als CPU
**Dateien:** `cortex.nim`, `cortex_opencl.nim`
**Symptom:** Sobald GPU aktiv war, brach die Trainingsgeschwindigkeit massiv ein statt
zu steigen.
**Ursache:** Die Attention-Berechnung lief Position für Position (ein Kernel-Dispatch
pro Token), während Matrixmultiplikationen bereits gebündelt liefen. Der
Dispatch-Overhead pro einzelnem OpenCL-Aufruf dominierte die eigentliche Rechenzeit.
**Fix:** Kompletter Umbau auf `batched_attention` (layer-major statt positionsmajor):
neue Kernel `attentionForwardBatched`, `attentionBackwardBatchedQuery`,
`attentionBackwardBatchedKey` in `cortex_opencl.nim`, neue Host-Funktionen
`ocl_attentionForwardBatched()`/`ocl_attentionBackwardBatched()`, neue
`CortexState`-Felder (`d_hidden_all`, `d_att_out_all`, `d_q_all`, Scratch-Puffer) sowie
Restrukturierung von `cortex_train_step()` (Forward- und Backward-Pass) in
`cortex.nim`. Zusätzlich w_q/w_k/w_v/w_o/w_up/w_down auf die bereits vorhandene
`matMulBatchedD`/`matMulTBatchedD`/`gradAccumBatchedD`-Infrastruktur umgestellt.
Ergebnis: 15–19.8x schneller, numerisch bit-identisch zum alten Pfad (verifiziert über
CPU/GPU, 1/8 Heads, bis zu 20 Trainingsschritte).
**Verbleibende Erkenntnis:** Auf diesem Laptop (AMD Renoir APU, RustiCL) bleibt GPU
danach immer noch ca. 2x langsamer als CPU (12 OpenMP-Threads) — kein Dispatch-Problem
mehr, sondern vermutlich eine Treiber/Hardware-Grenze. Bewusste Entscheidung: die
GPU-Schwelle (`$model_dim >= 512`) bleibt unverändert, um maschinenübergreifend
konsistent zu bleiben.

### 1.3 Eigener Bug in der eigenen Batched-Attention-Implementierung
**Datei:** `cortex_opencl.nim` (`ocl_attentionForwardBatched`/`ocl_attentionBackwardBatched`)
**Symptom:** Bei Multi-Step-Tests driftete der Loss ab Schritt 6–7 messbar auseinander
(bis zu 0.013 Abweichung).
**Ursache:** `ocl_get_buffer()` lädt Host-Daten nur beim *ersten* Aufruf für einen
gegebenen Pointer hoch. k_cache/v_cache ändern sich aber zwischen Trainingsschritten —
der gecachte GPU-Puffer blieb auf dem Stand von Schritt 1 hängen.
**Fix:** Explizites `clEnqueueWriteBuffer` für k_cache/v_cache vor jedem
Batched-Attention-Dispatch.

### 1.4 `diag_check.php`s eigener Messfehler (vorbestehend, nicht selbst verursacht)
**Datei:** `diag_check.php`
**Symptom:** Test 3 ("CPU only vs. GPU") lieferte unbrauchbare Ergebnisse.
**Ursache:** `gpu_initialized` in `cortex.nim` ist ein modulweites globales Flag, keine
Instanz-Eigenschaft. Test 2 hatte das Flag bereits auf GPU gekippt, bevor Test 3 lief.
**Reaktion:** Als bekannte Einschränkung dokumentiert (Kommentar in
`diag_common.php`) statt "gefixt" — ein echter CPU-vs-GPU-Vergleich braucht zwei
getrennte PHP-Prozesse (`diag_step_cpu.php` + `diag_step_gpu.php` + `diag_compare.php`).

---

## 2. Multi-Head-Attention-Umstellung (Net2Net-Skalierung)

Ziel: System vollständig auf Multi-Head-Betrieb umstellen, sodass die Anzahl
der Attention-Heads mit der Net2Net-Morphism-Synthese/Destillation mitskaliert, statt
für immer bei 1 Head (Legacy) zu verharren.

### 2.1 Head-Zahl ging beim Net2Net-Upscale verloren
**Dateien:** `head_config.php` (neu), `royal_distillery.php`
**Symptom:** Beim Aufstieg auf die nächste Evolutionsstufe (z.B. Drohne → Prinzessin)
wusste das System nicht, wie viele Attention-Heads das expandierte Modell haben sollte.
**Fix:** Neues Modul `head_config.php` mit `head_sidecar_path()`, `suggest_num_heads()`
(Ziel-Kopfgröße 64), `resolve_num_heads()` (Sidecar bindend, sonst Legacy-Fallback = 1,
eingefroren) und `write_num_heads()`/`migrate_heads()`. `royal_distillery.php` erhielt
konditionale Skalierungslogik: nur wachsen, wenn die Quelle bereits headSize=64-konform
war, sonst Head-Zahl beibehalten (verhindert das Zerstören bereits gelernter
Attention-Gewichte).

### 2.2 `.heads`-Sidecar ging an *sechs* verschiedenen Stellen verloren
Der immer gleiche Fehlerklasse — eine Datei wird kopiert/verschoben/umbenannt, der
zugehörige `.heads`-Sidecar aber nicht mit —, trat unabhängig an sechs Stellen auf:

1. **Steuer-Daemon**, REFLECT-Umbenennung: Sidecar fehlte nach dem
   kosmetischen Loss-Suffix-Rename.
2. **Steuer-Daemon**, DIGEST-Umbenennung: dito.
3. **Steuer-Daemon**, `action_distill()`s Tribut-Kopie nach `factory/tributes/`:
   `distill_princess()` bekam nie mit, welche Head-Struktur die Quellen hatten, fiel
   dadurch immer auf Legacy-1 zurück.
4. **`protocol_weiselzelle.php`**, Lehrer-Kopie (~Zeile 70).
5. **`snai_gossip_node.php`**: `.heads`-Requests wurden vom TCP-Server gar nicht
   aufgelöst, und nach einem erfolgreichen Modell-Download wurde nie automatisch auch
   der Sidecar angefragt.
6. **`royal_distillery.php`**, Tribut-Verbrauch nach erfolgreicher Destillation
   (`distill_princess()` und `distill_apex_entity()`): die `.snai`-Datei wanderte nach
   `factory/royal_fragments/`, der Sidecar blieb als Waise in `factory/experts/` zurück
   — **gefunden erst, nachdem die Endlosschleife aus Abschnitt 3.3 live beobachtet
   wurde.**

**Fix (überall gleich):** `@rename(head_sidecar_path($old), head_sidecar_path($new))`
parallel zum Haupt-Rename/Copy ergänzt. In `snai_gossip_node.php` zusätzlich:
`accept_tcp_client()` löst `X.heads`-Requests jetzt auf, `process_downloads()` fragt
nach jedem erfolgreichen Modell-Download automatisch den passenden Sidecar mit an.

---

## 3. Destillations-Pipeline (Prinzessin/Matriarchin-Synthese)

### 3.1 `$unique_selected` undefiniert
**Datei:** `royal_distillery.php`
**Symptom:** Verbrauchte Vorgänger-Modelle wanderten nie nach
`factory/royal_fragments/` — blieben für immer als Kandidaten sichtbar, Endlosschleife
auf derselben Generation.
**Ursache:** Beim vorherigen Tags-Fix wurde die `array_unique()`-Zeile entfernt, der
darauffolgende Loop hing aber noch davon ab.
**Fix:** `$unique_selected = array_unique($selected_for_merge);` wiederhergestellt.

### 3.2 Head-strukturell inkompatible Tribute wurden trotzdem zusammengemischt
**Datei:** `royal_distillery.php`
**Symptom:** Mit einem gemischten Bestand (alte Legacy-1-Head- neben neuen
2-Head-Experten) versuchte das System, 3 strukturell inkompatible Tribute zu einer
Prinzessin zu verschmelzen.
**Fix:** Neue Methode `largest_head_compatible_group(array $files, int $min_count)`:
gruppiert Kandidaten nach tatsächlicher (per `resolve_num_heads()` aufgelöster)
Head-Zahl, gibt die größte Gruppe zurück, die die Mindestgröße erreicht. Eingebaut in
`distill_princess()`s Tag-Gruppierung und `distill_apex_entity()`s
SFT/GEN-Aufteilung.

### 3.3 Auswahllogik und Feasibility-Check liefen auseinander → Endlosschleife
**Ort:** Steuer-Daemon (`action_distill()`, `get_dominant_drive()`)
**Symptom:** Live beobachtet: der Daemon startete wiederholt eine volle Destillation
("MAJESTÄTISCHE SYNTHESE"), die drüben in `distill_princess()` sofort mit "Keine 3
head-strukturkompatiblen Tribute" abbrach — und das in einer Schleife, die **~500
Zyklen in 12 Sekunden** durchlief (Busy-Spin, da der Abbruchpfad keinen
Hintergrundprozess startet, der die Pulse-Loop natürlich bremst).
**Ursachen (zwei getrennte Bugs):**
1. `action_distill()` wählte Tribute rein nach Zeitstempel, ohne die Head-Kompatibilität
   zu prüfen, die `distill_princess()` intern längst durchsetzte (Auswahl und
   Feasibility-Check waren inkonsistent).
2. `get_dominant_drive()` empfahl DISTILL rein nach *Anzahl* (`$sft_e >= 3`), ohne zu
   wissen, ob ein Kompatibilitäts-Konflikt vorlag — fiel dadurch nie zu DIGEST/HUNT
   durch, um neues, passendes Material zu holen.
**Fix:**
- `action_distill()` filtert SFT/GEN-Kandidaten jetzt selbst über
  `largest_head_compatible_group()`, *bevor* Tribute kopiert oder ATP verbrannt wird.
- `get_dominant_drive()` nutzt denselben Kompatibilitäts-Check, damit die Empfehlung
  DISTILL nur noch dann kommt, wenn `action_distill()` sie auch wirklich umsetzen
  kann — sonst korrekter Fallthrough zu DIGEST/HUNT.
- Zusätzliches Sicherheitsnetz: `sleep(2)` im verbleibenden Abbruchpfad von
  `action_distill()`, falls trotzdem einmal kein Hintergrundprozess läuft (verhindert
  jeden zukünftigen Busy-Spin dieser Art, unabhängig von der Ursache).

### 3.4 ATP-Gate blockierte DISTILL/ASCEND unbegrenzt unter Dauerlast
**Ort:** Steuer-Daemon (`get_dominant_drive()`)
**Symptom:** Live beobachtet: 6 SFT-Experten lagen bereit, DISTILL feuerte trotzdem
nie.
**Ursache:** DISTILL/ASCEND waren hinter `$this->atp > 50` gated, aber der
DIGEST-Fallback weiter unten hat *kein* ATP-Gate. Unter Dauerlast (4 Worker auf 12
Kernen) brennt `update_metabolism()` ATP praktisch nie über 50, während Gossip
ständig neue Nektar-Dateien nachlieferte — DIGEST gewann gegen DISTILL/ASCEND auf
unbestimmte Zeit.
**Fix:** ATP>50-Gate für DISTILL/ASCEND entfernt. Reife Ernten haben jetzt Vorrang vor
weiterem DIGEST, unabhängig vom ATP-Level (ASCEND behält seine eigene, feinere
`atp > 80`-Präferenz für den besonders teuren Aufstieg).

### 3.5 Kopf-Struktur-Mischung galt nicht für den GEN-Pool
**Datei:** `royal_distillery.php` (`distill_apex_entity()`)
**Symptom:** Die Head-Kompatibilitäts-Prüfung aus 3.2 wurde nur für die
Prinzessin-Destillation eingebaut, nicht für die Matriarchin/Apex-Synthese, die
zusätzlich in SFT-/GEN-Pools aufteilt.
**Fix:** `largest_head_compatible_group()` wird jetzt auch vor der SFT/GEN-Aufteilung
in `distill_apex_entity()` angewendet.

### 3.6 `action_distill()` kopierte Tribute ohne ihre `.heads`-Sidecars nach `tributes/`
Siehe 2.2, Punkt 3 — separat gelistet wegen Bezug zu diesem Themenblock.

---

## 4. Gossip/P2P-Netzwerk (`snai_gossip_node.php`)

### 4.1 Sidecar-Transferlücke
Siehe 2.2, Punkt 5.

### 4.2 Bereits verarbeitete Nektar-Dateien wurden endlos neu geholt
**Datei:** `snai_gossip_node.php` (`handle_udp_payload()`)
**Symptom:** Live beobachtet: `hf_GEN_6d781588.txt` und `hf_SFT_6a8cff0b.txt` liefen
mehrfach zwischen zwei Knoten hin und her (Up-/Download alle ~13 Minuten), obwohl
beide Inhalte längst digestet waren.
**Ursache:** Der Dedup-Check vor einem Download prüfte nur `file_exists()` in
`factory/input/`. Nach erfolgreichem Digest wandert die Datei aber nach
`factory/archive/` (bzw. `factory/trash/`) und verschwindet aus `input/` — aus Sicht
des Checks sah es dann so aus, als fehle sie noch, obwohl sie bereits einmal komplett
verarbeitet wurde. Jede erneute Ankündigung eines Peers löste einen kompletten
Re-Download + Re-Digest identischer Inhalte aus.
**Fix:** Neue Methode `is_nektar_already_processed()` prüft zusätzlich
`factory/archive/` und `factory/trash/`, bevor ein NEKTAR-Download gestartet wird.

---

## 5. Datenpipeline & Trainingsqualität

### 5.1 BPE-Tokenizer: Phantom-Dekrement bei doppelt auftretendem Merge-Paar
**Datei:** `bpe_init/bpe_forge.nim`
**Symptom:** Bei einem Wort, in dem dasselbe Merge-Paar zweimal direkt hintereinander
vorkam (Muster A,B,A,B), ging ein Paar-Zähler dauerhaft negativ.
**Ursache:** `newToks[^1]` zeigte beim zweiten Treffer auf das gerade erst
verschmolzene Token aus dem ersten Treffer, nicht auf einen echten Korpus-Nachbarn —
ein Dekrement traf ein Paar, das nie einen positiven Zähler hatte.
**Fix:** `prevWasMerge`-Flag ergänzt, verhindert den Phantom-Dekrement, wenn der
vorherige Schritt selbst ein Merge war. Verifiziert über isolierte Nim-Reproduktions-
Skripte vor und nach dem Fix.
**Wichtiger Nebenbefund:** `symbio_vocab.json` (aktuell im Einsatz) wurde **vor**
diesem Fix gebaut (29. Juli) — die Merge-Reihenfolge könnte
stellenweise suboptimal sein. Neubau bewusst zurückgestellt, da neue Token-IDs alle
bestehenden Checkpoints inkompatibel machen würden.

### 5.2 Fehlende Dateigrößen-Validierung vor `np.memmap()`
**Dateien:** `skripte/morph_audit.py`, `skripte/rms_probe.py`
**Symptom:** Auf Legacy-Dateien crashten beide Skripte mit unklarer `ValueError`.
**Fix:** Dieselbe Größenprüfung ergänzt, die `morph_fix.py` bereits hatte — sauberer
Abbruch statt kryptischem Crash.

### 5.3 Trainings-Chunks konnten mitten in einem Wort oder UTF-8-Zeichen beginnen
**Datei:** `train_cluster.php`
**Symptom:** Der "Darwin-Entropie"-Offset (host-spezifischer, deterministischer
Zufallsstart pro Trainingsdatei) sprang an eine rohe Byte-Position — potenziell mitten
in ein Wort oder sogar mitten in ein UTF-8-Mehrbyte-Zeichen (Umlaute).
**Fix:** Der Offset wird nach der Berechnung bis zum nächsten Whitespace vorgeschoben
(mit Ringpuffer-Suche für den Fall, dass das Suchfenster ans Dateiende läuft). Rein
lokal in einem frisch gestarteten Subprozess wirksam, kein Neustart des Daemons nötig.

---

## 6. Sonstiges

### 6.1 `tensor_io.php`: selbstverursachter Parse-Fehler
**Datei:** `tensor_io.php`
**Symptom:** Eigener neuer Docblock-Kommentar enthielt die Literalfolge `*/` (in
"m_\*/v_\*"), schloss den PHP-Kommentar vorzeitig — Parse-Fehler.
**Fix:** Umformulierung, die die Zeichenfolge `*/` vermeidet.

### 6.2 `matriarch_poc.php`: veralteter Methodenaufruf
**Datei:** `matriarch_poc.php`
**Symptom:** Rief `distill_matriarch()` auf, das inzwischen zu `distill_apex_entity()`
generalisiert worden war.
**Fix:** Aufruf angepasst.

### 6.3 `snai_inference.php`: Head-Zahl per Heuristik statt Sidecar
**Datei:** `snai_inference.php`
**Symptom:** `$num_heads = max(1, (int)($hidden_dim / 64));` statt echter
Sidecar-Auflösung — ignorierte Legacy-1-Head-Modelle mit größerem `hidden_dim`.
**Fix:** Ersetzt durch `resolve_num_heads($model_file, $hidden_dim)`.

### 6.4 libgomp-Exit-Race bei kurzlebigen OpenMP-Prozessen
**Dateien:** `snai_inference.php`, `cortex_bridge.php`, `tensor_ffi_adapter.php`
**Symptom:** Gelegentlicher Segfault kurz nach Prozessende bei Skripten, die
OpenMP-parallelisierte Funktionen (`matVecMul` u.a.) nutzen und sich kurz danach
beenden. Per gdb-Backtrace verifiziert: alle Threads hingen in `libgomp.so.1` fest.
**Fix:** `putenv('OMP_WAIT_POLICY=PASSIVE');` vor dem FFI-Laden von `libcortex.so` in
allen drei Dateien ergänzt.

### 6.5 `cp`-Alias verschluckte Binary-Swaps stillschweigend
**Kein Code-Bug**, sondern eine Shell-Falle während der Arbeit: `alias cp='cp -i'`
sorgte dafür, dass ein `cp libcortex.so ...`-Aufruf ohne TTY interaktiv nachfragte und
dabei No-Op blieb — per `md5sum`-Vergleich entdeckt. Seitdem `\cp -f` für alle
Binary-Swaps verwendet.

---

## 7. GPU-Backend-Diagnosemeldungen kontaminierten die Inferenz-Ausgabe (2026-08-15)

### 7.1 Native GPU-Statusmeldungen liefen auf STDOUT statt STDERR
**Dateien:** `cortex_opencl.nim` (`init_opencl_engine()`, `shutdown_opencl_engine()`,
`ocl_get_buffer()`), `cortex_cuda.cu` (`init_cuda_engine()`, `shutdown_cuda_engine()`)
**Symptom:** In der Inferenz-Ausgabe klebte ein einzelnes, scheinbar zusammenhangloses
Wort direkt vor der `⚡ [SYMBIO VRAM]`-Meldung (z.B. "Die⚡ [SYMBIO VRAM]..."). Live
gegen den echten `dashboard.php`-Endpunkt verifiziert: nicht nur das eine Wort betroffen
— die komplette Diagnoseausgabe (Architektur-Banner, Timing, Metrik-Footer) landete
unbereinigt in der als "AI-Antwort" angezeigten Chat-Nachricht.
**Ursache:** Alle GPU-Backend-Status-/Fehlermeldungen (`⚡ [SYMBIO VRAM]`,
`🚨 [SYMBIO GPU]`) nutzten `printf()`/`std::cout` — anders als jede sonstige
Diagnosemeldung im Projekt, die konsequent STDERR nutzt. `snai_inference.php`
dokumentiert STDOUT explizit als reinen Generierungs-Textkanal ("Fängt das Dashboard
per STDOUT ab") — die fehlplatzierten `printf()`-Aufrufe unterliefen das. Das
verklebte Wort selbst war kein Bug (echtes erstes generiertes Token, deshalb wechselt
es je Prompt), nur optisch verklebt durch die native Stdio-Pufferung.
**Fix:** Alle 8 betroffenen Aufrufe (6× `cortex_opencl.nim`, 2× `cortex_cuda.cu`) auf
`fprintf(stderr, ...)`/`std::cerr` umgestellt — reine Kanaländerung, keine
Logikänderung. `libcortex.so`/`libcortex_cuda.so` neu gebaut und live getauscht (alte
Version als `*.bak_prestderrfix_20260815` gesichert); der zu diesem Zeitpunkt laufende
Trainingsprozess nutzte unverändert weiter die alte, bereits gemappte `.so`, kein
Neustart nötig.
