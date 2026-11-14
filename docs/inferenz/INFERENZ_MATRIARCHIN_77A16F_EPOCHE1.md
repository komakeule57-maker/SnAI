# Inferenz-Test: neue Matriarchin `77a16f`, Zwischenstand nach Epoche 1 von 6

Kein Vergleichs-/Fix-Validierungstest wie die bisherigen `docs/inferenz/INFERENZ_*.md`-
Dokumente, sondern ein Status-Snapshot  ("wie ist der
aktuelle Stand der Matriarchinsynthese?") — ergänzt um einen spontanen
Inferenztest auf dem gerade erst fertigen Epoche-1-Checkpoint, während
Epoche 2 bereits lief. Zusätzlich relevant: die erste echte Produktiv-
Aktivierung des Neuron-Revival-Live-Hooks (2026-08-15) fiel in diese
Epoche.

## Testaufbau

- **Modell:** `secured_models/matriarchin_GEN_77a16f_epoch1.apex` — Kopie
  von `factory/apex/matriarchin_GEN_77a16f.apex` unmittelbar nach Epoche 1,
  **vor** dem Start von Epoche 2 gesichert (byte-identisch per `sha256sum`
  verifiziert), da dieselbe Datei als Basis für Epoche 2 diente und danach
  überschrieben worden wäre. Liegt außerhalb von `factory/`, von keinem
  `glob()`-Scan der Pipeline erfasst.
- **Architektur:** 512 Dim, 4 Layers, 8 Heads.
- **Herkunft:** APEX-SYNTHESE, Net2Net-Expansion aus
  `expanded_Matriarchin_apex_6a7f8bd32caa3.apex`. Session
  `matriarchin_GEN_77a16f`, GEN-Kalibrierung über 6 geplante Epochen
  (10 Artefakte/Epoche).
- **Epoche 1** (aus `train_matriarchin_GEN_77a16f.jsonl`): Train-Loss
  5.127227, Held-out-Loss **5.340738** (PPL 208.6666), Dauer **7h 42min**
  (`epoch_duration_sec: 27751.92`). Direkt danach automatisch als
  `best_matriarchin_GEN_77a16f.snai` gesichert (erster/bisher einziger
  Bestwert).
- **Neuron-Revival-Hook aktiv:** direkt nach Epoche 1 automatisch **4 tote
  Neuronen revivt** (3× `L1_w_down`, 1× `L2_w_down`, Skala ±0.02,
  Cooldown-Log `factory/checkpoints/revival_log_matriarchin_GEN_77a16f.json`).
  Der hier getestete gesicherte Checkpoint enthält diese Änderung bereits
  (Revival lief noch vor dem Sichern des Epoche-1-Standes).
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Sliding Window 64 Tokens), min_p 0.05, max. 256 neue Tokens.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL. Lief **parallel** zum
  aktiven Epoche-2-Training auf derselben GPU — keine Fehler, kein
  beobachteter Konflikt, kurzer Forward-Pass.

## Ergebnisse (drei Standard-Prompts)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (290 Zeichen)

Kollabiert fast sofort in Markdown-/Code-Fence-Müll (Backticks,
Tabellen-Syntax `align="..."`, `row...`) — kein zusammenhängender Text.
Kein Stop-Token erreicht, Generierung läuft bis ans Zeichen-Limit.

### Prompt 2: „Was ist ein Computer?“ (375 Zeichen)

Gleiches Kollaps-Muster wie Prompt 1 — Wiki-Tabellensyntax-Fragmente
(`class="hintergrundfarbe1"`, `rowspan=`, `align="center"`) statt Antwort.
Kein Stop-Token erreicht.

**Nachtrag:** dieser Output war der direkte Anlass für die Entdeckung des
Wikitabellen-/Vorlagen-Lecks im GEN-Trainingskorpus (2026-08-15). Die
hier sichtbaren Fragmente sind **kein vom Modell erfundenes Artefakt**,
sondern memoriertes, tatsächlich im (zum Testzeitpunkt noch unbereinigten)
Trainingskorpus vorhandenes rohes MediaWiki-Markup.

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (1059 Zeichen)

Deutlich kohärenter als die anderen beiden: durchgehend echte deutsche
Sätze mit halbwegs plausibler Grammatik, auch wenn inhaltlich am
eigentlichen Thema vorbei und mit erfundenen Wörtern durchsetzt
("Stähne", "Kokals"). Kein Stop-Token erreicht, Generierung läuft bis ans
Zeichen-Limit.

## Einordnung

Deckt sich mit dem bekannten Muster aus `docs/inferenz/INFERENZ_MATRIARCHIN_
REGULARISIERT.md` (2-3 von 3 Prompts kollabieren, kein sauberer Stopp) —
aber **nicht direkt vergleichbar**: diese Matriarchin steht erst bei
Epoche 1 von 6 geplanten, während die vorige (`9d26ec`) mit vollen 3
Epochen getestet wurde. Die Qualität hier ist erwartbar roh/früh, kein
Regressions-Befund. Kein A/B-Test gegen eine unrevivte Variante dieser
Epoche geplant oder durchgeführt — das separate, kontrollierte
Offline-A/B deckt die
Revival-Wirksamkeitsfrage bereits methodisch sauberer ab.

Wichtigster Nebenertrag dieses Tests: Prompt 2s Output führte direkt zur
Entdeckung und Behebung eines Datenqualitätsproblems, das schätzungsweise
96% der GEN-Korpusdateien betraf (2026-08-15) — ein Fall, in dem ein einzelner Inferenz-Stichprobentest
einen Trainingsdaten-Bug aufgedeckt hat, der sonst unbemerkt geblieben wäre.
