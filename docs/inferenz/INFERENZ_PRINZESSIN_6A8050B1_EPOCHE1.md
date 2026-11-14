# Inferenz-Test: neue Prinzessin `princess_SFT_6a8050b1332a5`, Zwischenstand nach Epoche 1 von 3

Kein A/B-Fix-Validierungstest, sondern ein Zwischenstand :
Vergleichspunkt "Kernfortschritt in Inferenzform" während des laufenden
Trainings, ergänzt um einen späteren Test der Endfassung (nach Epoche 3 /
ASCEND). Diese Prinzessin ist der erste komplette Lauf mit dem gesamten seit
dem 2026-08-15-Reset aktiven Fix-Stack: Loss-Masking, Weight-Decay/Dropout,
Label-Smoothing, GEN-`</s>`-Dichte-Retrofit, Wikitabellen-/Vorlagen-Corpus-
Cleanup und live Neuron-Revival.

## Testaufbau

- **Modell:** `secured_models/princess_SFT_6a8050b1332a5_epoch1.snai`
  (+ `.heads`) — Kopie von `factory/checkpoints/cp_princess_SFT_6a8050b1332a5_e1.snai`
  unmittelbar nach Epoche 1 gesichert, byte-identisch per `sha256sum`
  verifiziert (dieselbe Datei dient als Ladepunkt für Epoche 2 und wird
  dort nicht überschrieben, da pro Epoche ein eigener `_eN.snai`-Dateiname
  vergeben wird — trotzdem außerhalb von `factory/` gesichert, analog zum
  bisherigen Vorgehen).
- **Architektur:** 256 Dim, 2 Layers, 4 Heads (Net2Net-Wachstum aus einem
  128-Dim/2-Heads-Worker, nur Dim+Heads gewachsen, Layer-Zahl unverändert).
- **Epoche 1:** Train-Loss 6.468208, Held-out-Loss **5.980589** (PPL
  395.67), `gpu_backend: NONE` (erwartet, Tier unter 512 Dim).
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Sliding Window 64 Tokens), min_p 0.05, max. 256 neue Tokens.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL. Lief **parallel** zum
  aktiven Epoche-2-Training (CPU-Worker bei ~588% CPU) — kein Fehler, kein
  beobachteter Konflikt, alle drei Läufe unter 1,2s Wall-Zeit.

## Ergebnisse (drei Standard-Prompts)

### Prompt 1: „Erzähl mir etwas über die Natur.“

Erster Versuch: kein zusammenhängender Text, Interpunktions-/Silben-Müll
("-Sere, um das mitline-Serie, die die ist, dass die fübf...", diverse
verwaiste Klammern/Ziffern), läuft bis ans Zeichen-Limit, kein Stop-Token.
Zweiter Sample-Versuch (Stichprobe, Temperature-Sampling ist stochastisch):
„ Hier![SYMBIO]“ — stoppt nach 15 Zeichen. `[SYMBIO]` ist kein Artefakt-Bug,
sondern der reguläre, registrierte Rollen-Marker-Token (analog `[USER]` in
SFT-Dialogen, auch GEN-Chunks sind damit umschlossen) — das Modell hat ihn
memoriert und reproduziert ihn als (Fehl-)Antwort statt echten Fließtext zu
generieren.

### Prompt 2: „Was ist ein Computer?“

„ [Gekürzt]“ — stoppt nach 10 Zeichen. Vermutlich ein memorierter
Kürzungs-/Trunkierungs-Marker aus dem Korpus (nicht verifiziert, welche
Quelle genau). Frühester Stopp aller getesteten Prompts.

### Prompt 3: „Beschreibe einen sonnigen Tag.“

„ [Gekürzt]“ — identisches Muster wie Prompt 2, ebenfalls 10 Zeichen.

## Einordnung

Erwartbar roh für Epoche 1 von 3 direkt nach einem Net2Net-Sprung (128→256
Dim) — kein Regressionsbefund. Auffällig gegenüber früheren Epoche-1-Tests
(z. B. `docs/inferenz/INFERENZ_MATRIARCHIN_77A16F_EPOCHE1.md`, dort liefen alle drei
Prompts bis ans Zeichen-Limit ohne Stopp): **hier stoppen 2 von 3 Prompts
extrem früh** (10 bzw. 15 Zeichen) statt gar nicht — passt zur seit diesem
Lauf aktiven GEN-`</s>`-Dichte-Retrofit-Wirkung, kollabiert aber in die
Gegenrichtung (zu frühes Stoppen auf memorierte Kurz-Marker statt gar nicht
zu stoppen). Kein sauberer A/B-Test (n=1 Modell, kleine Stichprobe,
stochastisches Sampling ohne festen Seed — Prompt 1 lieferte bei zwei
Läufen zwei völlig unterschiedliche Ergebnisse). Reiner Zwischenstand;
der eigentliche Vergleich gegen die alte, voll austrainierte Laptop-
Prinzessin (`trained_256_e40.snai`) folgt nach Abschluss von Epoche 3.

## Nachtrag: `[Gekürzt]`-Marker im Epoche-1-Trainingsmaterial quantifiziert

Nachgezählt, woher das `[Gekürzt]`-Muster aus Prompt 2/3
stammt: der SFT-Sammler kappt zu lange SFT-Antwortwerte hart bei 1500
Zeichen und hängt wörtlich `"... [Gekürzt]"` an, direkt gefolgt vom
`</s>`-Stop-Token. In genau der Trainingsdatei, die Epoche 1 gefüttert hat
(`factory/checkpoints/chunk_princess_SFT_6a8050b1332a5_0.txt`, 9,7 MB,
80.514 Zeilen), tragen **503 von 842 `[SYMBIO]`-Antwort-Turns (≈60%)**
dieses Muster (`grep -c` verifiziert, nicht geschätzt). Erklärt plausibel
die auffällig frühen Stopps bei Prompt 2/3 — bei fast zwei Drittel aller
gesehenen Antworten war `[Gekürzt]</s>` das letzte Signal vor dem Stopp,
deutlich dominanter als jedes thematische Prompt-Signal.

**Bewusst zurückgestellt:** kein Fix jetzt, erst nach
dem finalen Epoche-3-Vergleichstest gegen `trained_256_e40.snai` angehen —
dieser Nachtrag dient nur der Absicherung, damit der Befund nicht verloren
geht.
