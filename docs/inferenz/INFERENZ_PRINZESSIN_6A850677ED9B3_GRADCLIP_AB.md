# Inferenz-Test: Prinzessin `princess_GEN_6a850677ed9b3` — beabsichtigt der konfundfreie GEN-vs-GEN-Vergleich, tatsächlich neu konfundiert

Gedacht als Fortsetzung von `INFERENZ_PRINZESSIN_6A849F2626793_GRADCLIP_AB.md`
(dort SFT-vs-GEN-Konfund dokumentiert) mit einer **GEN**-Prinzessin unter dem
Fix, direkt vergleichbar mit den beiden GEN-Bug-Referenzen
(`6a81f9adb25f8`, `6a826128b6548`). Stattdessen liefert dieser Test einen
anderen, schwereren Konfund: die Kalibrierung dieser Prinzessin lief
teilweise während des Zeitfensters, in dem die 154 fehlformatierten
`hf_SFT_*.txt`-Dateien noch
nicht quarantäniert waren.

## Testaufbau

- **Modell:** `factory/royal/princess_GEN_6a850677ed9b3.cortex`
- **Architektur:** 256 Dim, 3 Layers, 4 Heads (per Inferenz-Banner
  bestätigt).
- **Training:** automatischer Schwarm-Pfad, GEN-Zweig, Frisch-Nektar-pro-
  Epoche (jede Epoche zieht laut `royal_distillery.php` einen neuen,
  zufälligen 5-Dateien-Satz — 80% aus dem GEN-Pool, 20%-Quote aus dem
  `hf_SFT_`-Pool, unabhängig vom Tier-Tag 'GEN'/'SFT'). Epoche 1 fertig
  05:44 Uhr, Epoche 2 08:14 Uhr, Epoche 3 10:42 Uhr (alle 2026-08-19).
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.947805 (PPL 382.9) →
  5.782577 (PPL 324.6) → **5.762476 (PPL 318.1)**, finaler Best-Checkpoint.
- **GradNorm über den kompletten Trainingslauf** (1800 geloggte Schritte,
  alle 3 Epochen): Minimum 0.6933, Maximum 6.3145, Mittelwert 1.406 —
  weiterhin im vom Audit vorhergesagten Bereich, etwas höhere Varianz als
  bei der SFT-Prinzessin (Mittel 1.22), aber nichts in der Nähe der
  Vor-Fix-Werte (Minimum 254.8, Median 1303.7).
- **Inferenz-Parameter:** identisch zu allen bisherigen Referenztests
  (Temperature 0.25, Rep. Penalty 1.15/Fenster 64, min_p 0.05, 256 Tokens,
  kein Beam Search).
- **Datum:** 2026-08-19, ca. 10:43 CEST.

## Der neue Konfund: Kalibrierungs-Zeitfenster überschneidet sich mit der Quarantäne

Zeitleiste (rekonstruiert aus Prozess-Startzeiten und Konversationsverlauf):

- **~03:27 Uhr:** Kalibrierung dieser Prinzessin beginnt (Epoche 1). Zu
  diesem Zeitpunkt liegen die 154 fehlformatierten `hf_SFT_*.txt` (bare
  `[SYMBIO]`, kein `[USER]`-Tag) noch regulär in `factory/input/` — die
  20%-SFT-Quote der Epoche-1-Kalibrierung (1 von 5 Dateien) hatte also eine
  reale Chance (~29%, 154 von 538 damaligen `hf_SFT_`-Dateien), eine davon
  zu ziehen.
- **~04:49–04:52 Uhr:** Quarantäne durchgeführt, alle 154
  Dateien nach `factory/quarantine_sft_bare_symbio_20260819/` verschoben.
  Epoche 1 lief zu diesem Zeitpunkt noch (endete 05:44 Uhr).
- **05:44 Uhr / 08:14 Uhr:** Epoche 2 und 3 beginnen — zu beiden
  Zeitpunkten ist der `hf_SFT_`-Pool bereits bereinigt, ihre Kalibrierungs-
  Dateien sollten also sauber sein.
- **Nicht mehr direkt verifizierbar:** die konkreten Kalibrierungsdateien
  jeder Epoche (`factory/checkpoints/calib_royal_6a85067809f78_e{1,2,3}.txt`)
  wurden nach Gebrauch bereits durch den nächsten Digest-Zyklus überschrieben
  — welche Datei(en) Epoche 1 tatsächlich zog, ist nicht mehr nachweisbar,
  nur das Zeitfenster/die Wahrscheinlichkeit.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (853 Zeichen, kein sauberer Stopp)
```
 Ja, dass es eine sie.
[SYMBIO] Ich ist ich nicht?
[SYMBIO] Ja.
[SYMBIO] Sie Sie Sie, um die Zeit, ich ich zu den Umschen.

[SYMBIO] Ich Sie ich ich, das ich, dass ich.
[SYMBIO] Ja.
[SYMBIO] Ja, dass es, aber.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Sicher!
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Ja Sie ich?
[SYMBIO] Ja, die sie.
[SYMBIO] Ja.
[SYMBIO] Ja, ich nicht.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Sie ist er.
[SYMBIO] Ja.
[SYMBIO] Ja, dass ich.
[SYMBIO] Ja.
[SYMBIO] Sicher.
- Ich wir ich, dass es.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Ja Sie ich, die sie.
[SYMBIO] Sie Sie Sie.
[SYMBIO] Ja.
[SYMBIO] Ja, dass ich?
[SYMBIO] Sie ist er.
[SYMBIO] Es:
[SYMBIO] Ja, dass die Ansicht.
- Ich Sie.
[SYMBIO] Sicher.
[SYMBIO] Sie wir.
[SYMBIO] Ja.
[SYMBIO] Sie ist es, dass ich, das sie.
[SYMBIO] Ja.
[SYMBIO] Ja Sie.
[SYMBIO] Sie kann ich.
```
**40× `[SYMBIO]` in 853 Zeichen** — im Schnitt alle ~21 Zeichen ein neuer Tag.

### Prompt 2: „Was ist ein Computer?“ (721 Zeichen, kein sauberer Stopp)
```
 Ich Sie ich.
[SYMBIO] Ja.
[SYMBIO] Ich Sie es, dass die Möglichkeit, um das ich.
[SYMBIO] Ja.
[SYMBIO] Sie Sie ich nicht, wenn sie ich.
[SYMBIO] Ja.
[SYMBIO] Ich Sie wir.
[SYMBIO] Ja.
[SYMBIO] Sicher!
[SYMBIO] Ja Sie ich mir.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Ja.
-ch ich ich?
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] 1.00.5.
[SYMBIO] 1.
[SYMBIO] S..
[SYMBIO] Ja, die sie.
[SYMBIO] Ja.
[SYMBIO] Natie.
[SYMBIO] S..
[SYMBIO] Mein, dass es.
[SYMBIO] S..
[SYMBIO] Hüf.
[SYMBIO] C..
[SYMBIO] S..
[SYMBIO] ISBN 3-8-3.000.
[SYMBIO] 1.
[SYMBIO] 1.
[SYMBIO] Puy:
[SYMBIO] S..
[SYMBIO] D...
1.
[SYMBIO] F.
2.
- Ich,
Das Sie ich.
[SYMBIO] B..
[SYMBIO] 1.
[SYMBIO] Lir!
3.
[SYMBIO] T..
[SYMBIO] R..
[SYMBIO] S..
[SYMBIO]: 2.
[SYMBIO] K.
```
**41× `[SYMBIO]`**, dazu ein neues Artefakt: „ISBN 3-8-3.000." — ein
Buch-Metadaten-Fragment, passt zum GEN-Korpusanteil (PleIAs/German-PD).

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (641 Zeichen, kein sauberer Stopp)
```
 Ja, die sie das ersten Beispiel.
[SYMBIO] Sichere Sie ich.
[SYMBIO] Ja.
[SYMBIO] Ja.
[SYMBIO] Natürlich, dass es.
[SYMBIO] 1.
[SYMBIO] S..
[SYMBIO] Ja.
[SYMBIO] F.
[SYMBIO] B.
[SYMBIO] Rein.
[SYMBIO] H..

[SYMBIO] L.:
[SYMBIO] M..
[SYMBIO] Sirst....
[SYMBIO] 1.
[SYMBIO] H..
[SYMBIO] Pat,
2.
[SYMBIO] Acha,
[SYMBIO] Cuh..
[SYMBIO] B..
[SYMBIO] H..
[SYMBIO] S...
[SYMBIO] 1.
[SYMBIO] El.
[SYMBIO] H...
[SYMBIO] H.:
[SYMBIO] W...
- Sie ist die,
[SYMBIO] W..

Das ist ein.
[SYMBIO] F.

[SYMBIO] T..

[SYMBIO] Puy.
[SYMBIO] S..

2.

[SYMBIO] Ja.

1.

3.


[SYMBIO] D.:

- Ich.

Ich.
[SYMBIO] H...

Die Sie ist ich,
[SYMBIO] Sirh.

[SYMBIO] S..
```
**37× `[SYMBIO]`.** Ein erster Sample-Durchlauf dieses Prompts (verworfen,
da Temperature-Sampling nicht deterministisch ist) stoppte nach nur 75
Zeichen mit 2× `[SYMBIO]` — ein zweiter Durchlauf (oben dokumentiert) zeigt
dasselbe schwere Muster wie P1/P2. Der kurze Lauf war also ein Ausreißer,
nicht repräsentativ.

## Beobachtungen

- **Deutlich schwereres `[SYMBIO]`-Leck als bei der SFT-Prinzessin** (dort
  4× über 2 von 3 Prompts) — hier **37–41× pro Prompt**, im Schnitt alle
  ~18–21 Zeichen ein neues `[SYMBIO]`-Tag. Das Modell hat den Marker
  offenbar nicht als Dialog-Rollenwechsel gelernt, sondern als generischen
  Kurzsatz-/Aufzählungstrenner.
- **Kein `<unk>`-Spam, keine erfundenen Bindestrich-Komposita** (anders als
  bei den beiden vorherigen Prinzessin-Referenzen) — stattdessen kurze,
  grammatisch fast korrekte, aber inhaltsleere Ein-/Zweiwort-Fragmente
  („Ja.", „Sicher!", „Sie Sie Sie."), listenartig mit `[SYMBIO]`
  durchsetzt, teils nummeriert (1./2./3.) wie eine zerfallene Aufzählung.
- **Neues Artefakt:** „ISBN 3-8-3.000." (Prompt 2) — Buch-Metadaten-Leck,
  passt zum PleIAs/German-PD-Buchkorpus-Anteil.
- **Kein sauberer `</s>`-Stopp bei keinem der drei (repräsentativen)
  Prompts.**

## Vergleich mit den Bug-Referenzen vom 18.08.

| | `6a81f9adb25f8` (Bug) | `6a826128b6548` (Bug) | **`6a850677ed9b3` (gefixt, teil-kontaminiert)** |
|---|---|---|---|
| Held-out-Loss (final) | 5.817457 | 5.600596 | **5.762476** (zwischen beiden) |
| GradNorm | nicht geloggt | nicht geloggt | **0.69–6.31, Mittel 1.41** |
| Sauberer `</s>`-Stopp | ja, alle 3 Prompts | nein | nein |
| Kollapsmuster | kurze Funktionswort-Fragmente | schwerer `<unk>`-Spam | **schweres `[SYMBIO]`-Tag-Spam** (neues drittes Muster) |

## Einordnung

**Dieser Test ist NICHT der beabsichtigte konfundfreie GEN-vs-GEN-Vergleich
geworden.** Er ist wahrscheinlich stärker konfundiert als der vorherige
SFT-Test: Epoche 1 (von drei) lief mit einer realen Chance, aus dem noch
nicht bereinigten `hf_SFT_`-Pool zu ziehen; Epochen 2–3 liefen nachweislich
auf bereinigtem Material, konnten das gelernte Verhalten aber nicht
korrigieren (LR sinkt über die Epochen von 0.0001 auf 0.00001 — die
correctve Kraft der sauberen Epochen 2/3 ist damit strukturell schwächer
als die von Epoche 1 gesetzte Prägung). Da die konkreten
Kalibrierungsdateien bereits überschrieben sind, ist nicht mehr beweisbar,
ob und welche kontaminierte Datei tatsächlich gezogen wurde — nur das
Zeitfenster und die Wahrscheinlichkeit (~29% pro Epoche mit SFT-Quote)
sprechen dafür.

**Auffällig:** selbst die SFT-Prinzessin (100% sauberer Zeitraum, nur 4×
Leck) zeigte bereits ein `[SYMBIO]`-Leck. Das spricht dafür, dass ein Teil
des Effekts unabhängig von der Kontamination ist — vermutlich reicht schon
die reguläre 20%-SFT-Quote (samt kurzer, oft sehr kompakter Turns aus
manchen HF-Quellen) aus, um `[SYMBIO]` bei diesem winzigen Kalibrierungs-
Budget (5 Dateien/Epoche) als "billigen", hochfrequenten Trenner gelernt
zu bekommen. Die Kontamination hätte diesen ohnehin vorhandenen Effekt in
diesem Fall nur drastisch verstärkt, nicht allein verursacht.

**Für den ursprünglich beabsichtigten Vergleich (gefixt vs. Bug, ohne
SFT/GEN- oder Kontaminations-Konfund) wird ein weiterer GEN-
Prinzessin-Lauf gebraucht, der komplett nach der Quarantäne (nach ca.
04:52 Uhr) begonnen hat.** Der aktuell laufende Reset-Fortschritt der
Pipeline (nächste Prinzessin nach dieser) wäre dafür der nächste
Kandidat.
