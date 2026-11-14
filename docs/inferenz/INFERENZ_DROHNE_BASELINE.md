# Inferenz-Baseline: Drohne/Expert (128 Dim) — erster Lauf unter neuem Ladder + tiefenskalierter Init

Wie am 2026-08-16 geplant: sobald die
erste Drohne/Expert unter dem korrigierten `EVOLUTION_LADDER` (Prinzessin 2→3,
Matriarchin 4→6, Empress 8→12 Layer) und der neuen tiefenskalierten Genesis-Init
(`init_base.php`, GPT-2-artige `w_o`/`w_down`-Skalierung) fertig trainiert ist,
sollte ein Referenz-Inferenztest als Baseline dokumentiert werden — analog zu den
bestehenden Prinzessin-/Matriarchin-Referenztests. Es existierte bisher **keine**
dokumentierte Drohnen-Inferenz-Baseline im Projekt; dies ist der erste Durchlauf.

## Testaufbau

- **Modell:** `factory/tributes/expert_SFT_fe86a817501_L07.2514.snai`
- **Warum dieses Modell:** von den 3 lokal verfügbaren SFT-Experten (alle drei
  seit dem dritten Reset/Neustart um 10:14 CEST fertig trainiert, alle Tribute
  für die laufende Prinzessin-Synthese) der mit dem niedrigsten (besten)
  Held-out-Loss: **7.404329** (Epoch 2, laut
  `factory/checkpoints/best_expert_SFT_fe86a817501.json`). Die anderen beiden:
  `5f16a817701` (7.469579), `4276a81714e` (7.506204). Der Dateiname-Suffix
  (`L07.2514`) ist der finale Trainings-Loss, nicht der Held-out-Loss — Ranking
  hier bewusst nach Held-out, konsistent mit den bisherigen Tier-Baselines.
- **Architektur:** 128 Dim, 2 Layer, 2 Heads (headSize=64) — Drohne/Expert-Tier
  laut `EVOLUTION_LADDER`. Die Layer-Zahl selbst blieb bei der Ladder-Korrektur
  vom 16.08. für dieses Tier unverändert (nur Prinzessin/Matriarchin/Empress
  wuchsen) — die tiefenskalierte Genesis-Init betrifft aber jede neu
  initialisierte Drohne direkt, dieses Modell ist die erste vollständig damit
  trainierte Instanz.
- **Perplexity zur Einordnung:** Loss 7.404329 bei Vocab=8192 → Perplexity ≈
  1643 (≈20,1% des Zufalls-Niveaus 8192). Deutlich schlechter (näher am
  Zufall) als die 256-Dim-SFT-Prinzessin-Baseline (PPL ≈710, ≈8,7%, siehe
  `INFERENZ_BASELINE.md`) — erwartbar für das kleinste, erste Tier der Leiter.
- **Inferenz-Parameter** (`snai_inference.php`, unverändert/Standard):
  Temperature 0.25, Repetition Penalty 1.15, max. 256 neue Tokens,
  `beam_width=1` (kein Beam Search).
- **Hardware/Pfad:** Hauptrechner, NVIDIA GeForce RTX 3050. Wie dokumentiert
  aktiviert `snai_inference.php` OpenCL unabhängig von der Modellgröße (kein
  Dim-Gate wie beim Training).
- **Datum:** 2026-08-16, ~14:20 CEST.

## Nebenfund während der Vorbereitung: fehlende `.heads`-Sidecar bei `merge_tributes()`-Output

Erster Versuch lief gegen `factory/checkpoints/proto_royal_6a817ad0dde54.snai`
(die per `merge_tributes()` aus den 3 Experten-Tributen gemittelte "Königin",
kurz vor dem Net2Net-Wachstum zur laufenden Prinzessin). Ergebnis: fast
reiner Kommata-/Satzzeichen-Spam ohne jede Wort-Struktur — deutlich
degenerierter als die drei unten dokumentierten Läufe. Ursache gefunden:
`merge_tributes()` (`royal_distillery.php:684`) schreibt für die gemergte
Ausgabedatei **keine** `.heads`-Sidecar, obwohl die Merge-Vorstufe
(`largest_head_compatible_group()`) bereits sicherstellt, dass alle 3
Quell-Tribute dieselbe Head-Zahl haben (hier: 2). `snai_inference.php` fiel
mangels Sidecar auf den Default (1 Head, headSize=128) zurück und hat das
sogar automatisch als neue (falsche) Sidecar-Datei festgeschrieben — der
Kommentar im Skript selbst warnt genau davor ("selbe Fehlerklasse wie der
Net2Net-Gain-Bug"). Falsch geschriebene Sidecar wieder gelöscht, bevor sie
irgendwo greifen konnte.

**Wichtig zur Einordnung:** Das ist **kein** Trainings-Pipeline-Bug — der
nachfolgende `upscale_cortex()`-Schritt liest die Head-Zahl nicht aus der
`proto_*`-Sidecar, sondern erneut direkt aus dem ersten Quell-Tribut
(`$selected_tributes[0]`, siehe `royal_distillery.php:274`). Die Prinzessin,
die daraus gerade wächst, ist also nicht betroffen. Es ist eine reine
Falle für manuelle/externe Inferenz gegen die kurzlebigen `proto_*.snai`-
Scratch-Dateien, die die Pipeline nach erfolgreicher Kalibrierung ohnehin
selbst löscht (`@unlink($proto_cortex)`). Nicht behoben (kein
`cortex.nim`/`libcortex.so`-Bezug, reine `royal_distillery.php`-Änderung
wäre nötig, aber `proto_*`-Dateien sind nie ein vorgesehenes Inferenz-Ziel)
— hier nur dokumentiert, damit niemand denselben Fehlschluss zieht. Der
unten dokumentierte Test läuft stattdessen korrekt gegen einen der
originalen Experten-Tribute (eigene, korrekte `.heads`-Sidecar vorhanden).

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“
Prompt verdaut in 12ms, 310 Zeichen generiert:
```
,,, und,.,.,,
,,,.,..,..,, der,,
.
,,
,,.
,,.
,.,
,,,.
,.
.
,
,,,,
 die die,,,


 und und,,.,.,.




,,,,.,,,.
. der
 und,

.,
,,
,..,.,,,. die.
.,, und, die,,..,,,

,.,,,.,.,.

,.,
 die.,..,,
,,
,..,
.,,.
,.
, der.,., und.,,,,
.,
 und,
 die,.,,,,,,,.,,, und,.
, die,,
.,,
,..
.,.,.,
.,
,,,.,,,

,.,
.,
.,.
,,,
```

### Prompt 2: „Was ist ein Computer?“
Prompt verdaut in 6ms, 334 Zeichen generiert:
```
,.. die und,
,, und,,,,,,,,
,,.,,,,,,,
.,.
,,,,,,.,,,,,
,,..,,,,,,.,
,

, der,,
,.,,,,, und,,.,.
,..,,,, die,,,.
.,,., die.,,,
,.,, die. und
,,
.,
 die,,. und
.,,,,.,,
,
,,.,,.
,
, die,,. die,,,,,.,
 der,..,,,,.,,.,
, die
,,...,.,

,,,,,,, und..., und,
,,,. die
,,,.
,, und,,,, die
, die,.,, und,, und,
.,.,,.., die,
. der.


,,, die
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“
Prompt verdaut in 8ms, 328 Zeichen generiert:
```
, die und, und,.,, die.., der,.,,,,,
.,
,..,,
,.,
,.,
,,,., und,,.,.,,,
, die,,.
,,. und,.

,,,
,,.,,,.,,.,
,
 die,
 der,. die,,., die..,.,,,,,,, und,.,,
. die,
,,,,,,,


,,..,,.,,
,,
,...,.,
,..., der,,,,,,,,
..,.,,,,,, und,.,,,,,
.,

., die.
,,.. die,
,....,.,,,,., und,,

,, der,, und,,.,
,

, die, und und
,...,,,
,
,.....,,
```

## Beobachtungen

- **Kein Kollaps in Wortstruktur — nur Funktionswort-/Satzzeichen-Fragmente.**
  Alle drei Läufe bleiben praktisch komplett im Muster
  `[Satzzeichen] [die|der|und] [Satzzeichen]...`, ohne erkennbare
  Substantive, Verben oder zusammenhängende Wortfolgen. Deutlich schwächer
  als die 256-Dim-Prinzessin-Baseline (`INFERENZ_BASELINE.md`), die
  wenigstens Wortfragmente mit Endungen (-ischen, -burg) und Jahreszahlen
  produzierte.
- **Passt zur Perplexity-Einordnung:** PPL ≈1643 (≈20% des Zufallsniveaus)
  ist deutlich schwächer als die Prinzessin-Baseline (≈8,7%) — für das
  kleinste Tier (128 Dim, 2 Layer) mit entsprechend geringer Kapazität
  erwartbar, kein Hinweis auf einen Defekt.
- **Keine inhaltliche Kohärenz zum Prompt**, wie bei allen bisherigen
  Tier-Baselines vor der Matriarchin-Stufe.
- **Kein vorzeitiger Stopp auf `</s>`** bei allen drei Läufen (volles
  Zeichen-Budget ausgeschöpft) — konsistent mit der generellen Beobachtung,
  dass kleinere/schwächere Modelle seltener ein sauberes EOS lernen.
- **Sehr schnelle Inferenz** (Prompt-Verdauung 6-12ms, Gesamtlaufzeit
  ~1,2-1,4s) — 128 Dim ist trivial klein für die RTX 3050.

## Bewertung der ursprünglichen Einschätzung („Satzzeichen nahezu korrekt, Groove erfasst“)

Der frühere informelle Eindruck (frühere
Drohnen-Inferenztests, nie dokumentiert, lange vor dem gesamten Fix-Stack)
lautete: das Modell setzt Satzzeichen nahezu korrekt und erfasst den
„Groove"/Sprachrhythmus der Sätze im weitesten Sinne. Dieser erste
dokumentierte, reproduzierbare Test **bestätigt das nicht** — im Gegenteil.

- **Die Zeichensetzung ist nicht kohärent.** Alle drei Läufe enthalten
  durchgängig Ketten von 3-8 Satzzeichen in Folge ohne dazwischenliegendes
  Wort, z.B. `,,,.,..,..,,` (Prompt 1), `,,.,,,,,,,` (Prompt 2, acht Kommas
  hintereinander), `,.....,,` (Prompt 3). Echte deutsche Zeichensetzung
  produziert das nie — Kommas/Punkte markieren Phrasen-/Satzgrenzen, nicht
  sich selbst. Zusätzlich Duplikate ohne Trenner (`die die`, `und und`).
  Kohärent wäre stattdessen etwas wie „Wort, Wort, Wort. Wort und Wort.“ —
  Satzzeichen an Grenzen, mit Inhalt dazwischen. Das liefert dieses Modell
  nicht; es kollabiert fast vollständig in eine Schleife aus `,`/`.`/„die"/
  „der"/„und", ohne dass diese eine erkennbare syntaktische Rolle einnehmen.
- **Mögliche technische Teilerklärung für den Alternations-Effekt:** die
  Repetition Penalty (1.15) bestraft zuletzt benutzte Tokens. Bei einem so
  stark auf 4-5 Tokens konzentrierten Wahrscheinlichkeitsraum (`,` `.` „die“
  „der“ „und“) zwingt das den Sampler dazu, ständig zwischen genau diesen
  wenigen Tokens zu wechseln, statt (wie vermutlich bei Greedy-Decoding ohne
  Penalty) in eine Endlosschleife eines einzelnen Tokens zu fallen. Die
  scheinbare „Abwechslung“ zwischen Komma und Punkt könnte also größtenteils
  ein Artefakt der Decoding-Parameter sein, nicht ein Zeichen gelernter
  Zeichensetzungs-Kompetenz. Nicht verifiziert — reine Hypothese, siehe
  offene Frage unten.
- **Einordnung (korrigiert, 2026-08-16):** der alte Test lief nicht nur auf
  Basis eines beiläufigen menschlichen Eindrucks — Durchführung UND Auswertung damals von einer Gemini-KI übernommen, mit
  denselben hier standardisierten Prompts. Das entkräftet den ursprünglich
  vermuteten Prompt-Unterschied als Erklärung. Es bleiben aber andere
  Lücken: Rohausgabe, exaktes Modell und Loss-Wert aus diesem alten Lauf
  sind nirgends im Projekt dokumentiert/erhalten — ein direkter
  Text-für-Text-Vergleich ist weiterhin nicht möglich. Zusätzlich zu
  berücksichtigen: KI-Bewertungen generierter Textqualität
  ("LLM-as-judge") gelten als tendenziell milde/mustersuchend, gerade bei
  kurzen, funktionswortlastigen Fragmenten — Geminis damalige positive
  Einschätzung ist damit nicht automatisch verlässlicher als ein
  menschlicher Eindruck gewesen wäre. Dieser Test bleibt der erste Befund
  mit erhaltener Rohausgabe für dieses Tier — und die zeigt keinen
  bestätigten Groove, sondern Funktionswort-/Satzzeichen-Kollaps.

## Offene Frage, später zu prüfen: echte Verschlechterung oder Wegfall von Memorierung?

Hypothese (2026-08-16): der alte informelle Eindruck könnte auf
Memorierung beruht haben — falls die damalige Drohne wiederholt über
denselben (kleineren) Korpus lief, hätte sie Satzzeichen-Muster stellenweise
auswendig gelernt statt generalisiert. Seit den Korpus-Erweiterungen vom
2026-08-14 (Literatur-Sammler, diversifizierte GEN-Quellen, laufend
rotierender Nektar) bekommt jede Epoche tendenziell neueres/anderes
Material zu sehen — echte Generalisierung wäre dann sichtbar schwächer,
ohne dass die tatsächliche Modellqualität gesunken ist. Nicht geprüft, nur
notiert — **später untersuchen**, sobald relevant, z.B. durch:
- Kontrolllauf mit Greedy-Decoding (temperature≈0, keine Repetition
  Penalty) gegen dasselbe Modell, um zu sehen, ob der Alternations-Effekt
  von der Penalty kommt oder vom Modell selbst.
- Vergleich, ob frühere (undokumentierte) Drohnen-Tests auf einem kleineren/
  stärker wiederholten Korpus-Ausschnitt liefen als der aktuelle rotierende
  Nektar-Mix.

**Einordnung der Hypothese (2026-08-16):** in sich
logisch konsistent — kleines Modell (128 Dim, 2 Layer) + damals kleinerer/
öfter wiederholter Korpus → Memorierung von Oberflächenmustern statt
Generalisierung → wirkt "besser", ohne wirklich etwas gelernt zu haben;
diversifizierter/rotierender Korpus seit 14.08. → weniger Wiederholung pro
Epoche → die Memorierungs-Krücke fällt weg → sichtbar schwächer, ohne dass
die tatsächliche Modellfähigkeit gesunken sein muss. Ein bekannter,
plausibler Effekt bei kleinen Modellen — aber drei Punkte relativieren, wie
stark man sich darauf verlassen sollte, bevor es geprüft ist:

1. **Konfundiert mit mehreren anderen gleichzeitigen Änderungen.** Zwischen
   dem alten undokumentierten Test und heute liegen nicht nur die
   Korpus-Diversifizierung, sondern auch Label-Smoothing, Weight-Decay/
   Dropout, tiefenskalierte Init, Wikitabellen-Cleanup, `</s>`-Retrofit.
   Besonders **Label-Smoothing ist ein eigenständiger, direkter Kandidat für
   genau dieses Symptom** — es glättet die Zielverteilung bewusst weg von
   scharfen One-Hot-Targets, was unabhängig von Memorierung zu genau so
   einem diffusen Pendeln zwischen den paar häufigsten Tokens führen kann.
   Die Memorierungs-Hypothese isoliert einen Faktor (Korpusvielfalt), obwohl
   mindestens zwei plausible Mechanismen gleichzeitig aktiv sind.
2. **Korrektur (2026-08-16):** der alte Test war keine bloße Anekdote im
   Sinne eines beiläufigen menschlichen Eindrucks — Durchführung und
   Auswertung liefen über eine Gemini-KI, mit denselben hier
   standardisierten Prompts. Das entkräftet den Prompt-Unterschied als
   Erklärung. Es bleibt aber: keine Rohausgabe, kein exaktes Modell, keine
   Loss-Zahl aus diesem alten Lauf im Projekt dokumentiert — ein
   Text-für-Text-Vergleich ist also weiterhin nicht möglich. Zusätzlich
   relativierend: KI-Bewertungen generierter Textqualität ("LLM-as-judge")
   gelten als tendenziell milde/mustersuchend, gerade bei kurzen,
   funktionswortlastigen Fragmenten — Geminis positives Urteil damals ist
   also nicht automatisch zuverlässiger als ein menschlicher Eindruck
   gewesen wäre. Ein Teil der wahrgenommenen Verschlechterung könnte
   trotzdem daher kommen, dass hier zum ersten Mal die Rohausgabe
   zeichengenau geprüft wurde, statt sich auf eine (KI-)Zusammenfassung zu
   verlassen.
3. **Das konkrete Fehlerbild passt eher zu „generell unterfitted“ als zu
   „generalisiert, aber nicht memoriert“.** Ein Modell, das nicht memoriert,
   aber sinnvoll generalisiert, würde man eher an lexikalischer Varianz
   erkennen (verschiedene, wenn auch falsch verwendete Wörter) — nicht an
   einer engen Schleife über 4-5 Tokens (`,`/`.`/„die“/„der“/„und“). Das
   sieht eher nach schlicht sehr hohem Loss/geringer Kapazität für die
   jetzige Aufgabenschwierigkeit aus (PPL 1643, nahe am Zufallsniveau) als
   nach „kein Auswendiglernen mehr, aber sonst fit“.

**Fazit:** nachvollziehbare, nicht unplausible Hypothese — aber eine von
mehreren gleichzeitig plausiblen Erklärungen, aktuell nicht von den anderen
(v.a. Label-Smoothing) trennbar ohne einen kontrollierten Vergleich. Als
offene, später zu prüfende Frage sinnvoll, nicht als wahrscheinlichste
Erklärung zu behandeln.

## Für den Vergleich mit späteren Tiers/Läufen

Dieselben drei Prompts, dasselbe Skript, dieselben Parameter (Temperature
0.25, Rep. Penalty 1.15, 256 Tokens, kein Beam Search) wie bei allen
bisherigen Tier-Baselines. Diese Drohnen-Baseline ist der unterste
Referenzpunkt der Leiter — spätere Drohnen-/Expert-Läufe (z.B. nach dem
geplanten `symbio_vocab.json`-Neubau) sollten direkt hiergegen verglichen
werden, um zu sehen, ob strukturelle Änderungen (Ladder-Verhältnis,
tiefenskalierte Init) sich schon auf dieser Stufe messbar auswirken, auch
wenn die Drohne selbst architektonisch unverändert blieb.
