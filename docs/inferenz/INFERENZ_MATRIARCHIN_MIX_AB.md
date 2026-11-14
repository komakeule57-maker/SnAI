# A/B-Inferenztest Matriarchin: neuer 50-%-SFT-Mix gegen alten Mix

Erster Test der Quotenerhöhung vom 23.08. (`royal_distillery.php`, SFT-Anteil
im Kalibrierungs-Nektar von 0.20 auf 0.50) auf Matriarchin-Ebene. Anlass war
die Diagnose vom 22./23.08., dass der Korpusmix — nicht die Modellkapazität —
die messbare Ursache dafür ist, dass Modelle aller Stufen am Prompt vorbei
schreiben.

`matriarchin_GEN_f7cfff` ist das erste Modell, das **durchgehend** unter dem
neuen Mix kalibriert wurde (gemessen: 42 % SFT-Blöcke / 30 % SFT-Bytes im
Kalibrierungs-Pool gegen ~11 % vorher). Es wurde am 26.08. um 06:03 nach sechs
Epochen fertig.

Testdatum: 2026-08-26. Harness: `experiments/matriarchin_mix_ab/run_ab.sh`,
Rohdaten und Kennzahlen im zugehörigen `results/`.

## Testaufbau

- **`f7cfff` E6** — neuer Mix. `secured_models/matriarchin_GEN_f7cfff_epoch6.apex`,
  sha256 `3b05a752e70b3bba…`, byte-identisch zur `factory/apex/`-Datei
  verifiziert.
- **`1fdf68` E3** — Hauptreferenz. Gleiche Stufe, gleicher Gradient-Clipping-Fix,
  alter Mix. Letzte gesicherte Epoche dieses Laufs (nach E3
  gestoppt). sha256 `03e317f401bb69f9…`.
- **`1557a2` E3** — Zusatzreferenz aus der Bug-Linie (vor dem Clipping-Fix).
  Ausdrücklich **nicht** Teil des eigentlichen Kontrasts: hier springen zwei
  Variablen zugleich. Läuft nur mit, um die Zahlen an
  `INFERENZ_MATRIARCHIN_1FDF68_GRADCLIP_AB.md` anzubinden.
  sha256 `9402bf57ea25d882…`.
- **Alle drei:** 512 Dim, 6 Layers, 8 Heads (`.heads` vor dem Lauf gegengeprüft),
  GEN-Kalibrierung.
- **Inferenz-Parameter:** Projektstandards aus `snai_inference.php`
  (Temperature 0.25 bzw. 0.90, Repetition Penalty 1.15 / Window 64, min_p 0.05,
  max. 256 neue Tokens, kein Beam-Search). Die einzigen Patches sind Temperature
  und ein additives EOS-Flag in der Metrikzeile — ohne das lässt sich ein
  sauberer `</s>`-Stopp nicht von „Token-Budget aufgebraucht" unterscheiden.
- **Stichprobe: 60 Läufe je Checkpoint bei T=0.25** (3 Standard-Prompts × n=20)
  plus 10 bei T=0.90 (2 Prompts × n=5). Deutlich mehr als die n=8 des
  Gradient-Clipping-A/B: eine Matriarchin-Inferenz kostet nur ~5 s, und die Leitmetrik
  entscheidet sich an Differenzen weniger Treffer.
- **Metriken identisch** zu `experiments/princess_*/compare_inference.sh`, damit
  die Zahlen quer zu den Prinzessinnen-Reihen gelesen werden können. Eine
  Ergänzung: `dashrun`, siehe unten.

## Methodischer Nachtrag: das Held-out ist vergleichbar

In früheren Berichten war behauptet worden, Held-out-Werte
verschiedener Läufe ließen sich nicht gegeneinander lesen. Das ist falsch und
wird hiermit korrigiert.

`train_cluster.php` ruft `eval_held_out.php` immer gegen `./factory/holdout/`
auf. Dort liegt eine **einzige Datei**, `hf_GEN_c0_f92d6f.txt` (2,0 MB,
unverändert seit dem 05.08.). Alle Matriarchinnen aller Linien wurden also
gegen dieselbe Stichprobe bewertet. Dasselbe gilt für die beiden
Prinzessinnen-Experimente: `princess_chinchilla` und `princess_sftmix` haben
byte-identische Holdouts (beide sha256 `e95360dea3afb3b2…`).

**Aber:** das Holdout ist reines GEN — null `[USER]`-Marker auf 2,0 MB,
nachgezählt. Es misst ausschließlich unbedingtes Textmodellieren und kann per
Konstruktion nicht sehen, ob ein Modell auf einen Prompt eingeht. Ein Modell
mit 30 % Dialoganteil im Training *muss* dort schlechter abschneiden, ohne dass
das etwas über die eigentliche Frage aussagt. Der Loss läuft hier deshalb nur
nachrichtlich mit; Leitmetrik ist `topic_lift`.

Für diesen Test wurden alle drei Checkpoints frisch nachgemessen, mit gleichem
Budget von 2047 Token statt der geloggten 999. 2047 ist nicht frei gewählt,
sondern die Obergrenze — ein Probelauf mit 4000 lieferte `HELDOUT_TOKENS=2046`,
der Eval-Pfad ist durchs Kontextfenster gedeckelt.

| Checkpoint | Held-out geloggt (999 Tok) | frisch (2046 Tok) | PPL |
|---|---|---|---|
| `1557a2` E3 (Bug) | 4.651470 | **4.417360** | 82,88 |
| `1fdf68` E3 (Fix, alter Mix) | 5.072072 | **4.739318** | 114,36 |
| `f7cfff` E6 (Fix, neuer Mix) | 5.188638 | **4.842497** | 126,79 |

Die Reihenfolge bleibt bei verdoppeltem Budget erhalten — die geloggten Werte
sind also nicht bloß Rauschen. `f7cfff` liegt trotz **doppelter Epochenzahl**
hinter `1fdf68`.

## Messergebnis (je 60 Läufe bei T=0.25)

| | `f7cfff` E6 (neu) | `1fdf68` E3 (alt) | `1557a2` E3 (Bug) |
|---|---|---|---|
| Held-out (2046 Tok) | 4.842497 | 4.739318 | **4.417360** |
| **topic_own** | **1 / 60** | 9 / 60 | 1 / 60 |
| **topic_foreign** | 10 / 60 | 18 / 60 | 2 / 60 |
| **topic_lift** | **0,29** | 1,03 | 1,20 |
| `eos_ok` | 43 / 60 | 37 / 60 | **53 / 60** |
| `<unk>` gesamt | 5 | 1 | 2 |
| `[SYMBIO]`-Leck | 2 | 1 | 2 |
| `dashrun` | **13** | 5 | 2 |
| `register_lock` | 13 / 60 | 4 / 60 | 4 / 60 |
| `distinct_run_starts` | 39 | 47 | 42 |
| Ø Zeichen | 149,0 | 142,3 | 257,5 |

Bei T=0.90 (je 10 Läufe): `f7cfff` topic_lift 0,29, `1fdf68` 2,00,
`1557a2` 0,29. Bei n=10 sind das Einzeltreffer — nicht überinterpretieren.

## Leitmetrik: keine der drei Matriarchinnen ist prompt-konditioniert

Nullniveau von `topic_lift` ist 1,0 (zwei fremde Stichwortsätze stehen gegen
einen eigenen). `1fdf68` liegt mit **1,03** exakt darauf, `1557a2` mit **1,20**
praktisch ebenso. Beide sind vom Zufall nicht zu unterscheiden.

`f7cfff` liegt mit **0,29 unter** dem Nullniveau: ein einziger Treffer im
eigenen Thema gegen zehn in fremden. Das ist keine schwächere Konditionierung,
sondern gar keine — plus ein Nebeneffekt, der weiter unten erklärt wird.

**Die Quotenerhöhung hat auf Matriarchin-Ebene nicht geliefert, was sie
sollte.** Sie hat die Zielgröße nicht verbessert, sondern nach dieser Messung
verschlechtert.

Zur Einordnung gegen die Prinzessinnen-Reihe: der Chinchilla-Arm erreichte bei
E10 einen `topic_lift` von 8,29, der SFT-Mix-Arm bei E3 einen `topic_own` von
43 %. Beide Prinzessinnen (256 Dim) schlagen damit alle drei Matriarchinnen
(512 Dim) deutlich. Das spricht dafür, dass hier nicht die Kapazität
limitiert, sondern die Trainingsdauer pro Stufe: die Matriarchinnen sahen
6 bzw. 3 Epochen, die Prinzessinnen 10 bzw. 13.

## Was der neue Mix bewirkt hat: das Register

Der Mix ist nicht wirkungslos geblieben — er hat nur an anderer Stelle gewirkt
als beabsichtigt. Gezählt wurde, wie viele der 60 Läufe mit dem Chat-Öffner
`" Natür…"` beginnen:

| Checkpoint | `" Natür…"`-Öffner |
|---|---|
| `f7cfff` E6 (neuer Mix) | **29 / 60** |
| `1fdf68` E3 (alter Mix) | 16 / 60 |
| `1557a2` E3 (Bug) | **0 / 60** |

`f7cfff` hat die **Form** einer Assistenzantwort gelernt und greift bei fast
jedem zweiten Lauf darauf zu. Nur hat es nichts, womit es die Form füllen
könnte. Jeder Arm hat dabei seinen eigenen Attraktor, und jeder spiegelt seinen
Korpusmix:

- `f7cfff` → SFT-Floskel („Die Analyse der Herausforderung…", „Die optimale
  Reihenfolge der Themen…")
- `1fdf68` → Wikipedia-Register (Gemeinde, Landkreis, Ortsteil)
- `1557a2` → LaTeX-Mathematik (`$a = -b^2$`) — dasselbe Muster, das schon in
  `INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md` als offener Korpusbefund notiert
  wurde

Das erklärt auch den `topic_lift` unter Nullniveau: `f7cfff` produziert
generischen Instruct-Antworttext, und dessen Vokabular („Daten", „Analyse",
„Programm", „Ressourcen") überschneidet sich stark mit dem Computer-Stichwortsatz.
Es trifft fremde Themen also nicht zufällig, sondern systematisch — bei jedem
Prompt dasselbe.

## Qualitative Beispiele

Alle auf `Was ist ein Computer?`, T=0.25.

**`f7cfff` E6 (neuer Mix)** — Chat-Öffner, dann kurze themenfreie Sätze, in
einem Fall Strich- und Code-Kollaps:

> ` Natürlich,`
> `-/-------------------------------------------------------------------`
> `                     <unk>.cmp();`

> ` Natür, die du deine Mischung und das Farbe, die sie nicht zu den beiden anderenen.`

> ` Natür, die du die Blätter von der Erinnerung des Benutzerns zu behalten.`

**`1fdf68` E3 (alter Mix)** — vollständiges, grammatisch sauberes
Wikipedia-Deutsch, das den Prompt komplett ignoriert:

> „Der Name folgen: / Also ist eine Fläche von der Stadt, die in der Region der
> Stadt und der Gemeindegebiet des Landesamtes. / Die Liste der Gemeinde liegt
> im Landkreis Wölzberg (Schmale). / Die Gemeinde ist ein Ortsteil der Gemeinde
> (Hrsg.)…"

**`1557a2` E3 (Bug-Linie)** — kippt in LaTeX-Mathematik:

> „Ich kann, dass die erste ganze Zahl der Möglichkeiten zu lösen. Wenn ich das
> bedeutet, dass $a = -a$ und $a + c = - b^2 = -b^3$. Ich weiß, dass
> $a = -b^2 = -b^2 = -b^2 = -b^2 + z^2 = …"

## Die Strichketten: Herkunft geklärt, kein Wiki-Leck

Ein Probelauf hatte eine reine Bindestrichkette bis ans Token-Limit erzeugt.
Erster Verdacht war eine Wiederholung des Wikitabellen-Lecks vom 15.08. Der
Verdacht ist **widerlegt**. Messung über je 12 Dateien pro Klasse, einzeln
durchgestreamt:

| Klasse | Umfang | Ketten ≥10 Striche | ≥3 Striche | MD-Tabellenzeilen |
|---|---|---|---|---|
| GEN neu (≥24.08.) | 20,3 MB | **0** (0,00/MB) | 6 (0,30/MB) | 0 |
| GEN alt (<20.08.) | 29,1 MB | **0** (0,00/MB) | 244 (8,39/MB) | 0 |
| **SFT alt (<20.08.)** | 18,8 MB | **212 (11,30/MB)** | 425 (22,65/MB) | **373** |
| SFT neu (≥24.08.) | 5,1 MB | 4 (0,78/MB) | 13 (2,54/MB) | 1 |
| SFT `def` | 7,4 MB | 0 | 0 | 0 |
| SFT `gem` | 2,0 MB | 0 | 0 | 0 |
| `input_long` | 2,0 MB | 11 (5,44/MB) | 43 (21,25/MB) | 27 |

Im GEN-Material — Wikipedia, Bücher, Untertitel — gibt es **keine einzige**
lange Strichkette. Die Last liegt vollständig bei SFT, und die Trefferzeilen
sagen auch warum:

```
hf_SFT_0a08f888_c0.txt:10276   |-----------------|----------|-----------|
hf_SFT_0e283f32_c1.txt:143     |---------|-------------|
hf_SFT_0923ad27_c0.txt:11889   +-----------------
hf_SFT_0923ad27_c1.txt:2799    [USER] Welche Excel-Formel kann ich verwenden … Hier ist eine Beispiel-Excel-Tabelle
```

Markdown- und ASCII-Tabellentrenner aus den Instruct-Datensätzen. Die
Antwortseite ist beim Loss-Masking nicht maskiert, das Modell lernt sie also
aktiv.

**Es ist keine neue Kontamination, sondern eine Dosiserhöhung.** Alle
belasteten Dateien tragen Datum vor dem 20.08., lagen also schon im Korpus,
als `1fdf68` trainierte. Die neuen Quellen sind im Gegenteil sauber: `def` und
`gem` bei null, die avemio-Sammlung bei 0,78/MB statt 11,30. Was sich geändert
hat, ist der Anteil dieses Materials am Pool — von ~11 % auf 30 % der Bytes,
also das 2,7-fache. Die gemessenen `dashrun`-Werte (13 / 5 / 2) passen zu
dieser Dosis.

**Zum Mechanismus:** das Vokabular enthält von 8192 Einträgen **keinen
einzigen** mit zwei aufeinanderfolgenden Strichen. Eine 263-Zeichen-Kette ist
also keine Handvoll Tokens, sondern rund 200 Einzelentscheidungen, bei denen
der Strich jedes Mal Argmax blieb. Das ist ein klassischer
Wiederholungs-Attraktor mit `-` statt eines Wortes; die Repetition Penalty von
1.15 über 64 Token bricht ihn nachweislich nicht. `dashrun` wurde für diese
Reihe neu eingeführt, weil der Kollaps weder `<unk>`-Spam noch
Wortwiederholung ist und damit durch alle bisherigen Detektoren fällt.

**Randbefund:** das alte GEN-Material hat 8,39 Treffer/MB bei ≥3 Strichen
gegen 0,30 im neuen — Faktor 28. Beim Nachsehen sind das vereinzelte
`---`-Trennzeilen aus Büchern sowie Fehltreffer der Regex an Trennstrichen in
langen Zeilen; lange Ketten sind nicht dabei. Die betroffenen Dateien sind
überwiegend das archaische Material von vor dem Archaik-Filter (16.08.).

## Was der Test NICHT zeigt

Zwischen `f7cfff` und `1fdf68` unterscheidet sich nicht nur der Mix:

1. **Epochenzahl 6 gegen 3.** `1fdf68` wurde nach E3
   gestoppt. Ein epochengleicher Vergleich ist nachträglich nicht mehr
   herstellbar: der Steuer-Daemon hält nur `best_<session>.snai`, und das ist bei
   `f7cfff` bereits E6 — die Zwischenstände sind überschrieben.
2. **Korpusstand.** Zwischen den Läufen liegen ~5 Tage Scout-Nachschub plus
   drei neue SFT-Quellen (Definitionsfragen, umgebauter
   Gemma-Sammler, avemio/ShareGPT).
3. **Elternmodelle.** `f7cfff` stammt von Prinzessinnen, die selbst schon unter
   dem neuen Mix liefen.

Gemessen wird also „ganze Pipeline neu gegen ganze Pipeline alt", nicht der
Mix isoliert — dieselbe Einschränkung wie beim Gradient-Clipping-A/B. Ein sauberer Test
bräuchte zwei Läufe, die sich ausschließlich in der Quote unterscheiden.
Der laufende Kontrollversuch auf Prinzessin-Ebene
(`experiments/princess_sftmix` gegen `princess_chinchilla`) ist genau das und
bleibt die belastbarere Evidenzquelle.

Ebenfalls offen: alle drei Arme liegen beim `topic_lift` am oder unter dem
Nullniveau. Ein Test, bei dem **keine** Bedingung das Signal zeigt, kann
zwischen den Bedingungen nicht sauber trennen — der Unterschied 0,29 gegen
1,03 ist ein Unterschied zwischen zwei Nullen. Die Aussage „der neue Mix hat
nicht geliefert" ist gedeckt; die Aussage „der neue Mix ist schlechter" nur
schwach.

## Einordnung

Der Test war als Beleg für die Mix-These gedacht und ist gegen sie
ausgefallen. Auf Prinzessin-Ebene war der SFT-Anteil messbar die Ursache
fehlender Prompt-Bindung; auf Matriarchin-Ebene reproduziert sich das nicht —
das Modell übernimmt das Register, aber nicht die Konditionierung.

Die naheliegendste Erklärung ist nicht Kapazität, sondern **Trainingsdauer pro
Stufe**: die Prinzessinnen entwickelten ihre Konditionierung erst zwischen E3
(SFT-Arm) und E10 (Chinchilla-Arm), die Matriarchinnen haben 6 bzw. 3 Epochen
gesehen. Dass zwei kleinere Modelle die größeren auf der Zielgröße schlagen,
passt dazu und spricht gegen eine reine Kapazitätsdeutung.

Konkret offen:

- **Mehr Epochen auf Matriarchin-Ebene.** Die Held-out-Kurve von `f7cfff` lief
  bis E6 monoton (5.698 → 5.188), der letzte Schritt betrug aber nur noch
  −0,020. Ob weitere Epochen die Konditionierung nachziehen, ist der direkte
  Test dieser Erklärung.
- **Tabellentrenner aus dem SFT-Korpus filtern.** Betrifft nur Material von
  vor dem 20.08.; die neueren Sammler sind sauber. Ein Retrofit analog zu
  `retrofit_wikitable_leak.php` wäre der bekannte Weg.
- **Repetition Penalty gegen den Strich-Attraktor gegentesten.** Reine
  Inferenzeinstellung, kein Korpus- oder Engine-Eingriff.
- **`[SYMBIO]`-Lecks** bleiben bei allen drei Armen im einstelligen Bereich
  (2 / 1 / 2 auf 60 Läufe) — unauffällig, keine Handlung nötig.
