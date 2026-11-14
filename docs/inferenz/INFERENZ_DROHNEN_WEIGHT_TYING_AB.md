# A/B-Test Drohnen-Tier: Weight Tying (7 Layer) gegen ungebunden (2 Layer)

Messung 2026-08-28/29. Zwei Fragen, streng getrennt zu halten:

1. **Bringt Weight Tying etwas?** — der eigentliche A/B, beide Arme heute,
   gleiche Bedingungen, isolierte Variable.
2. **Wo steht das Drohnen-Tier inzwischen überhaupt?** — Vergleich gegen die
   beiden dokumentierten 128-Dim-Inferenzen vom 16.08. und 18.08. Nicht
   isoliert, nicht 1:1 vergleichbar, aber der eindeutigere Befund.

**Vorbemerkung zur Einordnung (29.08.):** die Drohne ist ein
128-Dim-Modell und **gar nicht zur Inferenz vorgesehen**. Sie ist die unterste
Stufe der Leiter, ihr Zweck ist, als Tribut in die Prinzessin einzugehen. Ein
Inferenzergebnis auf dieser Stufe ist für sich genommen nicht aussagekräftig
und darf nicht als Qualitätsurteil über das System gelesen werden. Es ist ein
Indikator — und als solcher hier ausgewertet.

## Testaufbau

- **Arme** (Herleitung: `docs/STUDIE_WEIGHT_TYING.md`):

  | | Geometrie | Parameter |
  |---|---|---|
  | ungebunden | 128 Dim / 512 FFN / **2 Layer** / Ctx 512 / 2 Heads | 2.491.008 |
  | gebunden | 128 Dim / 512 FFN / **7 Layer** / Ctx 512 / 2 Heads | **2.426.752** |

  Der gebundene Arm ist **2,6 % kleiner**. Das ist Absicht: gewinnt er, ist der
  Befund konservativ. Bei der Drohne sind 84,2 % aller Parameter die beiden
  `vocab*dim`-Tabellen — geteilt passen 7 statt 2 Layer in dieselbe Größe.

- **Material:** 4 Epochen à ~8,8 MB, **frisch je Epoche** (kein Wiederholen,
  sonst misst man Memorierung), reines GEN. Beide Arme sehen bitgleiche
  Dateien, Prüfsummen in `experiments/drone_tying_ab/data/CHECKSUMS.txt`.
- **Holdout:** byteidentisch zu dem der beiden Prinzessinnen-Arme
  (524.288 B, md5 gegengeprüft), damit die Zahlen in dieselbe Reihe passen.
- **Hyperparameter:** LR 0,0005, WD 0,01, Dropout 0,1, Label-Smoothing 0,1,
  je 3 Threads, beide Arme parallel.
- **Engine:** `experiments/drone_tying_ab/arm/libcortex.so`, der Tying-Build.
  Die produktive `libcortex.so` blieb während des gesamten Laufs unangetastet
  (sha256 gegen `backup_pre_tying_20260827_tying/` geprüft).

**Zwingende Randbedingung bei jeder Auswertung dieser Modelle:** der gebundene
Arm muss mit `SNAI_WEIGHT_TYING=1` gegen die Tying-Engine laufen. Ein
gebundenes Modell in einer ungebundenen Engine lädt den redundanten
`w_out`-Block über die Embeddings — das Ergebnis sieht auf den ersten Blick
nach Text aus und ist trotzdem Unsinn. Das Flag steht deshalb in jeder
Rohdatei als `<<CONFIG …>>`-Zeile mit drin, nachprüfbar statt erinnert.

**Vorbedingung, die vorher erfüllt wurde:** die Gradienten-Gegenprobe
(`diagnose/verify_weight_tying.php`) hat bestätigt, dass der zusammengelegte
Gradient korrekt ist — Kosinus-Ähnlichkeit 1,000000000000 gegen die
Referenzimplementierung, Streuung 1,571e-7 über 1024/1024 Elemente. Ohne
diesen Nachweis wäre der Qualitätsvergleich wertlos gewesen: der gefährliche
Fehler beim Tying ist still, das Training liefe weiter, die Zahlen sähen
plausibel aus, und ein Teil des Gradienten fehlte.

## Ergebnis 1: Held-out — gebunden gewinnt alle vier Epochen

| Epoche | ungebunden | gebunden | Vorsprung |
|---|---|---|---|
| 1 | 5,581775 | **5,437999** | −0,1438 |
| 2 | 5,276320 | **5,113990** | −0,1623 |
| 3 | 5,081747 | **4,885422** | −0,1963 |
| 4 | 5,049289 | **4,836527** | **−0,2128** |

Vier von vier, Vorsprung **monoton wachsend**, bei 2,6 % weniger Parametern.
Kein Zusammenlaufen.

**Der ungebundene Arm sättigt früher:** E3→E4 nur noch −0,032, der gebundene
−0,049. Die zwei Layer sind am Kapazitätsende, die sieben nicht.

**Nebenbefund, der für die Sache spricht:** in Epoche 3 hatte der gebundene
Arm den *schlechteren* Trainingsloss (5,005 gegen 4,953) bei besserem
Held-out. Kein Widerspruch, sondern Generalisierung — das geteilte Embedding
wirkt als Regularisierer.

### Das vorab benannte Risiko 2 ist widerlegt

`docs/STUDIE_WEIGHT_TYING.md` hatte gewarnt, die gemeinsame Matrix trage den
Summengradienten, die globale Norm steige, und der Clip greife dadurch härter
— was den Qualitätsvergleich konfundiert hätte. Der Rauchtest schien das zu
bestätigen (gebunden 3,5–8,2 gegen ungebunden 0,7–1,2).

**Das war ein Einschwingvorgang.** Alle großen Werte liegen in den ersten ~80
Schritten:

```
gebunden   Step 1(7,1) 2(4,2) 3(27,4) 10(50,3) 60(12,2) 70(14,1) 80(29,3)
           | ab hier: 355(2,1) 475(2,6) 585(2,7) 745(4,4) 845(2,1)
```

Über die vier Epochen liegt der gebundene Arm ab E2 durchgehend **niedriger**:

| GradNorm ⌀ | E1 | E2 | E3 | E4 |
|---|---|---|---|---|
| ungebunden | 1,169 | 1,556 | 1,556 | **1,711** |
| gebunden | 1,267 | 1,386 | 1,344 | **1,434** |

In E4 liegt das *Minimum* des ungebundenen Arms bei 1,211 — der Clip feuerte
dort in jedem einzelnen Schritt. Beim gebundenen Arm (Minimum 1,096) ebenfalls,
aber schwächer. Falls hier jemand unter härterem Clipping leidet, ist es der
ungebundene Arm.

Die Warnung aus der Studie war berechtigt und ist sauber ausgeräumt — genau
deshalb stand sie drin: *„angesichts der Projekthistorie sollte nach dem Umbau
die GradNorm-Verteilung explizit nachgemessen werden, nicht angenommen."*

## Ergebnis 2: Inferenz — der Held-out-Vorsprung bildet sich NICHT ab

Protokoll: **10 Prompts × 10 Läufe je Arm bei T=0,25** (100 Läufe je Arm), plus
4 Prompts × 3 bei T=0,90 als Gegenprobe. Themen und Frageformen bewusst
gekreuzt, damit sich Thema von Form trennen lässt.

Ein erster Durchgang mit nur drei Prompts hatte ein irreführendes Bild
geliefert (`results/inferenz_kennzahlen.txt`); die Aggregatwerte verdeckten,
dass beide Arme prompt-spezifisch versagen. Der breite Satz war die
Konsequenz daraus.

| Prompt | ungeb. Kollaps | ungeb. `<unk>`-L | geb. Kollaps | geb. `<unk>`-L |
|---|---|---|---|---|
| Beschreibe einen sonnigen Tag. | 0/10 | 0/10 | 0/10 | 0/10 |
| Beschreibe eine Stadt. | 0/10 | 3/10 | 0/10 | 0/10 |
| Erkläre mir das Wetter. | 0/10 | 4/10 | 0/10 | 0/10 |
| Erzähl mir etwas über die Natur. | 0/10 | 4/10 | 0/10 | 2/10 |
| Warum schlafen Menschen? | 1/10 | 5/10 | 0/10 | 1/10 |
| **Was ist ein Computer?** | 0/10 | 0/10 | **2/10** | 0/10 |
| **Was ist Musik?** | 0/10 | 1/10 | **3/10** | 0/10 |
| Was macht ein Arzt? | 0/10 | 2/10 | 0/10 | 0/10 |
| Was passiert im Winter? | 0/10 | 0/10 | 0/10 | 4/10 |
| Wie funktioniert ein Motor? | 0/10 | 0/10 | 0/10 | 1/10 |
| **gesamt** | **1/100** | **19/100** | **5/100** | **8/100** |

*Kollaps = Lauf mit >40 % wiederholten 4-Wort-Folgen. Die Schwelle ist
unkritisch: die Verteilung ist zweigipflig, kaputte Läufe liegen bei 55–91 %,
intakte unter 10 %, dazwischen liegt nichts. Wiederholung wird
**längenkontrolliert** gemessen — die rohe Quote zählt sonst faktisch die
Ausgabelänge mit.*

### Der Kollaps des gebundenen Arms ist formgebunden, nicht themengebunden

5 von 20 Läufen auf **„Was ist X?"**, 0 von 80 auf allem anderen. Computer und
Musik haben thematisch nichts gemeinsam. Technik allein löst es nicht aus —
„Wie funktioniert ein Motor?" läuft sauber (0/10, rep 0,5 %). „Was macht ein
Arzt?" und „Was passiert im Winter?" ebenfalls sauber. Es ist die
Definitionsfrage, die das Modell in einen Registerkollaps kippt:

> „Die Filme der Film- und Film- und Film- und Film- und Film-Preis. Der Film-
> und Film- und Film- und Film- und Film- und Film-…"

> „…Berlin-Klasse, Berlin, München, Berlin, Berlin, Berlin, Berlin, Berlin,
> Berlin, Berlin, München, Wien, Berlin, Berlin…"

Wohlgeformtes Deutsch, das nichts sagt — verhakt in einem Wikipedia-Film-/
Ortslisten-Register. Der ungebundene Arm zeigt das an denselben Prompts nicht
(Computer 0,9 %, Musik 0,9 % Wiederholung, kein Kollaps).

**Bei T=0,90 verschwindet der Kollaps vollständig** (0/3 auf beiden
Definitionsfragen). Es ist also keine Unfähigkeit, sondern eine
Modenverriegelung bei niedriger Temperatur.

### Inhaltstiefe geht an den ungebundenen Arm

`new_typ` (verschiedene Nicht-Prompt-Substantive je Lauf) über alle 100 Läufe:
ungebunden **29,6**, gebunden **24,5**. Der ungebundene Arm gewinnt an 7 von 10
Prompts — und das nicht nur wegen der Kollapse: bei „Wie funktioniert ein
Motor?" steht es 32,7 gegen 18,4 bei 0,5 % Wiederholung, bei „Was passiert im
Winter?" 29,0 gegen 15,5. Dort ist nichts kollabiert, der gebundene Arm sagt
schlicht weniger Verschiedenes.

### Bilanz des A/B

| | ungebunden | gebunden |
|---|---|---|
| defekte Läufe gesamt | 20/100 | **13/100** |
| Wiederholungskollaps | **1/100** | 5/100 |
| `<unk>`-Läufe | 19/100 | **8/100** |
| Inhaltstiefe `new_typ` | **29,6** | 24,5 |

Der `<unk>`-Vorteil des gebundenen Arms hat sich sauber repliziert (vorher bei
n=30: 1/30 gegen 10/30). Das ist der belastbarste Einzelbefund.

**Ein Held-out-Vorsprung von 0,213 bildet keines dieser Verhaltensmuster ab.**
Zum dritten Mal in diesem Projekt läuft Loss dem Verhalten davon — diesmal an
einer Architekturänderung statt an einem Optimierer-Fix.

## Ergebnis 3: der eigentliche Befund — wo das Drohnen-Tier inzwischen steht

Beide dokumentierten 128-Dim-Inferenzen zeigen dasselbe Bild.

**16.08.** (`INFERENZ_DROHNE_BASELINE.md`, `expert_SFT_fe86a817501`):

```
,,, und,.,.,,
,,,.,..,..,, der,,
.
,,
,,.
```

**18.08.** (`INFERENZ_DROHNEN_GRADCLIP_AB.md`, vier Experten, 12/12 Läufe):

```
, und,. der, die,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.,,,,,,,,,,,,,,, und.
```

Damals: keine Substantive, keine Verben, keine Flexion, keine Wortstruktur —
eine Schleife über `,` `.` „die" „der" „und". Kein einziger sauberer `</s>`-
Stopp in 15 Läufen.

**Heute** (dieselbe Geometrie, 128 Dim):

> „Die Telegenheit ist ein Dichter der Sünden, die sich auf den Hauber- und
> Persönlichkeitsstanzer und „Schwerbe" (). Die Zellen, die in der Fischung
> des Bäudes sind, wird von einem Schienen, der das Mühle, die im Bereich der
> Dichter, der die Wissenschaften und die Körferung der Geiseln- und Sünde-
> und Tümerzungen."

Das ist inhaltlich Unsinn — aber es ist **deutscher Satzbau**: Artikel-
Substantiv-Kongruenz, Relativsätze mit korrekter Einleitung („die sich auf…",
„der das…"), Präpositionalphrasen, Komposita, Genitivkonstruktionen,
Anführungszeichen paarweise. Die erfundenen Wörter („Telegenheit",
„Körferung", „Tümerzungen") sind morphologisch wohlgeformt — deutsche
Suffixe an deutschen Stämmen.

### Perplexität

| Modell | Loss | PPL | % des Zufalls (Vocab 8192) |
|---|---|---|---|
| 16.08. `fe86a817501` (Bug-Engine) | 7,4043 | 1643,1 | 20,06 % |
| 18.08. `7106a84930e` (gefixt) | 7,7173 | 2246,9 | 27,43 % |
| 18.08. `27c6a849c12` (gefixt) | 7,7139 | 2239,3 | 27,33 % |
| 18.08. `5b06a849a55` (gefixt) | 7,8549 | 2578,3 | 31,47 % |
| **28.08. ungebunden 2L** | **5,0493** | **155,9** | **1,90 %** |
| **28.08. gebunden 7L** | **4,8365** | **126,0** | **1,54 %** |

**Faktor 13 bis 18 in der Perplexität.** Von 20–31 % des Zufallsniveaus auf
1,5–1,9 %. Zum Vergleich: die 256-Dim-Prinzessin-Baseline lag am 16.08. bei
8,7 % — **die heutige Drohne unterbietet die damalige Prinzessin um mehr als
das Vierfache**, bei halber Dimension.

### Was diesen Vergleich einschränkt

Ehrlich und vollständig, weil die Zahl sonst mehr trägt, als sie kann:

1. **Anderer Held-out-Satz.** Die alten Werte stammen aus dem
   Pipeline-Holdout mit **999 Tokens**, den `INFERENZ_DROHNEN_GRADCLIP_AB.md`
   selbst als „winzig (hohe Varianz)" bezeichnet. Hier: 524.288 B, der
   gemeinsame Satz der Prinzessinnen-Läufe. **Die Werte sind nicht
   1:1 vergleichbar.** Ein Faktor 13–18 liegt allerdings weit außerhalb dessen,
   was Satz-Varianz erklärt — die Richtung des Befunds steht, die zweite
   Nachkommastelle nicht.
2. **Völlig andere Datenmenge.** Die alten Drohnen liefen laut Dokument
   „bewusst nur mit dem Minimum an Nektar pro Epoche" über 2 Epochen. Hier:
   35,2 MB über 4 Epochen mit frischem Material je Epoche. Das ist
   wahrscheinlich der größte Einzeltreiber.
3. **Alles andere hat sich auch geändert.** Zwischen dem 18.08. und heute
   liegen der Gradient-Clipping-Fix, der Korpus-Cleanup (Wikitabellen-Leck,
   `[Gekürzt]`-Marker, Archaik-Filter), die Quellen-Gewichtung des
   Foreager-Umbaus und die 100-%-Lang-Modus-Umstellung der SFT-Jagden.
4. **Andere Inferenzparameter.** Die alten Läufe liefen mit Repetition
   Penalty 1.15 und min_p 0.05; `INFERENZ_DROHNE_BASELINE.md` vermutet
   ausdrücklich, das Pendeln zwischen Satzzeichen könnte teilweise ein
   Artefakt der Penalty sein.

**Konsequenz:** der Fortschritt ist real und groß, aber er ist **nicht dem
Weight Tying zuzuschreiben**. Weight Tying ist die kleine Differenz obendrauf
(0,213 Held-out zwischen den Armen); der Sprung von 2247 auf 156 PPL liegt
zwischen ungebunden-heute und den Drohnen vom 18.08. — also in allem, was
seither an Daten und Korpusqualität dazugekommen ist. Der ungebundene Arm
allein bringt schon Faktor 14,4.

Damit ist die offene Frage aus `INFERENZ_DROHNE_BASELINE.md` — „echte
Verschlechterung oder Wegfall von Memorierung?" — nachträglich beantwortbar:
weder noch. Das Tier war schlicht **unterfittet**, wie dort unter Punkt 3 als
wahrscheinlichste Erklärung vermutet („eher generell unterfitted als
generalisiert, aber nicht memoriert"). Mit ausreichend Material verschwindet
der Funktionswort-Kollaps von selbst. Diese Vermutung ist damit bestätigt.

Ebenso relativiert sich das Nullresultat des Drohnen-Gradient-Clipping-A/B (18.08.): der Clipping-Fix
zeigte auf Drohnen-Ebene keinen Effekt, weil das Modell zu unterfittet war, um
von saubererem Gradientenbetrag zu profitieren. Das Dokument hatte genau das
als Erklärung angeboten und die Datenknappheit als Deckel benannt. Beides
bestätigt sich.

### Nebenfund: `&nbsp;`-Leck weiterhin im Korpus

In einer Ausgabe des ungebundenen Arms:

> „…eine „West" (Hrsg.&nbsp;Jahrhundert), dass die Brieden…"

Das am 20.08. gefundene `&nbsp;`-Leck aus Wikipedia (`INFERENZ_MATRIARCHIN_
1557A2_EPOCHE2_3.md`) steckt also weiterhin im GEN-Nektar und wird
mittrainiert. Weiterhin offen, hier nur als Lebendsichtung vermerkt.

## Bewertung

**Für ein 128-Dim-Modell ist das ein deutlicher Fortschritt.** Beide Arme
erzeugen grammatisch strukturiertes Deutsch, wo dasselbe Tier vor zehn Tagen
Satzzeichenketten produzierte. Die Erwartungshaltung „128 Dim = Satzzeichen"
ist überholt; das Tier liefert Syntax. Inhalt liefert es nicht, und das war
auch nie zu erwarten.

**Weight Tying selbst ist kein klarer Verhaltensgewinn.** Es tauscht:

- **gewonnen:** deutlich weniger `<unk>`-Zerfall (8 gegen 19 Läufe), insgesamt
  weniger defekte Läufe (13 gegen 20), klarer und monoton wachsender
  Held-out-Vorsprung, spätere Sättigung
- **verloren:** geringere lexikalische Vielfalt (24,5 gegen 29,6), plus eine
  neue systematische Schwäche bei Definitionsfragen
- **Preis:** rund doppelte Rechenzeit je Epoche bei gleicher Parameterzahl
  (7 statt 2 Layer sequentielle Tiefe)
- **noch kein Speichergewinn:** die gebundene Datei ist mit 41,7 MB *größer*
  als die ungebundene mit 29,9 MB — das Format schreibt den redundanten
  `w_out`-Block weiter mit. Der Platzgewinn käme erst mit dem Formatwechsel
  (`SNAI_V7`), Schritt 2 der Studie.

## Empfehlung und offene Punkte

**Weight Tying noch nicht in die Schwarm-Pipeline übernehmen.** Nicht wegen eines
allgemeinen Vorbehalts, sondern wegen eines konkreten Befunds: eine
systematische Schwäche ausgerechnet bei Definitionsfragen ist für ein Modell,
das später Instruktionen befolgen soll, die falsche Schwäche.

Offen, in dieser Reihenfolge sinnvoll:

1. **Bleibt die „Was ist X?"-Schwäche auf Prinzessin-Geometrie bestehen?**
   Wenn sie ein Drohnen-Kapazitätsartefakt ist, fällt das Hauptargument gegen
   die Übernahme weg. Das ist der entscheidende Test.
2. **Zweiter Seed auf Drohnen-Ebene.** Ein Seed, 5 Kollapse — die Zahl ist
   klein genug, dass ein zweiter Lauf sie bestätigen sollte.
3. **Formatwechsel `SNAI_V7`**, falls 1. positiv ausgeht — sonst bleibt der
   Speichervorteil rein theoretisch.

## Rohdaten

- `experiments/drone_tying_ab/results/summary.txt` — Held-out je Epoche/Arm
- `experiments/drone_tying_ab/results_wide/matrix.txt` — Inferenzmatrix
- `experiments/drone_tying_ab/results_wide/inf_{untied,tied}_T0{25,90}.txt` —
  alle 224 Läufe roh, je mit `<<CONFIG tying=…>>`-Kopf
- `secured_models/vergleich_20260827/drone_{tied7l,untied2l}_e4.snai` —
  beide Endmodelle, sha256 in `CHECKSUMS.txt`

---

## NACHTRAG 2026-09-01 — zweiter Seed gefahren (offener Punkt 2)

`experiments/drone_tying_ab_seed2/` (Volltext dort in `ERGEBNIS.md`). Eine Variable: der
Init-Seed. Material, Held-out, Engine und alle Hyperparameter bitgleich zu
diesem Lauf.

**Was hält:**

| | Lauf 1 ungeb./geb. | Seed 2 ungeb./geb. |
|---|---|---|
| Kollapse auf **„Was ist X?"** (n=20) | 0 / **5** | 3 / **8** |
| Kollapse sonst (n=80) | 1 / 0 | 2 / 1 |
| `<unk>`-Läufe (n=100) | 19 / **8** | 64 / **31** |
| Held-out-Vorsprung gebunden E1→E4 | 0,144 → 0,213 | 0,119 → 0,164 |

Über beide Seeds steht der gebundene Arm bei **13/40 auf „Was ist X?" gegen
1/160 auf allem anderen**. Die Formgebundenheit ist bestätigt, der
`<unk>`-Vorteil ebenfalls (rund 2:1 in beiden Läufen), der Held-out-Vorsprung
wächst in beiden monoton, und bei T=0,90 verschwindet der Kollaps erneut
vollständig.

**Was nicht hält — zwei Korrekturen an diesem Dokument:**

1. **„Der ungebundene Arm zeigt das an denselben Prompts nicht"** gilt nur für
   Lauf 1. Im zweiten Seed kollabiert er dort 3/20. Über beide Seeds liegt er
   bei 3/40 auf „Was ist X?" gegen 3/160 sonst — dieselbe Präferenz, viermal
   schwächer. **„Was ist X?" ist die schwierige Frageform für beide Arme; das
   Tying verstärkt eine vorhandene Schwäche, es erzeugt sie nicht.**

2. **„Inhaltstiefe geht an den ungebundenen Arm"** (`new_typ` 29,6 gegen 24,5)
   **ist widerlegt.** Im zweiten Seed dreht es sich um (15,3 gegen 23,5), und
   die Ursache ist messbar: `new_typ` korreliert über die 20 Arm×Prompt-Zellen
   mit der Zahl der `<unk>`-Läufe zu **r = −0,648** (≤3 `<unk>`-Läufe: 24,8;
   ≥4: 14,0). `<unk>`-Token verdrängen die Substantive, die die Metrik zählt.
   **`new_typ` ist keine unabhängige Qualitätsachse, sondern weitgehend
   nachgelagert zum `<unk>`-Problem.** Der Vorsprung von Lauf 1 wurde gemessen,
   als der ungebundene Arm nur 19 `<unk>`-Läufe hatte.

**Neuer Befund:** der **ungebundene** Arm ist der init-empfindlichere. Zwischen
den Seeds ändert sich der gebundene kaum (`new_typ` −1,0), der ungebundene
stark (−14,3, `<unk>` 19 → 64). Bei gleicher Parameterzahl und gleichem
Material ist die tiefe, gebundene Variante über zwei Draws die vorhersagbarere.

**Folge für die Bilanz dieses Dokuments.** Die Entscheidung „kein Tying in die
Schwarm-Pipeline" bleibt, ruht aber jetzt auf **einem** Bein statt auf dreien: der
Definitionsfragen-Schwäche. Die stützte sich hier zusätzlich auf die
Inhaltstiefe (widerlegt) und auf einen Kontrast von 5:1 (real 9:5). Dagegen
stehen drei replizierte Punkte für das Tying — Held-out, `<unk>`-Rate,
Init-Stabilität. Die Bilanz ist deutlich enger als hier beschrieben; sie kippt
nur nicht, weil die verbliebene Schwäche genau die Fähigkeit trifft, um die es
später geht.

Offener Punkt 2 ist damit erledigt. Punkt 3 (`SNAI_V7`) war schon vorher
entfallen.
