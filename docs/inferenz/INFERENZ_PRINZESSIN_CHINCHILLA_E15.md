# Inferenz Chinchilla-Prinzessin — Gesamtreihe Epoche 1–15 (2026-08-27)

**Modell:** `experiments/princess_chinchilla/results/princess_e15.snai`
(Vocab 8192 | Dim 256 | Ctx 1024 | 3 Layer | FFN 1024 | 4 Heads, 6,55 M Parameter)
**Checkpoint E15:** `0f5aa7f0491bf60f…`
**Korpus:** 459,9 MB eingefrorener Nektar, 15 Epochen mit je frischem Material
(jeder Token genau einmal gesehen), ~137 M Token ≈ 105 % Chinchilla.
**Held-out:** `data/holdout/holdout.txt`, 524 288 B, über alle 15 Epochen
identisch (md5 `9b0730bc5ea58e5c…`, byteweise dieselbe Datei wie im
SFT-Mix-Lauf — die beiden Reihen sind gegeneinander vergleichbar).
**Protokoll:** `compare_inference.sh`, 3 Prompts × n=10 bei T=0,25 plus
2 Prompts × n=3 bei T=0,90. E6 wurde am 27.08. nachgemessen, alle übrigen
Epochen liefen mit dem Training mit.

## Der Lauf ist vollständig und sauber ausgelaufen

15 von 15 Epochen, kein Abbruch, alle Checkpoints vorhanden. Der Held-out
fällt monoton von 4,993774 auf **3,457885**, also um 1,54 über den ganzen
Lauf. Die letzten drei Epochen liefen auf der LR-Untergrenze (0,00005, ab
E12,3 erreicht) und gewinnen entsprechend wenig — E14 → E15 nur noch 0,0066.
Das ist ein sauberes Auslaufen, kein Steckenbleiben.

## Gesamttabelle

| E | Held-out | topic_own | topic_foreign | lift | eos_ok | unk | avg_chars | rep40 |
|---|---|---|---|---|---|---|---|---|
| 1 | 4,993774 | — | — | — | 8/30 | 0 | 512,7 | 18,4 % (n=14) |
| 2 | 4,552109 | — | — | — | 15/30 | 0 | 131,2 | 21,1 % (n=1) |
| 3 | 4,117072 | — | — | — | 6/30 | 0 | 163,4 | 13,9 % (n=11) |
| 4 | 3,918850 | 6/30 | 0/30 | ∞ | 28/30 | 0 | 94,5 | 39,5 % (n=3) |
| 5 | 3,828849 | 2/30 | 6/30 | 0,67 | 25/30 | 0 | 164,2 | 8,3 % (n=13) |
| 6 | 3,753059 | 4/30 | 5/30 | 1,64 | 25/30 | 0 | 202,5 | 36,8 % (n=11) |
| 7 | 3,696801 | 6/30 | 4/30 | 2,89 | 29/30 | 0 | 138,7 | 11,4 % (n=3) |
| 8 | 3,538022 | 8/30 | 9/30 | 1,79 | 18/30 | 0 | 252,9 | 9,7 % (n=25) |
| 9 | 3,546186 | 4/30 | 2/30 | 3,60 | 13/30 | 0 | 148,6 | 15,1 % (n=20) |
| 10 | 3,523205 | 14/30 | 3/30 | 8,29 | 15/30 | 0 | 354,6 | 9,3 % (n=20) |
| 11 | 3,498776 | 9/30 | 2/30 | 7,60 | 24/30 | 0 | 157,4 | 18,4 % (n=10) |
| 12 | 3,479128 | 5/30 | 1/30 | 7,33 | 22/30 | 0 | 292,2 | 18,6 % (n=12) |
| 13 | 3,465269 | 2/30 | 3/30 | 1,43 | 26/30 | 0 | 161,3 | 26,8 % (n=6) |
| 14 | 3,464516 | 6/30 | 0/30 | 26,00 | 22/30 | 0 | 242,1 | 10,5 % (n=11) |
| **15** | **3,457885** | **5/30** | **3/30** | **3,14** | **12/30** | **0** | **285,9** | **11,0 % (n=21)** |

`topic_*` erst ab E4, davor war die Metrik noch nicht gebaut. `rep40` ist die
längenkontrollierte Wiederholungsrate, siehe unten.

## Befund 1: der Loss fällt, das Verhalten folgt nicht

Über 1,54 Loss hinweg bewegt sich die Prompt-Konditionierung **nicht**. E4 lag
bei 6/30, E15 bei 5/30; dazwischen schwankt `topic_own` zwischen 2 und 14 ohne
Richtung. Der `topic_lift` springt von 0,67 (E5) auf 26,00 (E14) und zurück auf
3,14 — bei Zählern zwischen 0 und 14 ist das Streuung, kein Signal.

Das ist die aus dem Projekt bekannte Loss/Verhalten-Divergenz, hier aber in
einer Form, die dem bisherigen Muster **widerspricht**.
Loss und Verhalten vor allem bei *kleinen* Differenzen zwischen ähnlichen
Checkpoints auseinanderlaufen, während große Tier-/Regime-Sprünge zuverlässig
korrelieren. Hier liegt ein großer Loss-Unterschied vor (4,99 → 3,46, über den
gesamten Lauf derselben Architektur) und trotzdem keinerlei Bewegung in der
Konditionierung.

## Befund 2: `<unk>` existiert auf diesem Budget nicht

Über alle 15 Epochen und alle 450 Läufe: **null `<unk>`.** Der Kollaps, der
die Matriarchin-Reihe geprägt hat und dort erst zwischen Epoche 1 und 2 von
selbst verschwand, tritt hier nie auf — nicht einmal in Epoche 1.

## Befund 3: die Leitmetrik zählt teilweise Degeneration als Erfolg

Beim Lesen der Rohdaten aufgefallen: E15, Prompt „Erzähl mir etwas über die
Natur", Lauf 2 lautet auszugsweise

> „Die Natur der Natur ist ein Beispiel für Natur und Natur der Natur. Sie ist
> eine breite Palette, die die Natur der Natur erfüllt."

Dieser Lauf zählt als `topic_own`-Treffer, weil „Natur" vorkommt. Eine
Phrasenschleife auf dem Prompt-Wort ist aber das Gegenteil dessen, was die
Metrik messen soll. Ein Teil der `topic_own`-Treffer ist damit Degeneration,
die als Konditionierung verbucht wird — die Metrik überschätzt systematisch
nach oben, und zwar am stärksten bei genau den Läufen, die am schlechtesten
sind. Für künftige Auswertungen wäre ein Ausschluss von Läufen mit hoher
Wiederholungsrate vor der `topic`-Zählung angebracht.

## Befund 4: ein Längenartefakt, das fast zu einer falschen Aussage geführt hätte

Erste Messung der Wiederholungsrate über die **gesamte** Ausgabe je Lauf:

| E | roh |
|---|---|
| 2 | 2,6 % |
| 7 | 4,4 % |
| 14 | 12,5 % |
| **15** | **33,7 %** |

Das sah nach einem klaren Ergebnis aus: E15 als schlechtester Wert des ganzen
Laufs, Degeneration nimmt mit sinkendem Loss zu.

**Die Zahl ist ein Artefakt der Ausgabelänge.** `avg_chars` läuft mit: E2 hatte
131 Zeichen, E15 hat 286. Wer länger schreibt, hat mehr Gelegenheit, ein
Trigramm zu wiederholen. Auf die ersten 40 Wörter jedes Laufs kontrolliert
(Läufe unter 40 Wörtern verworfen statt kurzgerechnet, sonst wäre die
Stichprobe wieder längenabhängig) liegt E15 bei **11,0 %** und damit im
Mittelfeld — E1 hat 18,4 %, E8 9,7 %, E14 10,5 %.

Die Schleifenrate je Token ist also nicht gestiegen. Was gestiegen ist: die
Ausgabelänge, bei gleichzeitig sinkender Stopp-Quote (E15: 12/30 gegen 22/30
bei E14). Dieselbe Rate wird dadurch sichtbarer, ohne schlimmer zu sein.

**Merker:** eine Verhaltensmetrik, die über einen ganzen generierten Text
mittelt, misst immer auch dessen Länge mit. Die Kontrolle war hier nicht
optional — sie hat das Vorzeichen des Befundes umgedreht.

## Befund 5: was der Text tatsächlich kann

Epoche 1 gegen Epoche 15, derselbe Prompt, dieselben Einstellungen.

**E1** — grammatisch stellenweise kaputt, Kongruenz- und Rektionsfehler:

> „Die Lösung erfordert, müssen die Anforderungen der Reihenfolge der Analyse
> der Gesamtzeitung der Anordnung der Regeln und die beste Kombination zu
> finden. Zuerst ist wichtig, dass jede Gruppe der Teilnehmer berücksichtigen…"

**E15** — durchgehend wohlgeformtes Deutsch, korrekte Syntax, Absatzstruktur,
Aufzählungen:

> „Die Überlegungen zur Lösung des Problems erfordern eine sorgfältige Analyse
> der gegebenen Einschränkungen und die Anzahl der verfügbaren Ressourcen. […]
> Um die richtige Reihenfolge zu finden, müssen die spezifischen Bedürfnisse
> jedes Problem analysiert werden."

Der Fortschritt in der **Form** ist real und deutlich. Inhaltlich hat sich
nichts bewegt: auf „Was ist ein Computer?" antwortet E15 mit
Ressourcenoptimierung und Teilnehmerzahlen, auf „Erzähl mir etwas über die
Natur" mit Kulturdenkmalen. Das Modell hat ein SFT-Register gelernt — „Die
Herausforderung besteht darin…", „Um die optimale Lösung zu finden…" — und
füllt es unabhängig vom Prompt.

## Einordnung in die Ladder

Zum Vergleich die anderen Stufen dieses Projekts, jeweils die beste
dokumentierte Ausgabe:

| Stufe | Dim/Layer | Parameter | Ausgabe |
|---|---|---|---|
| Drohne | 128/2 | ~1,3 M | `,,,.,..,..,,` — Schleife aus Satzzeichen und Funktionswörtern, keine Wortstruktur (`INFERENZ_DROHNE_BASELINE.md`) |
| **Prinzessin (Chinchilla)** | **256/3** | **6,55 M** | **durchgehend wohlgeformtes Deutsch, korrekte Syntax, kein Prompt-Bezug** |
| Matriarchin `1557a2` E3 | 512/6 | 27,3 M | wohlgeformte, stellenweise faktisch plausible Sätze („Die erste Hälfte des 19. Jahrhunderts wurde die Kirche im Jahr 1571 eingeweiht."), daneben weiter Wortsalat (`INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md`) |

**Wichtig beim Lesen dieser Tabelle:** die drei Stufen sind **nicht** auf
vergleichbarem Token-Budget trainiert. Die Prinzessin hier ist die einzige
Chinchilla-optimale Stufe des Projekts; Drohne und Matriarchin liefen deutlich
unterversorgt (Matriarchin bräuchte ~545 M Token). Die Tabelle vergleicht also
„beste bisher erreichte Ausgabe je Stufe", nicht „Stufe bei gleichem Budget".

## Fazit

Die Frage, für die der Lauf gebaut wurde — bringt Chinchilla-Sättigung den
erwarteten Qualitätssprung — ist **negativ beantwortet, aber sauber**:
volle 15 Epochen, frisches Material in jeder, durchgehendes Held-out-Tracking,
kein Abbruch, kein Optimierer-Reset. Der Sprung bleibt aus.

Die naheliegende Erklärung wäre Kapazität: bei 6,55 M Parametern reicht es für
Syntax, nicht für Prompt-Bindung.

**Diese Erklärung ist am selben Tag widerlegt worden** — siehe den Nachtrag
unten. Der Kontrollarm `princess_sftmix` erreicht mit derselben Architektur
`topic_own` 29/30. Die Grenze liegt am Datenmix, nicht an der Parameterzahl.

## Nachtrag (27.08., ~18:00): der Kontrollarm widerlegt die Kapazitäts-These

`experiments/princess_sftmix/` ändert gegenüber diesem Lauf **genau eine
Variable**: den SFT-Anteil, gemessen **77,8 % gegen 14,7 %** nach Bytes.
Gleiche Geometrie (256/3), gleiches Held-out. Nach dem Nachmessen von E7–E10:

| E | Chinchilla `own` | Chinchilla lift | SFT-Mix `own` | SFT-Mix lift |
|---|---|---|---|---|
| 6 | 4/30 | 1,64 | 3/30 | 1,56 |
| 7 | 6/30 | 2,89 | **22/30** | 3,91 |
| 8 | 8/30 | 1,79 | **29/30** | 7,87 |
| 9 | 4/30 | 3,60 | **23/30** | 7,23 |
| 10 | 14/30 | 8,29 | **23/30** | 8,55 |

Bis E6 sind die Arme ununterscheidbar (3/30 gegen 4/30). Zwischen E6 und E8
springt der SFT-Arm auf 29/30 und bleibt dort. Der Chinchilla-Arm erreicht über
15 Epochen nie mehr als 14/30 und endet bei 5/30.

**Die Ausgabe des SFT-Arms beantwortet den Prompt.** E10 auf „Was ist ein
Computer?":

> „Der Computer ist ein Computer, der sich auf eine Vielzahl von Anwendungen
> verwendet, um die Verwendung von Daten zu speichern und die Daten zu
> analysieren."

Zirkulär und holprig, aber thematisch korrekt und sauber gestoppt (156 Zeichen,
`eos=1`). Der Chinchilla-Arm antwortet auf denselben Prompt mit
Ressourcenoptimierung.

**Prompt-Bindung ist bei 256/3 also erreichbar.** Sie braucht einen
SFT-lastigen Mix und tritt erst nach ~7 Epochen auf — wer vorher misst, sieht
nichts und schließt falsch.

### Das Held-out ist zwischen den Armen nicht neutral

`holdout.txt` enthält **null `[USER]`-Marker, ist also zu 100 % GEN.** Der
GEN-lastige Chinchilla-Arm wird damit auf genau der Verteilung geprüft, auf der
er trainiert hat, der SFT-Arm auf einer fremden. Dass Chinchilla mit 3,4579
gegen 3,8911 „besser" dasteht, ist deshalb **kein Qualitätsurteil** — es misst
Fließtext-Fortsetzung, nicht Instruktionsbefolgung, und der schlechtere Wert
gehört zum Modell, das den Prompt tatsächlich beantwortet.

Für künftige Vergleiche über verschiedene Datenmixe hinweg braucht es entweder
zwei Held-out-Sätze (je einen pro Verteilung) oder einen, der dem Mix beider
Arme entspricht. Der aktuelle taugt nur innerhalb eines Arms.
