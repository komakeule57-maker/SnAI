# SFT-Nachtraining der Matriarchin — das Rezept der Prinzessin trägt eine Stufe höher nicht, und die Definitionsfrage kollabiert in Schweigen

Auswertung am 2026-08-30, 09:45 CEST. Der am 29.08. 02:36 gestartete Lauf
`experiments/matriarchin_sft_finetune/` ist am 30.08. 05:07 nach zwei Epochen
fertig geworden. Basis ist die GEN-Matriarchin `1fdf68` E3 (sha256
`03e317f401bb69f9…`, 512 Dim | Ctx 2048 | 6 Layer | 8 Heads), **10,4 MB**
reines SFT-Material (10 877 971 B, 16 860 Beispiele, sha256
`1149f550f1f8ba8d…`), Spitzen-LR 0,00005 (ein Zehntel der Vortrainings-Spitze),
dieselben Daten in beiden Epochen. Aufbau, Basiswahl und LR sind eins zu eins
vom Prinzessin-Fine-Tune übernommen (`INFERENZ_PRINZESSIN_SFT_FINETUNE_E1_E3.md`).

**Die Verhaltenskennzahlen sind nachgezogen, nicht mitgelaufen.**
`run_finetune.sh` maß nur Held-out und Kontextsonde. Das reicht nach Befund (a)
nicht: der Held-out entscheidet auf dieser Messreihe nichts. Die Zahlen unten
stammen aus `compare_inference.sh 0 1 2`, am 30.08. nachträglich gefahren —
gleiche Prompts, gleiches `tally()`, gleiches n wie bei der Prinzessin
(md5 des `tally()`-Blocks gegen `princess_sft_fresh_e2/` geprüft). Die
Checkpoints lagen unverändert vor, es wurde nichts nachtrainiert.

**Die Lernrate war nicht konstant.** `train_cluster.php:419–429` fährt Warmup
plus Cosine-Decay mit einem Boden bei 10 % der Spitze. Bei **zwei**
Schedule-Epochen springt das direkt auf den Boden: **E1 0,000050 → E2
0,000005**. Die Prinzessin hatte bei drei Epochen die Zwischenstufe 0,000025.

## Die Frage

Der Prinzessin-Fine-Tune hat in **einer** Epoche die Themenbindung
vervierfacht (`topic_own` 5/30 → 20/30). Die Frage war, ob dieselbe
Zwei-Phasen-Anordnung eine Stufe höher dasselbe leistet — bei vierfacher
Parameterzahl und doppeltem Kontextfenster.

## Kernergebnis

| | **E0 (Basis `1fdf68` E3)** | E1 | E2 |
|---|---|---|---|
| `topic_own` | 7/30 | 5/30 | **0/30** |
| `topic_foreign` | 13/30 | 5/30 | 3/30 |
| `topic_lift` | 1,11 | 2,00 | 0,29 |
| `new_typ` | 20,4 | 15,1 | 14,0 |
| `noun_ttr` | 0,84 | 0,77 | **0,66** |
| `echo_pct` | 0,1 | 0,0 | 0,0 |
| `taut_pct` | 0,0 | 0,0 | 9,1 |
| `distinct_run_starts` | 27 | 17 | **9** |
| `register_lock` | 2/30 | 4/30 | 8/30 |
| `eos_ok` | 20/30 | 29/30 | 27/30 |
| **Leere Antworten (`chars=0`)** | **0/30** | **3/30** | **7/30** |
| `unk` / `[SYMBIO]`-Leck | 1 / 2 | 0 / 0 | 0 / 0 |
| Held-out GEN | 4,9106 | 5,0076 | 5,0035 |
| Held-out SFT | 4,9177 | 4,7053 | **4,6880** |
| Kontextsonde | 0/50 | 0/50 | 0/50 |

**Die Antwort auf die Ausgangsfrage ist negativ.** Themenbindung entsteht
nicht, sie verschwindet: `topic_own` läuft 7 → 5 → 0. E0 liegt mit
`topic_lift` 1,11 exakt auf Zufallsniveau (bei zwei fremden Stichwortsets
gegen ein eigenes ist die Zufallserwartung `foreign` ≈ 2 × `own`; 13 zu 7
trifft das), und E2 findet das eigene Thema in **keinem** von 30 Läufen.

Der Schritt E0 → E1 (7 gegen 5) ist zu klein zum Interpretieren. Belastbar
sind die 0/30 gegen 7/30 und die monotone Richtung über drei Messpunkte.

## Der Zusammenbruch ist prompt-gebunden — und er ist Schweigen

Aufgeschlüsselt nach Prompt (`topic_own`, n=10 je Zelle):

| Prompt | E0 | E1 | E2 |
|---|---|---|---|
| „Erzähl mir etwas über die Natur." | 4/10 | 1/10 | 0/10 |
| „Beschreibe einen sonnigen Tag." | 3/10 | 4/10 | 0/10 |
| **„Was ist ein Computer?"** | **0/10** | **0/10** | **0/10** |

Die Definitionsfrage steht über alle drei Epochen auf null — **auch im
unveränderten Basismodell**. Das ist der Spiegel des Prinzessin-Befunds: dort
sättigte genau dieser Prompt ab E1 bei 10/10 und hielt, während die beiden
offenen Prompts einbrachen. Hier ist er von Anfang an tot.

Der Grund wird sichtbar, sobald man die Ausgabelänge danebenlegt (Mittel der
Zeichen je Lauf aus dem `META`-Feld, leere Läufe eingerechnet):

| Prompt | E0 | E1 | E2 |
|---|---|---|---|
| „Was ist ein Computer?" | 607,5 | 82,6 | **30,8** |
| „Erzähl mir etwas über die Natur." | 346,2 | 856,1 | 847,3 |
| „Beschreibe einen sonnigen Tag." | 574,4 | 403,7 | 245,6 |
| leere Antworten (`chars=0`) gesamt | 0/30 | 3/30 | 7/30 |

**Alle leeren Antworten entfallen auf den Computer-Prompt** — 3 von 10 in E1,
7 von 10 in E2. Das Modell setzt dort sofort `</s>` und sagt gar nichts. Die
beiden offenen Prompts werden gleichzeitig länger, nicht kürzer (Natur mehr
als verdoppelt). Es ist also kein globaler Verfall, sondern ein
formspezifischer: **die Definitionsform kollabiert, während der Rest
ausufert.**

Damit ist das die dritte unabhängige Replikation derselben Gestalt — nach dem
Drohnen-Weight-Tying-A/B (Kollaps 5/20 auf „Was ist X?", 0/80 sonst) und dem
Prinzessin-Fine-Tune (nur die Definitionsform hält, die offenen fallen). Neu
ist, dass der Mechanismus hier offen zutage liegt: sofortiges EOS.

## `eos_ok` misst hier nicht, was es zu messen vorgibt

`eos_ok` springt von 20/30 auf 29/30 und sieht wie der klarste Gewinn des
Laufs aus. Er ist zum Teil ein Artefakt: **eine leere Antwort mit sofortigem
`</s>` zählt als sauberer Stopp.** Von den 29 sauberen Stopps in E1 sind 3
leer, von den 27 in E2 sind 7 leer. Bereinigt bleibt E1 26/30 und E2 20/30 —
gegenüber 20/30 in E0 ist das in E1 noch ein echter Gewinn und in E2 gar
keiner mehr.

Dieselbe Falle steckt in `avg_chars` (243,6 / 249,2, unauffällig): die Kennzahl
mittelt über nicht-leere **Zeilen** und lässt leere Läufe komplett heraus. Der
Zusammenbruch beim Computer-Prompt ist in ihr unsichtbar. Beide Kennzahlen sind
nicht falsch gerechnet, sie beantworten nur eine andere Frage als die, für die
man sie beim Überfliegen hält.

**Merker:** eine Stopp-Metrik braucht neben sich die Länge, sonst ist
„hört sauber auf" von „sagt nichts" nicht zu unterscheiden. Der
`chars=0`-Zähler gehört nach dieser Erfahrung fest in `tally()`.

**Rückwirkend geprüft, und die Entwarnung ist eindeutig:** die vier
Prinzessin-Messreihen (`princess_sft_finetune` E1/E2/E3,
`princess_sft_fresh_e2` E2) haben **0/30 leere Antworten**. Ihre
`eos_ok`-Werte sind sauber, die dortigen Schlüsse bleiben unberührt. Der
Befund ist auf diesen Lauf beschränkt — er entwertet keine älteren Zahlen,
sondern zeigt, wo die Metrik eine Grenze hat.

## Was tatsächlich gelernt wurde: die Form ohne den Inhalt

`register_modal_prefix` lautet ab E1 „ Natürlich! Hier sind " — ein
Lehrbuch-Einstieg für eine Assistenzantwort. Dazu passt der Rest: `<unk>` und
`[SYMBIO]`-Leck gehen auf null, die einzige Strichkette (≥ 20 Bindestriche am
Stück, der bekannte Markdown-Tabellentrenner-Kollaps) verschwindet nach E0.
Gleichzeitig fällt `new_typ` von 20,4 auf 14,0, `noun_ttr` von 0,84 auf 0,66,
Tautologien tauchen in E2 auf (9,1 %) und die Vielfalt der Satzanfänge bricht
von 27 auf 9 ein.

Das ist die kontrollierte Replikation des `f7cfff`-Befunds vom 26.08.
(„lernte die Form einer Assistenzantwort ohne Inhalt") — diesmal mit
E0-Referenz und in der Zwei-Phasen-Anordnung statt im Mix-A/B.

## Loss und Verhalten laufen gegeneinander

Der SFT-Holdout verbessert sich monoton (4,9177 → 4,7053 → 4,6880), der
Trainings-Loss ebenso (4,2261 → 3,9594). `topic_own` fällt über dieselben
Epochen 7 → 5 → 0. **E2 ist nach dem Held-out die beste Epoche und nach jeder
Verhaltensmetrik die schlechteste.** Hätte man den Lauf bei `run_finetune.sh`
belassen, wäre er als Erfolg verbucht worden: −0,230 SFT-Holdout gegen
+0,093 GEN-Vergessen ist genau der Handel, den man sich von einer SFT-Phase
erhofft.

Das ist dieselbe Gegenläufigkeit wie beim Prinzessin-Fine-Tune, nur ohne den
dortigen Gewinn in Epoche 1. Befund (a) der Übergabenotiz hält damit auch auf
Matriarchin-Ebene.

**„Zu hohe Lernrate" scheidet als Erklärung aus.** E2 lief bei 0,000005, einem
Zehntel der Spitze und einem Fünftel dessen, was bei der Prinzessin die
Zwischenstufe war. Der Einbruch entstand unter dieser Lernrate.

## Qualitativ

**E0**, auf „Was ist ein Computer?" — Wikipedia-Ortsartikel-Register,
prompt-unabhängig:

> „Der Ort liegt etwa 15 Kilometer nordöstlich des Landesamtes (1600–1767) und
> die Gemeindegrenze von 10.000 Einwohner."
>
> „Die Kirche wurde am 12. Jahrhundert gegründet, die im 19. Jahrhundert in
> der Stadtgemeinde von 1818–1553."

**E2**, auf die offenen Prompts — Assistenz-Register, thematisch vollständig
entkoppelt. Links steht jeweils der Prompt:

> „Erzähl mir etwas über die Natur." → „Die optimale Lösung besteht darin, die
> spezifische Reihenfolge der verfügbaren Ressourcen zu berücksichtigen. Die
> Herausforderung besteht darin, dass die Analyse der Teilnehmer in den
> Bereichen der Th…"
>
> „Beschreibe einen sonnigen Tag." → „Natürlich! Hier sind einige der
> Anforderungen, die für die Regeln und die Anzahl der Analyse der
> Anforderungen von den anderen Themen zu berücksichtigen."

**E2**, auf „Was ist ein Computer?" — in 7 von 10 Läufen buchstäblich nichts
(`chars=0 eos=1`).

Der Registerwechsel von E0 zu E2 ist unübersehbar und geht in beiden Fällen am
Prompt vorbei: vorher Ortsartikel, nachher Projektmanagement-Floskeln.

## Was diese Zahlen NICHT hergeben

**1. Sie erklären nicht, warum das Rezept nicht überträgt.** Es stehen
mindestens drei Kandidaten, und dieser Lauf trennt sie nicht:

- **Materialmenge.** 10,4 MB für die vierfache Parameterzahl, wo die
  Prinzessin 36,3 MB bekam — Faktor 3,5 weniger bei Faktor 4 mehr Gewichten.
- **Unterfittete Basis.** `1fdf68` wurde am 21.08. nach Epoche 3 von 6
  gestoppt. Die Prinzessin-Basis `chinchilla_e15` war mit 15/15 Epochen
  durchtrainiert.
- **Die Stufe selbst.**

Die ersten beiden zeigen weg von „die Stufe ist zu groß" und decken sich mit
zwei bestehenden Befunden: dem offenen Punkt 6 („länger laufen lassen statt
größer bauen") und dem Drohnen-Nebenbefund vom 29.08. („es war unterfittet,
nicht zu klein"). Das ist eine Vermutung, kein Ergebnis.

**2. `topic_own` der Prinzessin und der Matriarchin messen nicht dasselbe.**
Die Prinzessin erreichte ihre 20/30 mit `echo_pct` im zweistelligen Bereich —
Prompt-Echo zählt in `topic_own` als Treffer, `compare_inference.sh` warnt
selbst davor. Die Matriarchin hat `echo_pct` 0,0 bis 0,1 über alle drei
Epochen: sie greift kein einziges Wort aus dem Prompt auf. Ihre 0/30 sind
deshalb strenger als die 20/30 der Prinzessin gut sind. Ein direkter Vergleich
der Absolutwerte über die Stufen hinweg trägt nicht; die Richtung innerhalb
jedes Laufs trägt.

**3. Ein Wert von 2,00 im T=0,90-Arm ist keiner.** `topic_lift` steht dort in
E2 auf 2,00 und sieht nach Gewinn aus — er steht auf `own=0/6` und
`foreign=0/6` und ist damit reine additive Glättung (0,5 / 0,25). Der ganze
T=0,90-Arm hat n=6 je Epoche und trägt hier nichts.

**4. n=30 je Epoche, geclustert über 3 Prompts.** Die Prompt-Ebene ist die
eigentliche Einheit, das effektive n ist also kleiner als 30. Für die Richtung
reicht es: der Computer-Prompt geht 0 → 3 → 7 leere Läufe, beide offenen
Prompts gehen unabhängig voneinander auf 0/10.

**5. Die Kontextsonde bleibt uninformativ.** 0/50 an allen drei Messpunkten,
Namensdichte 0,00 — das bestätigt Befund (c), unterscheidet die Epochen aber
nicht. Eine Bedingung, die überall null ist, trennt nichts.

## Einordnung und Konsequenz

**Die Zwei-Phasen-SFT-Anordnung wird auf Matriarchin-Ebene vorerst nicht
weiterverfolgt.** Sie liefert Formathygiene (`<unk>` und `[SYMBIO]`-Leck auf
null, Strichkette weg) und kostet dafür Themenbindung, Wortschatzvielfalt und
Antwortvielfalt — bei einer Definitionsfrage kostet sie die Antwort ganz.

Der nächste sinnvolle Test ist **nicht** eine dritte Epoche und **nicht** die
nächste Stufe, sondern die Trennung der beiden plausiblen Ursachen. Am
billigsten wäre die Materialseite: dieselbe Anordnung mit dem Vier- bis
Fünffachen an SFT. Das hängt allerdings an offenem Punkt 1 — seit der
Lang-Modus-Umstellung vom 27.08. liegt neues SFT-Material hinter dem
Ctx-≥-2048-Gate, und für die Matriarchin (Ctx 2048) ist genau das der Vorteil:
sie kommt an das Langmaterial heran, das die Prinzessin nicht erreicht.

Die Basis-Hypothese ließe sich ohne neues Material prüfen, indem `1fdf68`
zuerst über E3 hinaus auf seine sechs Epochen fertig trainiert und die
SFT-Phase dann wiederholt wird — das ist zugleich der direkte Test für
offenen Punkt 6.

## Reproduktion

```
experiments/matriarchin_sft_finetune/
  run_finetune.sh        Training (2 Epochen, LR 0.00005, WD 0.01, Dropout 0.1, LS 0.1)
  compare_inference.sh   Verhaltensmessung:  ./compare_inference.sh 0 1 2
  base/base.apex         1fdf68 E3, unveraendert   03e317f401bb69f9…
  results/sft_e1.apex    e3a6f1d7b9b6c732…
  results/sft_e2.apex    748d6e2561269bd2…
  results/summary.txt    Held-out + Kontextsonde je Epoche
  results/inferenz_e{0,1,2}_kennzahlen.txt
  results/inferenz_e{0,1,2}_T0{25,90}.txt        Rohausgaben
  data/CHECKSUMS.txt     eingefrorenes Material
```

Quervergleich auf derselben Stufe, gleiches Protokoll, **anderes Modell**
(GEN-Matriarchin `f50702`, `INFERENZ_MATRIARCHIN_F50702_REFERENZ.md`):
`topic_own` 2/30, `topic_lift` 1,11, `new_typ` 6,8, `unk` 183. Sekundär — die
Kontrolle dieses Laufs ist und bleibt E0.

**Zwei Anpassungen am Auswertungsskript**, beide beim Kopieren aus
`princess_sft_fresh_e2/` nötig geworden und beide von der Sorte, die stumm
falsche Zahlen erzeugt hätte:

1. **Epoche 0 zieht `base/base.apex`**, nicht `results/sft_e0.apex`. Ohne den
   Vorher-Wert wäre nicht zu trennen, ob SFT etwas bewirkt hat oder ob eine
   Matriarchin ohnehin so antwortet — dieselbe Begründung, mit der
   `run_finetune.sh` die E0-Referenz auf beiden Holdouts mitgemessen hat. Beim
   Computer-Prompt ist genau das der Punkt: 0/10 steht schon in E0.
2. **Kein `NUM_HEADS`-Default mehr.** Die Vorlage stand auf 4 und hätte den
   Wert bei fehlender Sidecar-Datei einer 8-Head-Matriarchin aufgeschrieben.
   Schlimmer: `resolve_num_heads()` in `head_config.php:52` fällt ohne Sidecar
   auf **1 Head** zurück *und schreibt das dauerhaft fest*. Beide Wege liefern
   Ausgaben, die wie ein kaputtes Modell aussehen, obwohl nur die Geometrie
   falsch geladen wurde — dieselbe Fehlerklasse wie der Checkpoint-Fund vom
   29.08. Das Skript bricht jetzt ab, statt zu raten.
