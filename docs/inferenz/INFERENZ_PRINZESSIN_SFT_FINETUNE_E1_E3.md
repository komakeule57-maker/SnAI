# SFT-Fine-Tune E1–E3 — der gesamte Ertrag fällt in Epoche 1, danach ist es ein Verlustgeschäft

Auswertung am 2026-08-29, 12:20 CEST. Der am 27.08. 19:06 gestartete Lauf
`experiments/princess_sft_finetune/` ist am 29.08. 09:19 nach drei Epochen
fertig geworden. Basis ist `chinchilla_e15` (sha256 `0f5aa7f0491bf60f…`,
byte-gleich gegen `experiments/princess_chinchilla/results/princess_e15.snai`
geprüft), **36,3 MB** reines SFT-Material (38 048 892 B, 91 981 Beispiele,
sha256 `9239976e4a264292…`), Spitzen-LR 0,00005 (ein Zehntel der
Vortrainings-Spitze), dieselben Daten in allen drei Epochen.

**Die Lernrate war nicht konstant.** `train_cluster.php:419–429` fährt Warmup
plus Cosine-Decay mit einem Boden bei 10 % der Spitze; bei drei
Schedule-Epochen ergibt das effektiv **E1 0,000050 → E2 0,000025 →
E3 0,000005**. Das ist für die Deutung unten wesentlich.

## Die Frage

`princess_chinchilla` mischt SFT zu 14,7 % in den Vortrainingsmix. Dieser
Lauf ändert genau eine Sache: er hängt eine **getrennte SFT-Phase** hinten
an, statt zu mischen. Die Frage war, ob die Zwei-Phasen-Anordnung die
Prompt-Bindung besser trägt als der Mix — und wie viele Epochen sie braucht.

## Kernergebnis

| | **Basis e15** | **E1** | E2 | E3 |
|---|---|---|---|---|
| `topic_own` | 5/30 | **20/30** | 10/30 | 11/30 |
| `topic_foreign` | 3/30 | 3/30 | 5/30 | 7/30 |
| `topic_lift` | 3,14 | **11,71** | 3,82 | 3,07 |
| `distinct_run_starts` | 5/30 | **14/30** | 9/30 | 13/30 |
| `register_lock` | 15/30 | **9/30** | 12/30 | 10/30 |
| `eos_ok` | 12/30 | **29/30** | 29/30 | 29/30 |
| `[SYMBIO]`-Leck | 10 | **0** | 0 | 0 |
| `unk` | 0 | 0 | 0 | 0 |
| `avg_chars` | 285,9 | 132,1 | 122,9 | 129,8 |
| Held-out GEN | 3,4579 | 3,6045 | 3,6307 | 3,6336 |
| Held-out SFT | 3,6364 | 3,4998 | 3,4565 | **3,4480** |

Eine Epoche vervierfacht die Themenbindung, räumt das `[SYMBIO]`-Leck weg und
bringt den `</s>`-Stopp von 12/30 auf 29/30. **Epoche 2 gibt den
Themengewinn fast vollständig zurück** — `topic_lift` steht in E3 bei 3,07
und damit auf dem Basiswert 3,14.

## Loss und Verhalten laufen hier gegeneinander, nicht nur auseinander

Der SFT-Holdout verbessert sich über alle drei Epochen monoton
(3,6364 → 3,4998 → 3,4565 → 3,4480). Die Prompt-Bindung halbiert sich über
dieselben Epochen. Das ist die schärfste Ausprägung der seit dem 19.08.
wiederholt beobachteten Loss/Verhalten-Divergenz: bisher hieß der Befund
„der Loss sagt über das Verhalten wenig", hier zeigen beide über denselben
Zeitraum in **entgegengesetzte** Richtungen. Nach dem Held-out wäre Epoche 3
die beste; nach dem Verhalten ist sie die zweitschlechteste.

Passend dazu: der Gewinn im SFT-Holdout fällt über die drei Epochen um
Faktor 16 (0,1367 → 0,0433 → 0,0085), das GEN-Vergessen bremst parallel mit
(+0,1466 → +0,0263 → +0,0029). Nach Epoche 1 ist der Handel im Wesentlichen
abgeschlossen — 0,147 GEN gegen 0,137 SFT getauscht, alles Weitere ist
Feinschliff auf der dritten Nachkommastelle.

## Der Aggregatwert verdeckt die Struktur

Aufgeschlüsselt nach Prompt fällt der Verlust nicht gleichmäßig an:

| Prompt | Basis e15 | E1 | E2 | E3 |
|---|---|---|---|---|
| „Was ist ein Computer?" | 3/10 | 10/10 | **10/10** | **10/10** |
| „Erzähl mir etwas über die Natur." | 2/10 | 5/10 | **0/10** | 1/10 |
| „Beschreibe einen sonnigen Tag." | 0/10 | 5/10 | **0/10** | 0/10 |

Der definitorische Prompt steht ab E1 am Anschlag und bleibt dort. Die
beiden **offenen** Prompts fallen auf null. Das Modell rastet in ein
enzyklopädisches Stichwort-Register ein: `register_modal_prefix` lautet ab
E2 „ Ein klassisches Beispiel ist", prompt-unabhängig.

Dass der Einbruch formgebunden ist und nicht global, ist derselbe
Befundtyp wie beim Drohnen-Weight-Tying-A/B vom 29.08. (Kollaps 5/20 auf
„Was ist X?", 0/80 sonst) — dort brach die definitorische Form, hier hält
sie als einzige. Der Merker vom 20.08. gilt weiter: **Aggregatmetriken über
gemischte Prompt-Formen sind irreführend, solange die Aufschlüsselung
fehlt.** Nur nach `topic_own=10/30` hätte dieser Lauf wie eine gleichmäßige
Verschlechterung ausgesehen.

## Qualitativ

**Basis e15** antwortet prompt-unabhängig in generischem Assistenz-Register.
Beide Beispiele sind Antworten auf „Was ist ein Computer?":

> „Die Überlegungen zur Lösung des Problems erfordern eine sorgfältige
> Analyse der gegebenen Einschränkungen und die Anzahl der verfügbaren
> Ressourcen."
>
> „Die Herausforderung besteht darin, die Anzahl der verfügbaren Ressourcen
> für jedes Projekt zu wählen. […]"

**E1** liefert auf dieselbe Frage einen strukturell echten Definitionssatz —
inhaltlich falsch, aber erkennbar eine Antwort auf die gestellte Frage:

> „**Ein Computer ist** eine Software, die sich auf den Computer-System
> spezialisiert hat."
>
> „**Ein Computer ist** eine Software-Server-Band, die in der Regel ein
> System auf dem Internet auftritt."

**E2** beantwortet die offenen Prompts mit thematisch entkoppelten
Wikipedia-Stubs. Links steht jeweils der Prompt:

> „Beschreibe einen sonnigen Tag." → „Ein klassisches Beispiel ist ein
> kleiner, hoher Berg im ostantarktischen Viktorialand."
>
> „Erzähl mir etwas über die Natur." → „Der Name des Gedichts ist ein sehr
> selten vorkommendes Mineral aus der Gruppe der Bakterien."

Formal saubere Sätze, sauberer Stopp, kein `<unk>`, kein Formatleck — und
thematisch vollständig entkoppelt. Passend dazu steigt `topic_foreign`
monoton (3 → 5 → 7): das Modell driftet in Wikipedia-Prosa ab.

## Was diese Zahlen NICHT hergeben

**1. Die 10/10 beim Computer-Prompt sind kein nachgewiesenes Verständnis.**
Das Wort „Computer" steht im Prompt, und `topic_own` zählt Prompt-Echo als
Treffer — `compare_inference.sh` warnt an der Stelle selbst davor. Die
Sättigung beim geschlossenen Prompt ist mindestens teilweise
Formübereinstimmung. Belastbar ist der **Abfall** bei den offenen Prompts,
nicht das Halten beim geschlossenen.

**2. Die Wortlisten matchen als Teilzeichenketten und sind großzügig.** Der
E1-Treffer „Ein klassisches Beispiel ist eine Stunde, die in der Regel ein
paar **Tage** vorkommt" zählt über das Stichwort „Tag", ohne inhaltlich
etwas mit einem sonnigen Tag zu tun zu haben. Das Absolutniveau von
`topic_own` ist dadurch nach oben verzerrt. Der **Vergleich** über die
Epochen bleibt gültig, weil dieselbe Liste auf alle vier Spalten angewandt
wird.

**3. n=30 je Epoche, geclustert über 3 Prompts.** Ein Zwei-Proportionen-Test
E1 gegen E2 (20/30 gegen 10/30) ergäbe p≈0,01, unterstellt aber
Unabhängigkeit der 30 Läufe. Die Prompt-Aufschlüsselung zeigt, dass die
Bindung an die Prompt-Ebene die eigentliche Einheit ist — das effektive n
ist kleiner und der wahre p-Wert größer. Für die Richtung reicht es: E2 und
E3 stimmen unabhängig voneinander überein (10 und 11 von 30), E1 ist der
Ausreißer.

**4. T=0,90 trägt hier nichts.** Mit n=6 je Epoche
(Basis 2/6, E1 2/6, E2 4/6, E3 3/6) ist kein Muster ablesbar. Der
Methodik-Merker vom 20.08. — n=1 trägt für Verhaltensaussagen nicht — gilt
für n=6 fast genauso.

## Einordnung und Konsequenz

Die Zwei-Phasen-Anordnung **funktioniert**, und zwar sofort: E1 erreicht
`topic_lift` 11,71 gegen 10,80 beim SFT-Mix-Kontrolllauf E3
(`INFERENZ_SFTMIX_KONTROLLLAUF_E1_E3.md`) — vergleichbares Niveau, aber in
einer statt drei Epochen und ohne den dortigen GEN-Verlust durch einen
dauerhaft veränderten Mix.

**Konsequenz für den Einsatz: eine SFT-Epoche, nicht drei.** Der Held-out
hätte drei empfohlen.

Zu „nur E1 war positiv" gehört eine Einschränkung: die **Formathygiene** aus
Epoche 1 bleibt über E2 und E3 vollständig erhalten (`eos_ok` 29/30 statt
12/30, `[SYMBIO]`-Leck 0 statt 10). Die beiden Zusatzepochen zerstören also
nicht alles — sie kosten die Themenbindung und liefern dafür nichts.

**Die naheliegende Erklärung „zu hohe Lernrate" scheidet weitgehend aus.**
E2 lief bereits bei halber, E3 bei einem Fünftel der Spitzen-LR — der
Einbruch entstand unter 0,000025 und wurde unter 0,000005 nicht repariert.
Bei 0,000005 bewegt sich das Modell zudem kaum noch (SFT-Holdout-Gewinn
0,0085 gegen 0,1367 in E1), eine noch kleinere Lernrate liefe also auf „es
passiert nichts" hinaus, nicht auf „der Einbruch bleibt aus".

**Nachtrag 2026-08-30 — die offene Frage ist beantwortet.** Ob der Einbruch
von der *Wiederholung* kommt (E2 und E3 sehen dieselben 36,3 MB ein zweites
und drittes Mal), hat der Kontrolllauf `experiments/princess_sft_fresh_e2/`
geprüft: gleiche Basis `sft_e1`, gleicher LR-Schedule, aber ungesehenes
SFT-Material. **Nein** — mit frischem Material steht `topic_own` bei 11/30
gegen 10/30 mit wiederholtem, beide weit unter den 20/30 aus E1. Die
SFT-Phase sättigt nach einer Epoche unabhängig vom Material.

Die Wiederholung ist aber für einen **zweiten, davon trennbaren** Schaden
verantwortlich: frisches Material verdreifacht `distinct_run_starts`
(9 → 24), drittelt `register_lock` (12/30 → 4/30) und räumt `topic_foreign`
ganz weg (5/30 → 0/30). Themenbindung verliert die zweite Epoche in jedem
Fall, Stereotypisierung kommt zusätzlich vom Wiederkäuen.

## Reproduktion

```
experiments/princess_sft_finetune/
  run_finetune.sh        Training (3 Epochen, LR 0.00005, WD 0.01, Dropout 0.1, LS 0.1)
  compare_inference.sh   Auswertung:  ./compare_inference.sh 1 2 3
  results/summary.txt    Held-out je Epoche
  results/inferenz_e{1,2,3}_kennzahlen.txt
  results/inferenz_e{1,2,3}_T0{25,90}.txt     Rohausgaben
```

Vergleichswerte der Basis:
`experiments/princess_chinchilla/results/inferenz_e15_kennzahlen.txt`.

**Fallstrick im Auswertungsskript, am 29.08. gefixt:** `compare_inference.sh`
war aus `princess_chinchilla` kopiert und suchte weiterhin
`results/princess_eN.snai`; dieser Lauf schreibt `results/sft_eN.snai`.
`snai_inference.php` lieferte daraufhin 108-mal die 46 Zeichen lange Zeile
`System-Fehler: Cortex-…`, und `tally()` wertete die zu einer vollständig
aussehenden Kennzahlentabelle aus (`topic_own=0/30`, `register_lock=30/30`,
`avg_chars=46.0`) — also zum Bild eines total kollabierten Modells. Das
Skript hat jetzt einen Existenz-Wächter, der stattdessen abbricht. **Ein
fehlendes Modell darf nicht wie ein kaputtes Modell aussehen** — die
Kennzahlen sind erst dann Zahlen, wenn geprüft ist, dass überhaupt ein
Modell geladen wurde.
