# Referenz-Inferenz Matriarchin `f50702` (2026-08-27) — Abbruchdokumentation

**Zweck:** Dieser Lauf wurde am 27.08. **bewusst abgebrochen**, nicht weil er
scheiterte, sondern weil vorher nachgerechnet wurde, dass er die Frage, für die
er hätte dienen können, **grundsätzlich nicht beantworten kann**. Diese Datei
hält den Zustand bei Abbruch fest, damit die Stufe nicht ganz undokumentiert
bleibt und ein späterer, fairer Lauf einen Vergleichspunkt hat.

**Modell:** `factory/apex/matriarchin_GEN_f50702.apex`, Stand Ende Epoche 1
(2026-08-27, 14:18), 327 235 616 B, sha256 `1ff7b256a1bd8a11…`
512 Dim | Ctx 2048 | 6 Layer | 8 Heads.
Epoche 2 lief bei Abbruch bis Step ~900 von ~2490 und wird verworfen.

**Protokoll:** `experiments/matriarchin_ref/compare_inference.sh 1` — identisch
zu dem der beiden Prinzessinnen-Arme, inklusive der am 27.08. ergänzten
Inhaltstiefe-Kennzahlen. Damit direkt vergleichbar.

## Warum abgebrochen: die Schwelle ist unerreichbar

Am 27.08. wurde gemessen, dass Prompt-Bindung bei 256 Dim schwellwertartig
einsetzt — der SFT-Prinzessinnen-Arm sprang zwischen Epoche 6 und 8 von 3/30
auf 29/30. Die dafür nötige kumulierte SFT-Menge lag bei **rund 200 MB**.

| | Prinzessin (SFT-Arm) | Matriarchin `f50702` |
|---|---|---|
| SFT-Anteil nach Bytes | 77,8 % | **46,8 %** |
| Material je Epoche | ~32 MB | 15,0 MB |
| geplante Epochen | 13 | **6** (`EVOLUTION_LADDER`) |
| kumuliertes SFT am Ende | 199,8 MB (bei E8) | **42,1 MB** |

`f50702` hätte über ihren **gesamten** Lauf 42 MB SFT gesehen — Faktor 4,7 unter
der Schwelle, und das bei 4,2× mehr Parametern, die sie eher anheben als senken.
Der Lauf konnte die Kapazitätsfrage also nicht beantworten, gleich wie er
ausgegangen wäre. Weiterlaufen hätte ~4 Tage Rechenzeit für ein Ergebnis
gekostet, das schon vorher als nicht aussagekräftig feststand.

## Kennzahlen (T=0,25, n=10 je Prompt, 3 Prompts)

| Kennzahl | `f50702` E1 | Prinzessin SFT E8 | Prinzessin CHIN E15 |
|---|---|---|---|
| topic_own | 2/30 | 29/30 | 5/30 |
| topic_lift | 1,11 | 7,87 | 3,14 |
| noun_ttr | **0,97** | 0,24 | 0,24 |
| echo_pct | 0,5 | 27,1 | 5,3 |
| new_typ | 6,8 | 4,8 | 10,0 |
| taut_pct | **0,0** | 20,0 | 20,0 |
| eos_ok | 29/30 | 18/30 | 12/30 |
| **unk** | **183** | **0** | **0** |
| avg_chars | 174,7 | — | 285,9 |

Bei T=0,90 (n=3, 2 Prompts): topic_own 0/6, lift 0,40, unk 0, eos 6/6.

## Was die Zahlen bedeuten — und wo sie täuschen

`noun_ttr` 0,97 und `taut_pct` 0,0 sehen auf den ersten Blick nach dem besten
Wert aller drei Modelle aus: keine Wortschleifen, keine Tautologien. Der Blick
in den Rohtext korrigiert das sofort:

> „Die Anordnung der Andrung der Zeit des besten Staaten. Der erste Analyse der
> ersten der beiden Täherigen sich die Anzahl der Welt und die die Verteilung
> von der Regelnacken-Standar."

> „Die Andrökung der Limischen und die Stadt der Predigen, der er als auch die
> Hamburger Fays."

Das Modell wiederholt sich nicht, **weil es fortlaufend neue Wörter erfindet** —
„Andrung", „Täherigen", „Regelnacken-Standar", „Andrökung", „Limischen",
„Fays". Es ist die Stufe vor der Wortbildung, nicht darüber.

**Daraus folgt eine Grenze der neuen Metrik, die festgehalten gehört:
`new_typ` kann ein echtes neues Substantiv nicht von einem halluzinierten
unterscheiden.** Bei einem Modell in diesem Zustand zählt es Kunstwörter als
Inhalt und weist 6,8 aus — mehr als der SFT-Arm mit 4,8, der tatsächlich
sinnvolle Substantive produziert. Die Kennzahl ist nur innerhalb eines Bereichs
gültig, in dem die Wortformen stimmen; darunter zeigt sie das Gegenteil dessen,
was sie messen soll. Für spätere Vergleiche wäre entweder ein Vokabular-Abgleich
nötig oder eine Vorbedingung „nur auswerten, wenn `unk` niedrig".

`unk=183` ist der eigentlich aussagekräftige Wert: **der `<unk>`-Kollaps
existiert auf dieser Stufe, während beide Prinzessinnen über alle 15 bzw. 10
Epochen und alle 450 Läufe null zeigen.** Das passt zur dokumentierten
Matriarchin-Historie (`INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md`), wo derselbe
Kollaps zwischen Epoche 1 und 2 von selbst verschwindet. Der Wert ist also
Epoche-1-typisch und **kein** Argument gegen die Stufe.

## Einordnung

Der Lauf ist als Vergleichspunkt nur eingeschränkt brauchbar: Epoche 1 gegen
Epoche 8 bzw. 15 der Prinzessinnen ist kein Vergleich auf Augenhöhe. Was er
belegt, ist ausschließlich der Startzustand der Stufe unter dem alten
Datenregime — und dass dieses Regime die Schwelle nicht erreichen konnte.

Der Checkpoint wird unter `secured_models/` abgelegt. Ein fairer Test verlangt
einen Neuaufbau mit deutlich höherem SFT-Anteil und genug Material, um über
~200 MB kumuliertes SFT zu kommen; die Größenordnung liegt bei rund einer Woche
Rechenzeit auf dieser Maschine (~11 h je Epoche bei 15 MB).
