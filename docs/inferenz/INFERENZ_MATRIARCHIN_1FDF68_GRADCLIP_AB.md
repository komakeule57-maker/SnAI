# A/B-Inferenztest Matriarchin: Gradient-Clipping gefixt vs. Bug

Erster Matriarchin-Checkpoint,
der komplett unter dem Gradient-Clipping-Fix vom 18.08. entstanden ist,
gegen die Bug-Referenz derselben Stufe und **derselben Epochenzahl**.

## Testaufbau

- **Gefixt:** `secured_models/matriarchin_GEN_1fdf68_epoch1.apex` — Kopie von
  `factory/checkpoints/best_matriarchin_GEN_1fdf68.snai`, unmittelbar nach
  Epoche 1 gesichert (sha256 `d735feed8304b0b4…`, byte-identisch zur
  `factory/apex/`-Datei verifiziert, die beim Abschluss von Epoche 2
  überschrieben wird). Held-out-Loss **5.486806** (PPL 241.48), Train-Loss
  5.80172, Epoche 1 von 6, Laufzeit 17:45 → 03:33 (~9h 48m), GPU CUDA.
- **Bug-Referenz:** `secured_models/matriarchin_GEN_1557a2_epoch1.apex`.
  Held-out-Loss **5.347133** (PPL 210.01), ebenfalls Epoche 1 von 6.
  Vollständige Erstbeschreibung: `INFERENZ_MATRIARCHIN_1557A2_EPOCHE1.md`.
- **Beide:** 512 Dim, 6 Layers, 8 Heads, GEN-Kalibrierung.
- **Inferenz-Parameter:** unverändert die Projektstandards aus
  `snai_inference.php` (Temperature 0.25, Repetition Penalty 1.15 / Window
  64, min_p 0.05, max. 256 neue Tokens, kein Beam-Search) — gegen die
  Quelldatei geprüft, identisch zu allen bisherigen Referenztests.
- **Stichprobe: 8 Läufe je Prompt je Checkpoint, 3 Standard-Prompts = 24
  Läufe pro Modell.** Bewusst mehr als die bisherigen n=1-Referenztests:
  bei Temperature-Sampling ist ein Einzellauf pro Prompt für eine
  Verhaltensaussage zu dünn (ein erster n=1-Durchgang hätte hier zu einer
  deutlich zu starken Schlussfolgerung geführt).
- **Wichtig — beide Seiten hier neu gemessen:** die Bug-Referenz wurde
  **nicht** aus dem alten Dokument übernommen, sondern am 20.08. unter
  exakt denselben Bedingungen frisch gefahren. Das eliminiert den
  Backend-Unterschied (die 18.08-Messung lief unter OpenCL, hier läuft auf
  beiden Seiten CUDA) und den Unterschied im Engine-Build.

## Messergebnis (je 24 Läufe)

| Checkpoint | Held-out | Ø Zeichen | Ø ohne Whitespace | Whitespace-Anteil | `<unk>` gesamt | Läufe mit `<unk>` |
|---|---|---|---|---|---|---|
| **1fdf68 E1 (gefixt)** | 5.486806 | 153.1 | 116.6 | 23.8 % | **1** | **1 / 24** |
| **1557a2 E1 (Bug)** | 5.347133 | 377.2 | 298.8 | 20.8 % | **788** | **13 / 24** |

Der `<unk>`-Unterschied ist groß und weit außerhalb dessen, was
Sampling-Rauschen bei n=24 erklären könnte: 1 Vorkommen in einem einzigen
Lauf gegen 788 Vorkommen in 13 von 24 Läufen.

## Qualitative Beispiele

**Gefixt (1fdf68 E1)** — kurz, terminiert sauber, erfundene Komposita,
kein `<unk>`:

> „Um der „Bie „Der Name des Landes- und „Mari“, die in der Prokung von den
> Ganzs. / Am 1. Juni 2012 wurde er als erste Tipps in der Saison 2015 in
> der Saison 2006/2011.“

**Bug (1557a2 E1)** — läuft ins Token-Limit, kippt in `<unk>`-Ketten:

> „Natürlich! / - 1. / -1.  [langer Leerzeichen-Lauf] `<unk>`. `<unk>`.
> `<unk>`. `<unk>`. …“ (bis zum Limit)

## Der Befund — und was ihn sofort wieder relativiert

**Auf den ersten Blick ein klarer Punktsieg für den Fix:** bei gleicher
Stufe und gleicher Epochenzahl vermeidet der gefixte Checkpoint den
`<unk>`-Kollaps praktisch vollständig, der die Bug-Referenz dominiert. Und
das, obwohl sein Held-out-Loss **schlechter** ist (5.4868 vs. 5.3471) —
ein weiterer, diesmal besonders sauberer Beleg für die schon mehrfach
dokumentierte Loss/Verhalten-Divergenz.

**Der Vergleich mit den Epochen 2 und 3 der Bug-Linie (zeitgleich
gemessen, siehe `INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md`)
entzieht dieser Deutung aber einen erheblichen Teil ihrer Kraft:**

| Checkpoint | Held-out | `<unk>` gesamt | Läufe mit `<unk>` |
|---|---|---|---|
| 1557a2 **E1** (Bug) | 5.347133 | 788 | 13 / 24 |
| 1557a2 **E2** (Bug) | 5.072810 | **0** | **0 / 24** |
| 1557a2 **E3** (Bug) | 4.651470 | 241 | 2 / 24 |
| 1fdf68 E1 (gefixt) | 5.486806 | 1 | 1 / 24 |

Der `<unk>`-Kollaps verschwindet in der Bug-Linie **von selbst** zwischen
Epoche 1 und 2 — ganz ohne Fix. Er ist damit primär ein Artefakt eines
sehr frühen Trainingsstands, kein Signatur-Symptom des Bugs. Der gefixte
Checkpoint sieht bei Epoche 1 aus wie die Bug-Linie bei Epoche 2+.

Das ist **vereinbar** mit „der Fix beschleunigt die Verhaltensreifung",
**beweist es aber nicht** — denn der gefixte Lauf unterscheidet sich noch
in mindestens drei weiteren Punkten von der Referenz:

1. **Andere Kalibrierungsdaten.** Jede Epoche zieht frischen Nektar per
   Zufall; die beiden Läufe haben nie dasselbe Material gesehen.
2. **Sauberer Korpus.** `1fdf68` trainierte, nachdem die 154
   fehlformatierten `hf_SFT_*.txt` am 19.08. um ~04:52 quarantäniert waren,
   `1557a2` davor. Dieser Konfund zeigt in **dieselbe** Richtung wie das
   Ergebnis und lässt sich nicht abtrennen.
3. **Andere Abstammung.** `1fdf68` ist aus Prinzessinnen hervorgegangen,
   die selbst unter dem Fix trainiert wurden, `1557a2` aus bug-trainierten.
   Der Test misst also „ganze Pipeline unter dem Fix" gegen „ganze Pipeline
   unter dem Bug", nicht den isolierten Effekt auf der Matriarchin-Stufe.
   Für die ursprüngliche Hypothese (vererbte Net2Net-Gewichte können
   sich unter dem Bug nicht sauber ausgleichen) ist genau das der
   *richtige* Vergleich — als Nachweis eines reinen Optimierer-Effekts ist
   es der falsche.

**Ein Konfund wirkt ausdrücklich gegen das Ergebnis** und stärkt es damit:
`1557a2` bekam nach Epoche 1 automatisch **9 tote Neuronen revivt** (2×
`L3_w_down`, 7× `L5_w_down`), `1fdf68` bekam nach Epoche 1 **keinen einzigen
Revival-Eingriff** (im Log geprüft). Die Bug-Referenz hatte also eine
zusätzliche Reparaturmaßnahme und schnitt trotzdem schlechter ab.

## Was der Test NICHT zeigt

Der gefixte Checkpoint ist **nicht** sprachlich besser als die späteren
Bug-Epochen. Er ist nur kürzer und sauberer: Ø 153 Zeichen gegen Ø 274 bei
`1557a2` E3, und die E3-Ausgaben sind erkennbar flüssiger (teils
vollständig wohlgeformte deutsche Sätze, siehe `INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md`). Wer
aus diesem Test „der Fix macht bessere Modelle" liest, überdehnt ihn — er
zeigt „bei gleichem Epochenstand weniger Zerfallsartefakte", mehr nicht.

Der Whitespace-Kollaps (Generierung kippt in reine Leerzeichen bis zum
Token-Limit) existiert auf **beiden** Seiten weiter: 1 von 24 Läufen beim
gefixten Checkpoint, und mit 34.8 % Whitespace-Anteil ist ausgerechnet
`1557a2` **E2** hier der schlechteste aller vier gemessenen Checkpoints.
Der Fix beseitigt diese Fehlerklasse also nicht.

## Einordnung

Der A/B ist damit abgeschlossen, aber mit einem schwächeren Ergebnis als
die reine Epoche-1-Gegenüberstellung suggeriert. Ein wirklich sauberer
Nachweis bräuchte zwei Läufe, die sich **ausschließlich** in der
Clip-Schwelle unterscheiden — gleiche Kalibrierungsdateien, gleicher
Korpusstand, gleiche Elternmodelle. Das ist mit der laufenden Pipeline
nicht nebenbei zu bekommen; es wäre ein dedizierter, doppelt gefahrener
Kurzlauf auf einer kleineren Stufe (Drohne oder Prinzessin), bei dem der
Nektar einmal fixiert und beide Varianten darauf trainiert werden.

Wert des Tests trotzdem: die Sorge, der Fix könnte das Training
verschlechtern, ist ausgeräumt — trotz nominell schlechterem Held-out-Loss
ist das beobachtbare Verhalten bei gleichem Epochenstand nicht schlechter,
sondern in der auffälligsten Fehlerklasse deutlich besser.
