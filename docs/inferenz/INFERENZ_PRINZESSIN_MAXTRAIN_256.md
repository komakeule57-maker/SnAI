# Inferenz-Test: Prinzessin, 256-Dim-Maximaltraining (Epoche 25/40)

Zwischenstand-Test des laufenden Laptop-Maximaltrainings (`longtrain_256`)
— Genesis-Modell ohne Net2Net-Vorgeschichte, voller
24-MB-Korpus, 40-Epochen-Warmup+Cosine-Decay-Lauf. Anders als
`docs/inferenz/INFERENZ_BASELINE.md` (Schwarm-trainierte SFT-Prinzessin, kurze
Kalibrierung) geht es hier um dieselbe Architektur unter deutlich längerem,
gezieltem Training — Zwischenstand, kein Endergebnis (Lauf läuft weiter bis
Epoche 40).

## Testaufbau

- **Modell:** `factory/checkpoints/cp_trained_256_e40_e25.snai` (Laptop)
- **Trainings-Loss:** Epoche 25 Avg Loss **3.663391** — reiner
  Trainings-Fit-Wert, kein Held-out-Tracking bei diesem Lauf (bei diesem
  Tokens/Parameter-Verhältnis (~1,3) vermutlich überwiegend Memorierung).
- **Herkunft:** Genesis-Modell (kein Net2Net-Rauschen), PID 819792 seit
  2026-08-07, alter Codestand ohne Batch-Shuffling (bewusst, für einen
  späteren Shuffling-A/B-Vergleich).
- **Architektur:** 256 Dim, 2 Layer, 4 Heads.
- **Inferenz-Parameter** (`snai_inference.php`, unverändert): Temperature
  0.25, Repetition Penalty 1.15, max. 256 neue Tokens.
- **Hardware/Pfad:** Laptop, AMD Renoir APU (OpenCL-Fallback, RustiCL).
- **Datum:** 2026-08-11

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (751 Zeichen)
```
Skills, die sich mit dem „Noder“ („Schlaus-Schlaus) ist eine ehemalige Weltbewegung im Jahrgang, wo er sich in der Lage zu verstehen und sie in den USA gebracht wurde. In den Jahren wurde sie als „Das Film“ ausgezeichnet.

In der 1930er Jahre wurde sie in der Tradition des Films“.

Als sie in der Filmmusik, als „The“ und „The“.

Im Februar 2017 war sie in der Filmmusik in der Filmmusik, der Filmmusik in der Filmmusik in der Filmmusik in der Filme und Filmmusik.

Marcus Ross (* Filmfilm) in der Filmmusik (1971), in der Filmmusik in der Filmmusik, in der Filmmusik)

Während der Film (1867–174)
Karla (1905–1915)
Christ (* 1962), deutscher Film (1867)
John Mittern (* 1963)

Patrick Collar (* 1770)
Filma (* 1949), deutscher Schauspieler

Lou
```

### Prompt 2: „Was ist ein Computer?“ (887 Zeichen)
```
ontheidigt sich die meistenkleidung, die in der Physikalender, die als auch von den Mantelier des AI-Theor-Abschluss-Kriebe und die in der Physik-Systeme.

Diese der Gelegenheit ist eine französische Gemeinde im Südamerika. Die ehemalige Kirche wurde mit einem Jahr nach dem Zweiten Weltkrieg, der von der Provinzial-Weltmeisterschaften (Niederländischer) und der Stadtbahn (1876–1982).

International Bridge war ein deutscher Physiker und der Stadt.

1994 wurde der Stadtgemeinde Ross (1954–1951), der von der Stadt Nathania, der im Senat, der an der Universitätsbibliothek, der im Landkreis Weimar von der Stadt. Im Jahre 1950 wurde die Stadt der Stadt im Landkreis Az.

Zur Gründung der Stadt wurde das Dorf im Landkreis Kalifornien und der Stadt Baden-Württemberg, der die Stadtverwaltung in der Stadt.

1973 wurde nach der Stadt Maude-Hochemeister der Stadt.

1976 wurde die Stadt
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (1107 Zeichen)
```
Der Wegformationen der Wydawnictwo M, der sich mit dem Muba<unk>ani<unk>-Spie ist und die Mu<unk>a-Analytics. In den Jahren wurde er zum Teil des Musters von M<unk>ozky<unk>o (Leo) und Mio<unk>o (<unk><unk><unk>...<unk>  [kollabiert nach ~200 Zeichen in eine lange <unk>-Wiederholungsschleife bis zum Zeichenlimit]
```

## Beobachtungen

- **Grammatikalisch deutlich flüssiger** als sowohl die kurz-kalibrierte
  SFT-Prinzessin aus `docs/inferenz/INFERENZ_BASELINE.md` als auch die erste
  Matriarchin aus `docs/inferenz/INFERENZ_MATRIARCHIN.md` — überwiegend korrekte
  deutsche Sätze statt Wortfetzen-Collage.
- **Keine inhaltliche Kohärenz zum Prompt** bei allen drei Versuchen —
  driftet sofort in Wikipedia-artige Biografie-/Orts-/Jahreszahlen-
  Aufzählungen ab, unabhängig vom Thema (Muster deckt sich mit der
  Matriarchin-Beobachtung "spürbar mehr echte Wikipedia-Struktur", hier
  aber bereits bei einem kleineren Tier).
- **`<unk>`-Kollaps-Fehlermodus** bei Prompt 3 erneut aufgetreten — derselbe
  Fehlermodus wie bei der Matriarchin dokumentiert.
- Nebenbefund, kein Blocker: `[cortex] WARN: adam_t fehlt im Checkpoint,
  m/v sind aber warm -> adam_t = 1000` — reine Inferenz-Randnotiz aus dem
  Trainings-Checkpoint-Format, betrifft nicht die Generation-Qualität.

## Einordnung

Reiner Zwischenstand bei 25 von 40 geplanten Epochen — der Lauf läuft
weiter, Fertigstellung grob am 15.08. erwartet. Bereits hier deutlich
flüssigere Sprachoberfläche als bei den kurz kalibrierten Schwarm-Modellen
gleicher Größe, aber weiterhin ohne inhaltliche Prompt-Kohärenz. Da kein
Held-out-Loss-Tracking existiert, ist unklar, wie viel davon Generalisierung
vs. Memorierung des Korpus ist. Vergleichspunkt für den finalen Test nach
Epoche 40.
