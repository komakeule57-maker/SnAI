# Inferenz-Vergleich: Matriarchin vor/nach LR-Boost-Fix

Direkter Vorher/Nachher-Vergleich auf Matriarchin-Ebene (ergänzt
`docs/inferenz/INFERENZ_VERGLEICH_LR_FIX.md`, das denselben Vergleich auf
Prinzessin-Ebene macht). Beide Matriarchinnen stammen aus der regulären
Schwarm-Pipeline auf dem Hauptrechner, gleiche Architektur, gleiches
Inferenz-Setup — der einzige kontrollierte Unterschied ist der 3x-LR-Boost
für frisch Net2Net-expandierte Parameter.

## Testaufbau

| | Vor dem Fix | Nach dem Fix |
|---|---|---|
| Modell | `matriarchin_GEN_b760a8.apex` | `matriarchin_GEN_4b0367.apex` |
| Loss Epoche 1 | 6.05171 | **6.01512** |
| Loss Epoche 2 | 5.71142 | **5.66841** |
| Loss Epoche 3 (final) | **5.56625** | 5.57051 |
| Architektur | 512 Dim, 4 Layer, 8 Heads | identisch |
| LR-Boost | nein | ja, 3x auf frisch expandierte Bereiche |
| Nectar | ~15,3 MB | ~17,6 MB (andere Prinzessinnen-Kombination) |

**Parameter:** Temperature 0.25, Repetition Penalty 1.15, max. 256 neue
Tokens (identisch zu allen bisherigen Vergleichsdokumenten).

**Zum Loss:** Epoche 1 und 2 liegen beim Boost durchgehend niedriger, die
finale Epoche 3 liegt minimal (0,008) höher als beim alten Lauf — eine
Differenz, die während des Laufs zwischenzeitlich (Step 2725: 5.573) noch
klar unter dem alten Endwert lag. Vermutlich Lauf-zu-Lauf-Rauschen (andere
Nectar-Zusammensetzung, andere Zufallswerte beim Net2Net-Padding), kein
klarer Effekt in die eine oder andere Richtung auf Loss-Ebene allein.

## Ergebnisse (roh)

### Prompt 1: „Erzähl mir etwas über die Natur.“

**Vor dem Fix** (648 Zeichen, siehe `docs/inferenz/INFERENZ_MATRIARCHIN.md`):
```
Sicher, die „bspm Jahrministerische Schätigen und in der Sie-Vertar)
Dereinigte den ersten Filme in der Gände des B<unk><unk>cyi (Ler)
[...]
```

**Nach dem Fix** (641 Zeichen):
```
In fen, der Hährung von jedoch die Musik in den ferner der Hitparischen Hakter-Bar-Platal-Bel-Kleits-Scheik.

Die Battler in der Bes „Heglisch-Manda (Nach).

Als Jahre: Die erste Tilli- und die für das Ahrigen Villa-Grsg.

1989: Theorie, ISBN 978-063-1-4-5.

Die der Kart-LVer-Württemberg.

Die der Machstation
Carl August 2011 wurde, S.

Der Jahr 2022, der Dame in der Schichten in der Bundesrepublik der Universität.

Die dem Jahr 1983, dass die Ex-Gremann (* 1815–1997), der Stadt-Bebsite des Zweiten Weltkrieg war ein.

Im Jahr 189632-Vie-Prock- und die Leuber.

Die der Kirche ist der Stadtgebiet des Zweiten Weltkriegs wurde ein neues
```

### Prompt 2: „Was ist ein Computer?“

**Vor dem Fix** (958 Zeichen): kollabierte in eine lange `<unk>`-Wiederholungsschleife bis zum Zeichenlimit.

**Nach dem Fix** (736 Zeichen):
```
In der Hoci, die Tes-Kuon-Morischen Gemeinde in den Bodatal. Die Bewertung des Landesgebiet der Zeit in der Folin von den ersten Jahren in der Puner und das „He-Wie“), die Sturzung von der Stadt-Beistation (* 17. Januar) war er eine US-Schige Jahre in der Mive für die Abgangsbachs und in der Universität der Welt. Jahrhunderts wurde die Verantwortungsjahrte.

Die der Nitierung der Saisonspielen im Jahr 2015. Jahrhundert: Der Wunderten, dass die sich als Mattliche, um die Reformierten.

Aniversität der Website der Haupthalenutschen, ISBN 978-3-42-06-9.
1976.

Die der Kital-Prock-Stand-Fürttemberg.

Die der Bundesrepublik der Vermeistung von dem Bau der Geschichte der Savik-Grsg.

Die der Deutschen Kirche, die DVer-Schel-Rar-L-W
```
**Kein `<unk>`-Kollaps diesmal** — durchgängig Wortfetzen statt Endlosschleife.

### Prompt 3: „Beschreibe einen sonnigen Tag.“

**Vor dem Fix** (575 Zeichen): Wortfetzen-Fragmente, kein `<unk>`-Kollaps.

**Nach dem Fix** (688 Zeichen):
```
Die jedoch ferner der Haken, die Musik von ferner der fährten und die Tattung des Sieischen Lachs, der Bamburger K<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>t, die Single und die in der Hestel-Prog.

Der Rünter für das Aufstieg bis 1893.

Be der Wohls-Wie-Maro-Sch-Ge-Ge-Amirte.

Nach dem ersten Jahren wurde er sich ein förderungsstehen, der Hitierung der Machs-Vas-Les-Clafle-Koli-Ein-Hres-Scharkum (* 31. September).

Die der Universität der Efalender-Stand-Fürttemberg.

Die der Stadtgebiet von den „Meteron“, um die Nalischen Puka, die DT.

Die der Hörden (19986).

In der Gemeinde der Saison 2005, dass eine unmittelbar im Jahr 2000, der Lur des Jahres 10. Jahrhunderts
```
**Kurzer `<unk>`-Ausreißer** (11x hintereinander), aber das Modell **erholt
sich danach** und produziert weiter normale Wortfetzen — anders als der
komplette, unumkehrbare Kollaps bei Prompt 2 vor dem Fix.

## Beobachtungen

- **Kein klarer Sprung bei der reinen Textqualität** — weiterhin dieselbe
  Grundsignatur wie vor dem Fix: Wikipedia-typische Fragmente (Jahreszahlen,
  ISBN-Nummern, ortsnamenartige Wortfusionen, Klammer-Referenzen), keine
  inhaltliche Kohärenz zu den Prompts, viele frei erfundene Wortfusionen
  ("Hitparischen", "Kart-LVer-Württemberg", "Efalender-Stand-Fürttemberg").
- **Der `<unk>`-Konfidenz-Kollaps ist deutlich seltener/kürzer geworden**:
  vor dem Fix ein Fall mit komplettem, irreversiblem Kollaps (Prompt 2, bis
  zum Zeichenlimit); nach dem Fix nur noch ein kurzer, selbst-korrigierender
  Ausreißer (Prompt 3, 11 Zeichen) und ein Durchlauf ganz ohne Kollaps
  (Prompt 2). Das deckt sich mit der Hypothese, dass die
  differenzierte LR die frische Kapazität schneller "sicher" macht, auch
  wenn sich das nicht in einer klar messbar niedrigeren End-Loss
  niederschlägt.

## Einordnung

In Summe ein durchwachsenes, aber leicht positives Bild: keine Verbesserung
bei Loss oder inhaltlicher Kohärenz, aber ein spürbar selteneres/schwächeres
Auftreten des Konfidenz-Kollaps-Fehlermodus. Für eine wirklich belastbare
Aussage bräuchte es mehr als einen Lauf pro Konfiguration (Lauf-zu-Lauf-
Streuung ist hier nicht von echten Effekten zu trennen) — als Momentaufnahme
spricht aber nichts dagegen, den Boost beizubehalten, und für die geplante
128-Dim-Drohnen-Baseline und weitere Maximaltrainings ist er ohnehin aktiv.
