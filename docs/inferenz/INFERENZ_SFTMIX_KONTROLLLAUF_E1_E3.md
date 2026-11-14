# SFT-Mix-Kontrolllauf E1–E3 — Mix oder Kapazität? Antwort: überwiegend Mix

Auswertung am 2026-08-23, 06:30 CEST. Der am 21.08. 20:23 gestartete
Kontrolllauf `experiments/princess_sftmix/` hat drei Epochen fertig, der
parallel laufende Chinchilla-Lauf steht bei Epoche 7.

## Die Frage

Der Chinchilla-Lauf (~11 % SFT nach Bytes) entwickelte über E1–E3 **keine**
Prompt-Konditionierung, während die Perplexität von 147 auf 61 fiel. Zwei
Erklärungen waren nicht trennbar:

- **(a) Mix** — nur ~10 % des Gradienten trainieren „auf eine Frage antworten".
- **(b) Kapazität** — 6.555.392 Parameter reichen nicht, ein Prompt-Thema
  über 200 Token durchzuhalten.

Der Kontrolllauf ändert genau eine Variable: SFT-Anteil ~69 % statt ~11 %.
Identisch gehalten sind Genesis-Basis (sha256 `5df547331e0223cc…`, beide
byte-gleich geprüft), Holdout (`e95360dea3afb3b2…`), Epochengröße,
Epochenzahl, LR/WD/Dropout/LabelSmoothing und die Inferenz-Harness.

## Ergebnis bei gleicher Epoche 3

Gleiche Basis, gleiche Hyperparameter, gleicher Tokenstand — nur der Mix
unterscheidet sich:

| | Chinchilla E3 (11 % SFT) | **SFT-Mix E3 (69 % SFT)** |
|---|---|---|
| `topic_own` | 1/15 (6,7 %) | **13/30 (43,3 %)** |
| `topic_foreign` | 3/15 (20,0 %) | **2/30 (6,7 %)** |
| `topic_lift` (geglättet) | 0,86 | **10,80** |
| `distinct_run_starts` | 6/15 | **22/30** |
| Held-out | 4,1171 | 4,3938 |

Aufgeschlüsselt nach Prompt (SFT-Mix E3): **Natur 8/10, Computer 2/10,
Tag 2/10.** Der Chinchilla-Lauf erreicht über E4 *und* E5 zusammen
**Computer 0/60**.

**Der Held-out ist beim Kontrolllauf schlechter — das war vorab so
angekündigt** und ist kein Qualitätsurteil: das Holdout ist reines GEN, und
dieser Lauf sieht davon nur ein Drittel. Leitmetrik ist `topic_lift`.

## Qualitativ: der Unterschied ist unübersehbar

**SFT-Mix E3** greift den Prompt auf und baut einen Definitionsrahmen:

> „**Die Natur ist** eine Art von Natur und Natur, die wir unseren Natur in
> unsere Natur und Natur in unserer Natur erzeugen."
>
> „**Ein Computer ist** eine Person, die Sie mir lernen, wie wir unsere
> Person und unseres Lebenserlebnis zu unserer Zukunft."

**Chinchilla E5** liefert themenfreien Assistenz-Baustein, unabhängig vom
Prompt:

> „Ich habe keinen Zugriff auf eine sorgfältige, um ein paar Ideen zu
> erstellen und ein paar Ideen für unsere Liebe zu geben."
>
> „Natürlich! Hiernen Sie ein paar Vorschläge, um eine sorgfältige Details zu
> finden, die du verwenden kannst:"

Beide Antworten sind inhaltlich wertlos. Der Unterschied liegt exakt auf der
gemessenen Achse: der eine hängt sich ans Thema, der andere an die Form.

## Einschränkungen — die Kapazitätshypothese ist nicht tot

**1. Der Chinchilla-Lauf ist kein flacher Nullbefund.** E4 zeigte
`topic_own=6/30` bei `topic_foreign=0/30`. Nachgeprüft: alle sechs Treffer
stammen vom Natur-Prompt („…um eine Vielzahl von Natur und Natur zu
erzeugen", „…das Naturschutzgebiet zu erzeugen"), Computer und Tag je 0.
E5 fiel wieder auf 2/30 zurück. Das Signal **flackert** also und hängt am
häufigsten Wort. Zur Einordnung: die ungeglättete Lift-Formel gab dafür
`inf` aus — deshalb rechnet die Harness seit dem 23.08. mit additiver
Glättung (+0,5 auf beide Zähler), `own` und `foreign` stehen ohnehin roh
daneben.

**2. Auch bei 69 % SFT kann das Modell nur das Themenwort echoen, nichts
darüber sagen.** „Die Natur ist eine Art von Natur und Natur" ist
Konditionierung ohne Inhalt. Die saubere Lesart ist deshalb **zweistufig**:

- Der **Mix** entscheidet, ob das Modell überhaupt aufs Thema anspringt.
  Hier ist der Befund deutlich, Faktor ~6 in der Trefferquote.
- Die **Kapazität** entscheidet vermutlich, ob es dann etwas Sinnvolles
  sagen kann. Dafür spricht der Kollaps in Wiederholung („Natur und Natur
  und Natur"), den mehr SFT-Daten nicht behoben haben.

Es war also keine Entweder-oder-Frage. Beide Faktoren wirken, auf
verschiedenen Ebenen.

**3. n=30 bleibt dünn für Einzelepochen.** Beide Läufe haben bei E2 einen
Einbruch (Chinchilla 0/15, SFT-Mix 3/30). Belastbar ist der Vergleich bei
E3 und die Richtung über die Reihe, nicht der einzelne Epochenwert.
