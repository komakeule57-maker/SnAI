# Referenz-Inferenztest Matriarchin `1557a2`, Epochen 2 und 3

Nachholung der beiden offenen Referenztests. Epoche 1 war am
18.08. dokumentiert (`INFERENZ_MATRIARCHIN_1557A2_EPOCHE1.md`), die
Checkpoints für Epoche 2 und 3 lagen seit dem 18.08. gesichert bereit, waren
aber nie getestet worden. Gemessen am 20.08. gemeinsam mit dem
Gradient-Clipping-A/B, deshalb mit identischer Methodik — die
Zahlen aus beiden Dokumenten sind direkt vergleichbar.

## Testaufbau

- **Modelle:** `secured_models/matriarchin_GEN_1557a2_epoch{2,3}.apex`
  (sha256-verifiziert am 18.08. gesichert).
- **Architektur:** 512 Dim, 6 Layers, 8 Heads. Alle drei Epochen stammen aus
  demselben Lauf, alle **vor** dem Gradient-Clipping-Fix trainiert.
- **Inferenz-Parameter:** Projektstandard (Temperature 0.25, Repetition
  Penalty 1.15 / Window 64, min_p 0.05, max. 256 neue Tokens, kein Beam).
- **Stichprobe:** 8 Läufe je Prompt, 3 Standard-Prompts = **24 Läufe je
  Checkpoint** (Epoche 1 zum Vergleich unter denselben Bedingungen neu
  gemessen, nicht aus dem alten n=1-Dokument übernommen).

## Messergebnis über die drei Epochen

| Epoche | Held-out | PPL | Ø Zeichen | Ø ohne Whitespace | Whitespace-Anteil | `<unk>` gesamt | Läufe mit `<unk>` |
|---|---|---|---|---|---|---|---|
| **1** | 5.347133 | 210.01 | 377.2 | 298.8 | 20.8 % | 788 | 13 / 24 |
| **2** | 5.072810 | 159.71 | 245.0 | 159.8 | **34.8 %** | **0** | **0 / 24** |
| **3** | 4.651470 | 104.74 | 274.2 | 229.3 | **16.4 %** | 241 | 2 / 24 |

## Qualitativer Verlauf

**Epoche 1** (bereits am 18.08. dokumentiert): dominiert von `<unk>`-Ketten
und Leerzeichen-Läufen, alle Prompts liefen ins 256-Token-Limit.

**Epoche 2 — der Sprung passiert hier.** Der `<unk>`-Kollaps ist in 24
Läufen **kein einziges Mal** aufgetreten. Stattdessen erscheinen erstmals
grammatisch zusammenhängende, wenn auch inhaltlich leere Sätze:

> „Ich bin ich, dass du dich nicht in deinem und deine Zeit für die Handlung
> von einem Schatten zu machen. Sie kann auch eine gute Möglichkeit, um sie
> zu helfen, wie sie sich selbst.“

> „Die auf dem Weg in der Mitte des Zweiten Weltkrieges, die von den
> Schachtel zu finden. Sie wird als auch eine Frau und das Hause, die im
> Jahr darauf hinzufügen. […]“

Erkauft wird das mit dem **höchsten Whitespace-Anteil aller vier bisher so
gemessenen Checkpoints (34.8 %)** — die Fehlerklasse wandert von
`<unk>`-Spam zu Leerzeichen-Läufen, sie verschwindet nicht. Besonders
ausgeprägt bei Prompt 2 (einzelne Läufe mit 264 Zeichen Ausgabe, davon nur
26 Nicht-Whitespace).

**Epoche 3 — deutlich flüssiger, dafür sichtbare Korpus-Artefakte.**
Erstmals vollständig wohlgeformte, faktisch plausibel klingende Sätze:

> „Um den ersten Mal, der in der Folge der Insel und die beiden dreiten
> Blätzen, die in der Nähe des zweiten Teiles von der Stadt. **Die erste
> Hälfte des 19. Jahrhunderts wurde die Kirche im Jahr 1571 eingeweiht.**
> Das Gebäude wurde am 18.&nbsp;Jahrhundert und dem Namen des Landes
> errichtet.“

Der Whitespace-Anteil fällt auf den besten Wert der Reihe (16.4 %), und
`<unk>` taucht nur noch in 2 von 24 Läufen auf (davon einer als einzelner
Ausreißer mit 236 Vorkommen — ein Lauf, der ins Limit rannte, während die
anderen 23 sauber terminierten).

**Zwei neue Korpus-Befunde aus den Epoche-3-Ausgaben**, beide vorher nicht
dokumentiert:

1. **`&nbsp;` im Klartext** („am 18.&nbsp;Jahrhundert") — ein
   HTML-Entity-Leck aus dem Wikipedia-Anteil des GEN-Korpus. Dieselbe
   Fehlerklasse wie das am 15.08. bereinigte Wikitabellen-/Vorlagen-Leck,
   aber ein anderes Muster, das die damalige Bereinigung nicht erfasst hat.
2. **LaTeX-Mathematik im Fließtext.** Prompt 2 („Was ist ein Computer?")
   erzeugte eine komplette Pseudo-Rechnung: „die Wahrscheinlichkeit, dass es
   sich aus dem Fall von `$5$` ist, was bedeutet, dass `$4 = 0$`. […] die
   zweite Zahl `$3 = -1 + 1 = -2 = -2 = -1$`" — Durchschlag aus den
   SFT-Mathe-Datensätzen. Kein Fehler im engeren Sinn (das Material ist
   legitim im Korpus), aber ein Hinweis, dass der Mathe-Anteil bei einem
   allgemeinen Prompt dominant werden kann.

## Einordnung

Die Loss-Reihe fällt sauber und deutlich (5.347 → 5.073 → 4.651, PPL 210 →
160 → 105), und **diesmal folgt das Verhalten dem Loss** — der Fortschritt
von unlesbarem `<unk>`-Spam über grammatische Hülsen zu wohlgeformten
Sätzen ist über die drei Epochen unübersehbar. Das passt zu dem mehrfach
beobachteten Muster: **bei großen
Loss-Differenzen korreliert Loss zuverlässig mit Qualität, bei kleinen
Differenzen zwischen ähnlichen Checkpoints nicht.** Hier liegen zwischen
Epoche 1 und 3 fast 0,7 Loss-Punkte — eine große Differenz, und die
Korrelation hält.

**Konsequenz für den Gradient-Clipping-A/B:** dieses Ergebnis ist der Grund, warum der
Gradient-Clipping-A/B schwächer ausfällt als zunächst gedacht. Da der
`<unk>`-Kollaps in der Bug-Linie schon zwischen Epoche 1 und 2 von selbst
verschwindet, ist sein Fehlen beim gefixten Epoche-1-Checkpoint kein
Beweis für die Wirkung des Fixes. Details dort:
`INFERENZ_MATRIARCHIN_1FDF68_GRADCLIP_AB.md`.

**Konsequenz fürs Aufräumen:** die beiden Korpus-Artefakte (`&nbsp;`,
LaTeX-Dominanz) sind neue, noch nicht adressierte Punkte — beide klein, aber
`&nbsp;` gehört in dieselbe Kategorie wie die bereits gefixten
Wikipedia-Lecks und wäre mit einem Sammler-seitigen Filter plus
Bestandsbereinigung billig zu schließen.
