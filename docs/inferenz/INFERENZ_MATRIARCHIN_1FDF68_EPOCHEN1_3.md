# Referenz-Inferenz Matriarchin `1fdf68`, Epochen 1–3 — vollständiger Vergleich gegen die Bug-Referenz `1557a2`

Abschluss der am 20.08. begonnenen Vergleichsreihe. Der Lauf wurde
nach Epoche 3 gestoppt (analog zu `1557a2`, das damals ebenfalls
bei Epoche 3 endete), alle drei Checkpoints sind gesichert. Damit liegt
erstmals eine **epochenweise Gegenüberstellung gefixt gegen Bug über die
komplette Reihe** vor.

## Testaufbau

- **Gefixt:** `secured_models/matriarchin_GEN_1fdf68_epoch{1,2,3}.apex`
  (sha256 `d735feed…`, `090aeced…`, `03e317f4…` — jeweils unmittelbar nach
  Epochenende gesichert, vor dem Überschreiben durch die Folgeepoche).
- **Bug-Referenz:** `secured_models/matriarchin_GEN_1557a2_epoch{1,2,3}.apex`.
- **Beide:** 512 Dim, 6 Layers, 8 Heads, GEN-Kalibrierung, ~20 MB frischer
  Nektar pro Epoche.
- **Inferenz-Parameter:** Projektstandard (Temperature 0.25, Repetition
  Penalty 1.15 / Window 64, min_p 0.05, max. 256 neue Tokens, kein Beam).
- **Stichprobe: n=8 je Prompt, 3 Standard-Prompts → 24 Läufe je Checkpoint**,
  insgesamt 144 Läufe. Alle Bug-Werte wurden am 20.08. unter denselben
  Bedingungen frisch gemessen, nicht aus älteren n=1-Dokumenten übernommen.

## Held-out-Loss

| Epoche | `1fdf68` (gefixt) | PPL | `1557a2` (Bug) | PPL | Differenz |
|---|---|---|---|---|---|
| 1 | 5.486806 | 241,5 | **5.347133** | 210,0 | Bug −0,140 |
| 2 | 5.292680 | 198,9 | **5.072810** | 159,7 | Bug −0,220 |
| 3 | 5.072072 | 159,5 | **4.651470** | 104,7 | Bug −0,421 |

Der Bug-Lauf ist in **jeder** Epoche besser, und der Abstand **wächst
monoton**. `1fdf68` E3 (5.0721) liegt praktisch exakt auf `1557a2` E2
(5.0728) — der gefixte Lauf hängt rund eine volle Epoche zurück.

## Inferenzverhalten (je 24 Läufe)

| Checkpoint | Ø Zeichen | Ø ohne WS | Whitespace | `<unk>` gesamt | Läufe mit `<unk>` |
|---|---|---|---|---|---|
| **gefixt E1** | 153,1 | 116,6 | 23,8 % | **1** | **1 / 24** |
| **gefixt E2** | 502,8 | 427,4 | 15,0 % | **1** | **1 / 24** |
| **gefixt E3** | 530,5 | 439,8 | 17,1 % | **1** | **1 / 24** |
| Bug E1 | 377,2 | 298,8 | 20,8 % | **788** | 13 / 24 |
| Bug E2 | 245,0 | 159,8 | 34,8 % | 0 | 0 / 24 |
| Bug E3 | 274,2 | 229,3 | 16,4 % | **241** | 2 / 24 |

**Das Bild ist exakt umgekehrt zum Loss.** Der gefixte Lauf ist über alle drei
Epochen praktisch `<unk>`-frei (ein einziges Vorkommen in 72 Läufen), der
Bug-Lauf schwankt erratisch zwischen 788, 0 und 241. Zusätzlich wächst beim
gefixten Lauf die Ausgabelänge monoton (153 → 503 → 531 Zeichen), während der
Bug-Lauf keinen erkennbaren Trend zeigt (377 → 245 → 274).

## Qualitative Beispiele

**gefixt E2** („Erzähl mir etwas über die Natur."):

> „Die Herausforderung besteht darin, dass sie die Anforderungen der Daten in
> den verschiedenen Themen berücksichtigt werden."

**gefixt E3** („Erzähl mir etwas über die Natur."):

> „Die Analyse der Lösung des Wappers, die das Unternehmen für den anderen
> Einfluss auf eine Vielzahl von der Zeit in der Kategorie Butter und die
> Geschichte der Gemeinschaft der jeweiligen Einschränkungen. **Die optimale
> Lösung erfordert eine Kombination der verfügbaren Ressourcen, die die
> Gesamtzahl der Themen zugeordnet werden.**"

Der hervorgehobene zweite Satz ist grammatisch vollständig wohlgeformt.
Inhaltlich bleibt es Konfabulation, aber es gibt keinen Zerfall.

**Bug E3** zum Vergleich (aus `INFERENZ_MATRIARCHIN_1557A2_EPOCHE2_3.md`):

> „…**Die erste Hälfte des 19. Jahrhunderts wurde die Kirche im Jahr 1571
> eingeweiht.** Das Gebäude wurde am 18.&nbsp;Jahrhundert und dem Namen des
> Landes errichtet."

Beide Läufe erreichen bei E3 also wohlgeformte deutsche Sätze. Der
Unterschied liegt nicht in der Grammatik, sondern in der Zerfallsneigung.

## Der zentrale Befund: Loss und Verhalten laufen auseinander

Das Modell mit dem **schlechteren** Held-out-Loss (gefixt, 5.072 gegen 4.651
bei E3) hat den **deutlich saubereren** Output. Das ist die ursprüngliche
Hypothese vom 19.08. in ihrer klarsten bisherigen Ausprägung — und
diesmal nicht an kleinen Loss-Differenzen zwischen ähnlichen Checkpoints,
sondern über eine Differenz von 0,42 Punkten hinweg.

Das widerspricht dem bisher festgehaltenen Muster („bei großen
Loss-Differenzen korreliert Loss zuverlässig mit Qualität"), zumindest wenn
man `<unk>`-Zerfall als Qualitätsmaß nimmt.

## Widerspruch zum kontrollierten A/B — offen

Dieses Ergebnis steht **gegen** den konfundfreien A/B vom 20.08.
(`experiments/gradclip_ab/`, Lauf 2), wo der Fix bei frischen Daten an jedem
Messpunkt den besseren Held-out-Loss lieferte:

| Test | Regime | Held-out-Sieger |
|---|---|---|
| A/B Lauf 2 (kontrolliert) | Drohne 128 Dim, 8 MB, 4 Chunks | **gefixt** |
| Diese Reihe (unkontrolliert) | Matriarchin 512 Dim, 60 MB, 3 Epochen | **Bug** |

Zwei Unterschiede kommen als Erklärung in Frage, keiner ist belegt:

1. **Diese Reihe ist nicht kontrolliert.** Andere Nektarziehungen pro Epoche,
   anderer Korpusstand (`1fdf68` trainierte nach der Quarantäne-Episode),
   andere Abstammung (fix-trainierte gegen bug-trainierte Prinzessinnen).
   Der kontrollierte A/B bleibt die belastbarere Evidenz über den Optimierer
   als solchen.
2. **Der Skalenunterschied ist groß** — 128 gegen 512 Dim, 8 gegen 60 MB. Ob
   der Clipping-Effekt mit der Modellgröße die Richtung wechselt, ist offen
   und mit den vorliegenden Daten nicht entscheidbar.

Ein sauberer Test wäre der kontrollierte A/B auf Prinzessin- oder
Matriarchin-Ebene — also dasselbe Verfahren wie in `experiments/gradclip_ab/`,
nur eine Stufe höher. Kosten: nach den gemessenen Durchsätzen (Prinzessin
~4,2 MB/h, Matriarchin ~2,0 MB/h) ein Mehrfaches des Drohnen-Laufs.

## Einordnung

Der Lauf wurde nach Epoche 3 von 6 gestoppt — beide Reihen enden also am
selben Punkt und sind fair vergleichbar, aber beide sind unfertig. Ob sich
der Abstand bei Epoche 4–6 weiter öffnet, schließt oder umkehrt, ist nicht
bekannt.

Das Verhaltensbild spricht klar für den Fix, das Loss-Bild klar dagegen. Wer
nur eine der beiden Metriken betrachtet, kommt zum gegenteiligen Schluss —
was für sich genommen der wichtigste methodische Ertrag dieser Reihe ist.
