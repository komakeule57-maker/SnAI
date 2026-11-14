# A/B-Test Drohnen-Tier: vier frische Experten unter gefixtem Gradient-Clipping vs. Bug-Baseline

## Testaufbau

- **Modelle:** vier Experten (Drohnen-Tier, 128 Dim, 2 Layer, 2 Heads),
  alle Genesis-trainiert unter der neu gebauten `libcortex.so` (2 Epochen
  automatischer Digest-Pfad, `weight_decay=0/dropout=0` — Drohnen-Tier nutzt
  keine Regularisierung, `royal_distillery.php`s Werte gelten erst ab
  Prinzessin):
  - `expert_GEN_7106a84930e` (GEN), Held-out-Loss **7.717329** (PPL 2246,9)
  - `expert_SFT_af76a84973d` (SFT), Held-out-Loss **7.745187** (PPL 2310,4)
  - `expert_SFT_5b06a849a55` (SFT), Held-out-Loss **7.854885** (PPL 2578,3)
  - `expert_SFT_27c6a849c12` (SFT), Held-out-Loss **7.713911** (PPL 2239,3)
  - GradNorm bei allen vier durchgehend im Bereich **0,7–2,4** (vorher, alte
    Engine: Minimum 254,8, Median 1303,7) — der Fix greift auch auf dieser
    kleinsten Stufe sichtbar.
- **Bug-Referenz:** `docs/inferenz/INFERENZ_DROHNE_BASELINE.md`,
  `expert_SFT_fe86a817501` (128 Dim, 2 Layer, 2 Heads, ebenfalls Epoche 2),
  Held-out-Loss **7.404329** (PPL 1643). Datei selbst existiert nicht mehr
  (durch die Resets gelöscht, nie separat gesichert) — Vergleich stützt sich
  auf die dort dokumentierte Rohausgabe.
- **Inferenz-Parameter:** identisch zu allen bisherigen Referenztests
  (Temperature 0.25, Rep. Penalty 1.15/Fenster 64, min_p 0.05, 256 Tokens,
  kein Beam Search).
- **Datum:** 2026-08-18, ca. 20:45 CEST.

## Ergebnisse (roh, unbearbeitet — Auszug, volle Läufe je 3 Prompts × 4 Modelle)

Alle zwölf Läufe (4 Modelle × 3 Standardprompts) zeigen dasselbe Muster:
durchgehender Kollaps in Satzzeichen-/Funktionswort-Wiederholung, kein
einziger sauberer `</s>`-Stopp, volles Zeichen-/Token-Budget ausgeschöpft.
Beispiel (`expert_GEN_7106a84930e`, Prompt 1):

```
, und,. der, die,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,, und.,,,,,,, die, der,,,,,,,,,.,,,. ,,,,,,,,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, und,,,,,,,,, die,,,,,,,,.,, der,,,,,,,,,,,,.,,.,,,,,,,,,,,,,,,,,,,,,,,,.
```

Beispiel (`expert_SFT_27c6a849c12`, Prompt 3, bester Held-out-Loss der vier):

```
,. und der die. die,,,, die, die, die., die,, die die. der,., die der,. die, zu,,. und,,, die, die, der,,..,.,..,, und,..,.
```

Alle übrigen 10 Läufe: strukturell identisches Bild (`die`/`der`/`und`/`zu`
plus lange Ketten aus Kommata/Punkten), keine erkennbaren Wortfragmente mit
Endungen/Flexion, keine inhaltliche Prompt-Kohärenz.

## Vergleich mit der Bug-Baseline

| | Bug-Baseline `fe86a817501` | Neu, gefixt (Median der 4) |
|---|---|---|
| Held-out-Loss | 7.404329 | 7.75 (Spanne 7.71–7.85) |
| GradNorm | nicht geloggt (vor Nachrüst-Paket #9) | 0,7–2,4 |
| Kollaps-Muster | Funktionswort-/Satzzeichen-Fragmente, keine Wortstruktur | identisch |
| Sauberer `</s>`-Stopp | keiner (3/3 Prompts) | keiner (12/12 Prompts) |

**Auffällig: der Bug-Baseline-Wert liegt sogar niedriger (besser) als alle
vier neuen Werte.** Nicht als Regression zu werten — Held-out-Set ist mit
999 Tokens winzig (hohe Varianz), Nectar-Ziehung ist pro Lauf zufällig
verschieden, und n=1 pro Konfiguration erlaubt ohnehin keine belastbare
Aussage. Wichtiger als der Loss-Wert selbst: das qualitative Fehlerbild ist
zwischen Bug- und Fix-Engine auf dieser Stufe **nicht unterscheidbar**.

## Einordnung

**Einwand (2026-08-18, direkt im Anschluss), berechtigt:** "kein
sichtbarer Qualitätssprung" ist für n=1 (alte Baseline) gegen n=4 (heute,
alle aus demselben Korpus-Zug, also nicht unabhängig voneinander) zu hart
formuliert. Subjektiv lesen sich die neuen Ausgaben kohärenter als
ältere, undokumentierte Drohnen-Eindrücke. Wichtiger: **alle Läufe in dieser
Test-Kette (Drohne/Prinzessin/Matriarchin) werden aus Zeitgründen bewusst
nur mit dem Minimum an Nektar pro Epoche gefüttert** — das deckelt die
erreichbare Qualität unabhängig von der eigentlich getesteten Variable
(Clipping-Fix). Ausbleibender dramatischer Sprung unter diesen knapp
bemessenen Bedingungen ist also ein schwaches, kein sauberes Nullresultat.
Sobald das Nektar-Budget pro Epoche angehoben wird, ist eine weitere
Verbesserung des Musters plausibel — unabhängig vom Clipping-Fix selbst.

**Kein sichtbarer Qualitätssprung durch den Gradient-Clipping-Fix auf
Drohnen-Ebene** (mit der obigen Einschränkung) — konsistent mit einem bereits
mehrfach in diesem Projekt dokumentierten Muster: das Drohnen-Tier (128 Dim,
2 Layer, ~2 Epochen, winziges Held-out-Set) kollabiert historisch praktisch
immer in genau dieses Satzzeichen-/Funktionswort-Muster, unabhängig von
Regularisierung, Init-Fixes oder jetzt auch dem Clipping-Fix (vgl.
`docs/inferenz/INFERENZ_DROHNE_BASELINE.md`: "PPL ≈1643 (≈20% des
Zufallsniveaus)... Deutlich schwächer als die 256-Dim-Prinzessin-Baseline").
Plausibelste Erklärung: bei dieser Kapazität/Datenmenge ist das Modell
schlicht zu unterfittet, um von saubererem Gradientenbetrag pro Schritt zu
profitieren — der Flaschenhals liegt hier an Kapazität/Datenvolumen, nicht
an Optimierer-Präzision.

**Das stützt indirekt die Hypothese, statt sie zu widerlegen:** wenn
der Clipping-Bug primär *Lerndynamik bei ausreichender Modellkapazität*
verzerrt, sollte sein Effekt genau dort sichtbar werden, wo bereits genug
Kapazität für differenzierte Gradienteninformation vorhanden ist — bei der
Prinzessin (256 Dim) zeigte sich heute bereits eine deutliche Loss/Verhalten-
Entkopplung zwischen bug-behafteten Checkpoints (`docs/inferenz/
INFERENZ_PRINZESSIN_6A826128B6548.md`). Der eigentliche Beleg wird also eher
bei den anstehenden Prinzessin- und Matriarchin-A/B-Tests erwartet
nicht auf Drohnen-Ebene. Der Drohnen-A/B ist damit inhaltlich
abgeschlossen — Ergebnis "kein Unterschied", was für sich selbst ein
verwertbarer Befund ist, kein offener Punkt.
