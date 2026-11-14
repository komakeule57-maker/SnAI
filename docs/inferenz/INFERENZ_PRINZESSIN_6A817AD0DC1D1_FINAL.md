# Inferenz-Test: Prinzessin `princess_SFT_6a817ad0dc1d1` — erste Prinzessin unter dem korrigierten 3/6/12-Ladder

Diese Prinzessin ist die erste komplett unter dem am 16.08. korrigierten
`EVOLUTION_LADDER` (Prinzessin 2→3 Layer) UND der neuen tiefenskalierten
Genesis-Init (`init_base.php`, GPT-2-artige `w_o`/`w_down`-Skalierung)
gewachsene Prinzessin — Net2Net-Wachstum aus einer Drohne, die bereits mit
der neuen Init initialisiert wurde. Dritter kompletter Reset (10:14 CEST)
als Ausgangspunkt, kein A/B-Test (gleichzeitig auch neuer Korpus-/
Zufalls-Stand).

## Testaufbau

- **Modell:** `factory/royal/princess_SFT_6a817ad0dc1d1.cortex`
- **Architektur:** 256 Dim, **3 Layers** (bisher 2 — erster Lauf mit dem neuen
  Ladder-Wert), 4 Heads (headSize=64). Net2Net-Wachstum aus 3 SFT-Drohnen-
  Tributen: `Vocab 8192 | 128->256 Dim | 512->1024 Ctx | 2->3 Layers`,
  Head-Skalierung 2→4 Heads.
- **Training:** automatischer Schwarm-Pfad, gestartet 2026-08-16 10:54:41
  CEST (Worker 217258), fertig 17:26:34 CEST (~6h32m für 3 Epochen).
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.992076 (PPL 400.24) →
  5.88478 (PPL 359.52) → **5.865353** (PPL 352.61, finaler
  Best-Checkpoint). Abnehmender Grenzertrag (−0.107 dann −0.019), ähnliches
  Plateau-Muster wie beim Vorgänger.
- **Regularisierung (Log bestätigt):** AdamW-Weight-Decay 0.01,
  FFN-Dropout 0.1, Label-Smoothing 0.1, LR-Boost 3x für die neu
  gewachsenen Parameter (Net2Net-Grenze erkannt).
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Fenster 64), min_p 0.05, max. 256 neue Tokens — identisch zu allen
  bisherigen Referenztests.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL-Pfad (`snai_inference.php`
  hat kein Dim-Gate, aktiviert GPU unabhängig von der Modellgröße;
  Training selbst blieb bei 256 Dim bewusst auf CPU, GPU-Gate liegt bei
  512 Dim).
- **Datum:** 2026-08-16, ca. 18:15 CEST.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (262 Zeichen, kein sauberer Stopp)
```
2:

Dere:

En- und:



4-



-
```
*(fast ausschließlich Leerzeilen, wenige kurze Fragmente — s. Beobachtungen)*

### Prompt 2: „Was ist ein Computer?“ (285 Zeichen, kein sauberer Stopp)
```
 Hier sind einige Beispiele, die du wie sie zu helfen, um ihre Informationen können.

Die! Sie kann einige Beispiele:

Die einige ich eine Beispiele:

2.

An:

1mür: Die Anste:

-In: "Derer:

Die:

Dere", dass du ein paar Sie zu einer Vielzahl

Die Sie ist eine Informationen

Das!
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (660 Zeichen, kein sauberer Stopp)
```
Die einige Beispiele:  1:

Am der Fener, die eine S. Lura-Schal, die das die Trar und die Anstung von den deine zu einer kürt.

3. Die Schars, um die Küzte, dass sie in der Erfack, wie die du mit dem Fassenen, die sich auf die Bippe und die Kunden und die Poba- und die Telecken und die Mischer zu einem kürstung von der Trok.

1.  Die Vermackten:

Die Stachgar-Kue, die sie eine sen und die Küz, die die sie zu den Fizien zu einer schuflich, um die du und die deine.

2. "Mles", dass sie in der
1. Die Tuton:

Die Boba-Cini-Sere, die eine ein, dass es für die küreln, die sich zu den anderen zu verbessern und das, um die du zu einer Annisung zu vereine.

2.
```

## Beobachtungen

- **Neues, drittes Kollapsmuster — anders als beim Vorgänger.** Der
  Vorgänger `6a8050b1332a5` kollabierte dominant ins `[Gekürzt]`-Artefakt
  (seitdem gefixt). Hier kein einziges
  `[Gekürzt]` mehr sichtbar — der Korpus-Fix wirkt. Stattdessen drei
  unterschiedliche Fehlerbilder je Prompt:
  - Prompt 1: fast reiner Leerzeilen-/Whitespace-Kollaps mit vereinzelten
    Kurzfragmenten (`2:`, `Dere:`, `En-`, `4-`) — die schwächste der drei
    Ausgaben.
  - Prompt 2: erkennbare, teils grammatikalisch fast korrekte Fragmente
    ("Hier sind einige Beispiele, die du wie sie zu helfen, um ihre
    Informationen können.") gefolgt von Aufzählungs-/Listen-Versatzstücken
    (`2.`, `1mür:`) und abbrechenden anführungszeichen-Fragmenten.
  - Prompt 3: mit Abstand am dichtesten — durchgehende, listenartige
    Struktur (`1:`, `3.`, `1.`, `2.`) mit erfundenen, morphologisch
    deutsch-ähnlichen Wörtern (Fener, Anstung, Bippe, Küzte, Vermackten,
    Stachgar-Kue) ohne semantische Kohärenz, aber mit klarer
    Satz-/Listenrhythmik.
- **Kein Prompt endet über ein sauber vorhergesagtes `</s>`** — alle drei
  laufen bis zum Token-Budget durch (bzw. bis das Skript selbst terminiert).
- **Listen-/Aufzählungsformat als wiederkehrendes Muster** (nummerierte
  Punkte `1.`/`2.`/`3.` bei Prompt 2 und 3) — im SFT-Korpus vermutlich
  häufig (Anleitungs-/FAQ-artiges Material), wird hier unabhängig vom
  Prompt-Inhalt reproduziert.
- **Erfundene, aber deutsch-phonotaktisch plausible Wortfragmente**
  (Prompt 3) — deutlich über dem Funktionswort-Spam der Drohnen-Baseline
  (`INFERENZ_DROHNE_BASELINE.md`), aber weiterhin ohne erkennbare
  inhaltliche Prompt-Kohärenz.

## Vergleich mit historischen Werten

**1. Gegen den direkten Vorgänger `princess_SFT_6a8050b1332a5`**
(`docs/inferenz/INFERENZ_PRINZESSIN_6A8050B1_FINAL.md`, letzte Prinzessin
vor dem Ladder-Fix, 2 statt 3 Layer):

| | `6a817ad0dc1d1` (neu, 3 Layer) | `6a8050b1332a5` (alt, 2 Layer) |
|---|---|---|
| Held-out-Loss (final) | 5.865353 (PPL 352.6) | 5.862187 (PPL 351.5) |
| Layer | 3 | 2 |
| Prompt 1 | 262 Zeichen, Whitespace-Kollaps | 219 Zeichen, Wortsalat → `[Gekürzt]`-Fragment |
| Prompt 2 | 285 Zeichen, Teilfragmente + Listen | 10 Zeichen, `[Gekürzt]`-Sofortkollaps |
| Prompt 3 | 660 Zeichen, listenartig, erfundene Wörter | 10 Zeichen, `[Gekürzt]`-Sofortkollaps |
| `[Gekürzt]`-Artefakt | keins mehr | dominant bei 2/3 Prompts |

**Held-out-Loss praktisch identisch** (Unterschied +0.003, im Rauschen) —
der zusätzliche Layer zeigt sich **nicht** in einem messbaren Loss-Vorteil
bei dieser Stichprobengröße (n=1 Modell je Konfiguration, konfundiert mit
Korpus-/Zufallsunterschieden zwischen den Läufen). Qualitativ aber deutlich
**mehr generierter Inhalt vor dem Abbruch** (285/660 vs. 10/10 Zeichen bei
Prompt 2/3) — das ist jedoch primär dem zwischenzeitlich gefixten
`[Gekürzt]`-Korpus-Bug zuzuschreiben, nicht nachweisbar dem zusätzlichen
Layer oder der tiefenskalierten Init. Kein sauberer A/B-Test (mehrere
gleichzeitige Änderungen: Ladder-Fix, tiefenskalierte Init, dritter Reset,
neuer Korpus-Zufallsstand, `[Gekürzt]`-Fix bereits aktiv).

**2. Gegen die Drohnen-Baseline** (`docs/inferenz/INFERENZ_DROHNE_BASELINE.md`,
128 Dim, 2 Layer, PPL ≈1643, ≈20,1% Zufallsniveau): deutlich niedrigerer
PPL (352.6 vs. 1643, ≈4,3% vs. ≈20,1% Zufallsniveau) und sichtbar mehr
Wortstruktur (Prompt 3 hier zeigt zusammenhängende, wenn auch unsinnige
Wortfolgen mit Endungen/Flexion, die Drohne blieb reiner Satzzeichen-/
Funktionswort-Spam) — erwartbarer Kapazitätssprung zum nächsten Tier.

**3. Gegen `docs/inferenz/INFERENZ_BASELINE.md`** (Legacy-1-Head-Prinzessin,
Loss 6.56528, PPL ≈710): niedrigerer Loss/PPL (352.6 vs. 710), ähnlich wie
schon beim Vorgänger `6a8050b1332a5` festgestellt.

## Einordnung

Der Ladder-Fix (2→3 Layer) und die tiefenskalierte Init laufen fehlerfrei
durch die volle Pipeline (Net2Net-Wachstum, 3 Trainingsepochen, Kalibrierung)
— technisch verifiziert. Ein **isolierter Qualitäts- oder Loss-Effekt des
zusätzlichen Layers ist aus diesem Test aber nicht ableitbar**: Held-out-Loss
liegt im Rauschen identisch zum Vorgänger, und der auffälligste qualitative
Unterschied (kein `[Gekürzt]`-Kollaps mehr) stammt nachweislich vom
zwischenzeitlich gefixten Korpus-Bug, nicht vom Ladder-/Init-Fix. Für einen
belastbaren Vergleich bräuchte es einen kontrollierten Lauf mit identischem
Korpus-Snapshot und nur der Layer-Zahl als Variable — nicht Teil dieses
Tests. Nächster Schritt laut geplanter Kette: dieselbe Prüfung bei der
nächsten fertigen Matriarchin (erste komplett unter dem neuen 6-Layer-
Verhältnis).
