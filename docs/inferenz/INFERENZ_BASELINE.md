# Inferenz-Baseline: SFT-Prinzessin (256 Dim)

Vergleichswerte für spätere Tier-Vergleiche (Matriarchin, Empress, ...). Ausgangspunkt
war die Frage, ob die aktuellen Prinzessinnen Satzzeichen und Artikel (der/die/das)
schon zuverlässig richtig setzen — ohne dokumentierte Vergleichswerte war das bisher
nur ein grober Eindruck aus Ad-hoc-Tests. Dieses Dokument hält drei reproduzierbare
Durchläufe fest, damit sich künftige Tests (z.B. mit der fertigen Matriarchin) direkt
dagegen vergleichen lassen.

## Testaufbau

- **Modell:** `factory/royal/princess_SFT_6a6fa21f081c0.cortex`
- **Warum dieses Modell:** von den 5 lokal verfügbaren SFT-Prinzessinnen die mit dem
  niedrigsten (besten) Trainings-Loss (6.56528 laut `factory/royal_synthesis.log`,
  Eintrag vom 2026-08-03 04:07). Andere verfügbare SFT-Prinzessinnen lagen bei
  6.7998–7.1966 Loss.
- **Architektur:** 256 Dim, 2 Layer, **1 Attention-Head** (Legacy-Struktur — dieses
  Modell stammt von vor der Multi-Head-Migration).
- **Perplexity zur Einordnung:** Loss 6.56528 bei Vocab=8192 → Perplexity ≈ 710
  (≈8.7% des Zufalls-Niveaus 8192).
- **Inferenz-Parameter** (`snai_inference.php`, unverändert): Temperature 0.25,
  Repetition Penalty 1.15, max. 256 neue Tokens.
- **Hardware/Pfad:** Laptop, AMD Renoir APU. Auffällig: die Inferenz aktiviert den
  OpenCL/GPU-Pfad unabhängig von der Modellgröße (`⚡ [SYMBIO VRAM] ... eingerastet
  auf: [AMD Radeon Graphics ...]`) — anders als beim Training, wo `train_worker.php`
  GPU erst ab `model_dim >= 512` aktiviert. `snai_inference.php` hat offenbar kein
  entsprechendes Gate.
- **Datum:** 2026-08-04

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“
Prompt verdaut in 645ms, 941 Zeichen generiert:
```
„-n der die der dem den der die die der der vonischen der dem der vorische der der der der demspische der dem der im Januar vonische, der der der dem der der der der der der der demischerische der der der der der demischen der der der der der der fünfischen dem der seiner demische demsport und der der der einer der der dem der dem derrecht.

 „Ser“, derhard (19).
D-P-T-e, in der vonischen der demburg, des der der zweiischen dem der der der der der dem der den der dem der der der dem der der der der der wurde und der der seinem dem der Deutschland.


  <unk>-.
          „Hinischer der deutschermann und der der der der der der der der der der der der der dem der den der der der demischen der der demischerischen demischen dem der im Märzburg vonische der den dem der der dem der dem der der dem der amischeschische dem der dem der dem der der der der der der inischen der Jahrehalle mit der der die dem der allem nach der dem der der
```

### Prompt 2: „Was ist ein Computer?“
Prompt verdaut in 663ms, 581 Zeichen generiert:
```
„-3), von der die die den demische der in der der der der dermann (nischen der seinerbau.
        „19–8.
*<unk>-Der und der der der amischen von der der im Jahr am 2010 wurde wurde des der viergebiet des 2010 von der der dem der Rischen deutschersg-L-Per (5-V.

      
                                                                                       „S-1, später an der der dem der in dem der der der nach der Sie--B-4:  „Gen, er von der Landwirtschaft von der der der der der Stadtburg, das der den nach der drei der dort der der von der die den von der sichige der S-
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“
Prompt verdaut in 613ms, 416 Zeichen generiert:
```
„-), die der der von dem der der und nach der der den der der der der der der amische demischen der der der der amischer der der der seinem 19 und im Oktoberburg in der wurde des Jahre von der der der 13.
*
<unk> (8)
```
(nach dem letzten sichtbaren Zeichen folgen nur noch Leerzeilen bis zum
Zeichen-Limit)

## Beobachtungen

- **Artikel/Funktionswörter dominieren tatsächlich stark** — „der/die/dem/den“
  erscheinen in praktisch jedem Satzfragment, oft mehrfach hintereinander. Das
  bestätigt die erwartete Perplexity-Einordnung (Funktionswörter sind niederentropisch
  und werden zuerst zuverlässig gelernt), geht aber über „meistens richtig gesetzt“
  hinaus — die Wiederholungsrate wirkt eher wie ein Trapping-Modus als wie gezielte,
  korrekte Verwendung.
- **Echte deutsche Morphologie blitzt durch:** Adjektivendungen (-ischen, -ische),
  Monatsnamen (Januar, März, Oktober), Ortsnamen-Suffixe (-burg), Jahreszahlen
  (19XX, 2010, 13.), Anführungszeichen und Gedankenstriche an einigermaßen
  plausiblen Stellen. Kein Zufallsrauschen, aber auch keine zusammenhängenden Sätze.
- **Keine inhaltliche Kohärenz zum Prompt** — keine der drei Antworten geht erkennbar
  auf Natur, Computer oder einen sonnigen Tag ein.
- **Generation degeneriert mit der Länge:** alle drei Durchläufe kippen nach ca.
  150–300 Zeichen in Leerzeilen, `<unk>`-Tokens oder einzelne Sonderzeichen —
  passt zur früher diskutierten Beobachtung, dass Fehler sich autoregressiv
  aufschaukeln, je weiter die Generierung vom Prompt-Kontext wegdriftet.
- **Bewertung der ursprünglichen Einschätzung** („in 2 von 3 Fällen noch unsicher“):
  mit diesem — laut Loss besten verfügbaren — Modell wirkt das Bild sogar etwas
  durchwachsener als „unsicher“; alle drei Durchläufe zeigen die Artikel-Wiederholung
  deutlich. Möglich, dass der ursprüngliche informelle Test mit einem anderen Modell
  oder anderen Prompts lief.

## Für den Vergleich mit späteren Tiers

Dieselben drei Prompts, dasselbe Inferenz-Skript, dieselben Parameter (Temperature
0.25, Rep. Penalty 1.15, 256 Tokens) sollten für den Vergleich mit der fertigen
Matriarchin (und späteren Tiers) wiederverwendet werden, um einen echten
Vorher-Nachher-Vergleich zu haben statt eines neuen Eindrucks ohne Referenzpunkt.
