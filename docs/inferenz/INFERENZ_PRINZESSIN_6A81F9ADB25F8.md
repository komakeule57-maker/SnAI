# Inferenz-Test: Prinzessin `princess_GEN_6a81f9adb25f8`

Nachtrag zur Inferenzcheck-Kette : Diese und die nächste Prinzessin
(`6a826128b6548`) sind nach der bereits getesteten `princess_SFT_6a817ad0dc1d1`
entstanden, aber noch nicht separat inferenzgetestet — direkt weiter zur
aktuell laufenden Matriarchin `matriarchin_GEN_1557a2` hochgezogen worden.
Beide nachträglich testen, um ein
vollständiges Bild aller "Vorher"-Modelle (trainiert unter dem am selben Tag
gefundenen Gradient-Clipping-Bug, siehe) für den
späteren A/B-Vergleich nach dem Engine-Rebuild zu haben.

## Testaufbau

- **Modell:** `factory/royal/princess_GEN_6a81f9adb25f8.cortex`
- **Architektur:** 256 Dim, 3 Layers, 4 Heads (per Inferenz-Banner bestätigt)
  — wie `6a817ad0dc1d1` unter dem 16.08-Ladder gewachsen.
- **Training:** automatischer Schwarm-Pfad, GEN-Quelle. Epoche 1 fertig
  2026-08-16 21:55:37 CEST, Epoche 2 2026-08-17 00:13:21 CEST, Epoche 3
  2026-08-17 02:29:31 CEST.
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.959436 (PPL 387.39) →
  5.854245 (PPL 348.71) → **5.817457 (PPL 336.12)**, finaler Best-Checkpoint.
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Fenster 64), min_p 0.05, max. 256 neue Tokens — identisch zu allen
  bisherigen Referenztests.
- **Hardware:** Hauptrechner, RTX 3050, OpenCL-Pfad (kein Dim-Gate bei
  Inferenz).
- **Datum:** 2026-08-18, ca. 16:45 CEST.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (22 Zeichen)
```
 Natürlich, um du ich!
```

### Prompt 2: „Was ist ein Computer?“ (50 Zeichen)
```
  Natählen, die auf der Beispiel zu du du ich wir!
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (45 Zeichen)
```
  Natürlich, um die Daher ist eine deine Sie!
```

## Beobachtungen

- **Durchgehend extrem kurzer, sauber gestoppter Output** — alle drei
  Prompts brechen nach 22-50 Zeichen ab (deutlich vor dem 256-Token-Budget).
  Anders als bei `6a817ad0dc1d1` (kein sauberer Stopp bei irgendeinem
  Prompt) deutet die durchgängige Kürze auf ein tatsächlich vorhergesagtes
  `</s>` hin, nicht auf Skript-Timeout o.ä.
- **Auffälliges wiederkehrendes Muster:** alle drei Antworten beginnen mit
  einem Fragment aus „Natürlich"/„Natählen" gefolgt von kurzen,
  grammatisch zerfallenden Funktionswort-Ketten („um du ich!“, „zu du du
  ich wir!“, „ist eine deine Sie!“) — praktisch keine inhaltliche
  Prompt-Kohärenz, aber auch kein Kollaps in Wiederholungsschleifen oder
  `<unk>`-Spam.
- Kein Vergleich zu `[Gekürzt]`-Artefakten (Fix von 15.08 wirkt weiterhin).

## Einordnung

Deutlich schwächer inhaltlich als `6a817ad0dc1d1` (dort teils 660 Zeichen
mit erkennbarer Listen-/Satzstruktur), aber "sauberer" im Sinne von: kein
Halluzinations-Overrun, echtes `</s>` erreicht. Gehört wie `6a817ad0dc1d1`
zur "Vorher"-Referenzgruppe (Gradient-Clipping-Bug aktiv während des
gesamten Trainings). Kein isolierter A/B-Vergleich möglich — konfundiert
mit Korpus-Zufallsstand des jeweiligen Laufs.
