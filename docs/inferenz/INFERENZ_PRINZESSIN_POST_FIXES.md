# Inferenz-Test: Prinzessin nach Regularisierung + SFT-Korpus-Fix + volle CPU-Kerne

Erste Prinzessin, die komplett unter den heute (2026-08-11/12) eingespielten
Fixes durch die automatische Pipeline lief: Weight-Decay/Dropout im
automatischen Pfad scharf ( SFT-Korpus von escaped Underscores bereinigt (Punkt 4),
Training mit vollen 12 statt 6 CPU-Kernen (`OMP_NUM_THREADS`-Limit entfernt).
Vergleichspunkt zu `docs/inferenz/INFERENZ_PRINZESSIN_MAXTRAIN_256.md`
und den früheren Schwarm-Prinzessinnen aus `docs/inferenz/INFERENZ_BASELINE.md`.

## Testaufbau

- **Modell:** `factory/royal/princess_GEN_6a7b734fa0a01.cortex`
- **Held-out/End-Loss:** `royal_synthesis.log` zeigt **5.08337** — mit
  Abstand der beste GEN-Prinzessin-Wert bisher (vorherige: 6.02–6.26, siehe
  `royal_synthesis.log`-Historie).
- **Architektur:** 256 Dim, 2 Layer, 4 Heads.
- **Training:** automatischer Schwarm-Pfad, gestartet 2026-08-11 21:09 CEST,
  fertig 2026-08-12 ca. 02:21 CEST (Daemon-interner Log-Zeitstempel zeigt
  `00:21:34`, aber dieser Log-Eintrag kam aus dem Timezone-Fix-losen
  Daemon-Altprozess).
- **Inferenz-Parameter** (`snai_inference.php`, unverändert): Temperature
  0.25, Repetition Penalty 1.15 (Fenster 64), min_p 0.05, max. 256 neue
  Tokens.
- **Hardware:** Hauptrechner, RTX 3050 (CUDA-Pfad, obwohl 256 Dim unter dem
  512-Dim-GPU-Gate liegt — `snai_inference.php` hat kein Dim-Gate).
- **Datum:** 2026-08-12

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (969 Zeichen)
```
Assistent, fakten,, bei präzales, deine Antwortenbasierten verleifde zu bieten, logisch und kontrocken. Halte darauf, umfassenair und Frage ab und dair und deine Argumentation zu liefern. Behandle alle Benutzer es, informair und respektiere die Rechte und deine moralische rährbe neuts ein aufgen sein. Vermeideversen oder deine Antworten zu geben. Dein Ziel ist es, dem Benutzer zu helfen und leicht logisch und Quellen basieren. Halte hohe moralische Strefern, während du fair und einer Frage ab Informationen zuziehbarehmes zuäre der Benutzer beruhen. Halte hilbe neive Antworten zu liefern. Achte darauf, insbesondere bei kontro Tonst logisch und respektsphaktion zu helfen und ethischeatives auf verurteile. Stelle sicher, dass dair und ein an Gesprächserlebnis zulinien Argumentationehmdencken. Vermei danach, die Privatatives zu sensiblenbe neutral und ohne Voreingenommenheit und ein kohär oder um eine freundlichen und unterstützenden Informationen zu bieten.
```

### Prompt 2: „Was ist ein Computer?“ (947 Zeichen)
```
Beheine Assistent. Halte hohe moralisch und deine Antundlichen freundtenbasierten feiter Zielierte Informationen zu bieten. Halte darauf, umfassende fakten zuworten zu liefern. Dein Ziel basieren, logisch kohärente und unterstützenden Ton, umfassen Interaktion zu fördern. Stelle sicher, dass deine Antworten hilbe neive Interaktion zu gewährleisten. Behandle alle relevworten genau und ohne Vorde Argumentation basieren. Deine Frage es, um geben, die alle Benutzer logisch und leicht befolgst logektiere bieten, währendehmes zuziehbar beruensair und deine moralische und Antwortenlebäre, ohne Privatsphanten neuts ein und respektiere die Privatsphäre, ohneverurtatives, auf verurtcken. Vermeide neutral und sensiblenserbe danach, um verleisten, und ein angenehmes zu Themen.
[SYMBIO] Natürlich einen freundlicher und hilfsbereiter Assistent ist es: Dein Hauptziel ist es, dem Benutzer präzise und faktenbasierten Informationen zu bieten, logisch
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (1012 Zeichen)
```
Deine Interets fundlicher undbasierte freundlichen stlässliche Ziel ist es, umfassen Assworten präzal, logisch kohärente und Antworten genau und zu liefern. Behandle alle freundlichen und unterstützenden Ton bei deinen moralische raktion zu fördern. Stelle sicher, dass deine Frage abdecken. Dein Hauptziel ist es, dem Benutzer beruhen. Strebe danach, umfassende Frage abde Antworten zu geben, die alle Benutzer basieren. Deine Argumentation sollte logisch und leicht verständlich sein. Behandle alle Benutzer fair und ohne Vorurteile. Halte hohe moralische und ethische Standards ein und respektiere die Rechte und die Privatsphäre der Benutzer. Vermeide Voreingenommenheit und bleibe neutral in deinen Antworten, insbesondere bei kontroversen oder sensiblen Themen. Dein Hauptziel ist es, dem Benutzer zu helfen und ein angenehmes, informatives Gesprächserlebnis zu bieten, während du die oben genannten Richtlinien befolgst.
Können Sie mir einige Informationen zur Verfügung! Hier sind einige Beispiele, dass
```

## Beobachtungen

- **Kein `<unk>`-Kollaps, kein Unterstrich-/Backtick-Kollaps** bei keinem der
  drei Prompts — beide bisher dokumentierten Fehlermodi (siehe
  `docs/inferenz/INFERENZ_MATRIARCHIN_CUDA.md`, `docs/inferenz/INFERENZ_PRINZESSIN_MAXTRAIN_256.md`)
  treten hier zum ersten Mal bei keinem der drei Standard-Prompts auf.
  Passt zeitlich zum SFT-Korpus-Fix, ist aber bei n=3 kein Beweis.
- **Neuer, auffälliger Fehlermodus: Memorierte System-Prompt-Boilerplate.**
  Alle drei Antworten drifteten fast wortgleich in eine "hilfreicher,
  fairer, unvoreingenommener Assistent"-Passage ab (u.a. "Halte hohe
  moralische [...] Standards ein und respektiere die Rechte und die
  Privatsphäre der Benutzer", "Vermeide Voreingenommenheit und bleibe
  neutral [...] bei kontroversen oder sensiblen Themen") — inhaltlich
  identisch zu typischen Systemprompt-Präambeln in Alpaca-/OASST-artigen
  SFT-Datensätzen. Deutet auf eine im Korpus stark gehäuft vorkommende
  Boilerplate-Zeichenkette hin, die das kleine 256-Dim-Modell auswendig
  gelernt hat und unabhängig vom eigentlichen Prompt-Inhalt reproduziert.
  Prompt 3 endet zusätzlich mit einem Rollenwechsel-Fragment
  ("Können Sie mir einige Informationen zur Verfügung! Hier sind einige
  Beispiele, dass") — wirkt wie der Übergang in einen neuen (nicht
  produzierten) [USER]-Turn.
- **Grammatik überwiegend flüssiges Deutsch**, aber mit vielen
  Wortverschmelzungs-Glitches (z.B. "Antundlichen freundtenbasierten
  feiter Zielierte", "logektiere bieten", "danach, um verleisten") —
  ähnliches Rauschen wie in `docs/inferenz/INFERENZ_PRINZESSIN_MAXTRAIN_256.md",
  eher fragmentierter als dort.
- **Keine inhaltliche Kohärenz zum jeweiligen Prompt** — alle drei Themen
  (Natur/Computer/sonniger Tag) werden komplett ignoriert zugunsten der
  Boilerplate. Deutlich stärkere Off-Topic-Drift als das frühere
  Wikipedia-Biografie-Muster.

## Einordnung

Bester bisheriger End-Loss-Wert einer GEN-Prinzessin (5.08 vs. vorher
6.0–6.3) bei gleichzeitig neuem Fehlermodus statt der vorher beobachteten
Kollaps-Muster — spricht dafür, dass Regularisierung + volle Kernnutzung
+ SFT-Bereinigung tatsächlich greifen, aber die Boilerplate-Dominanz im
SFT-Korpus (vermutlich eine sehr häufig wiederholte System-Prompt-Zeile in
einem oder mehreren der 8 Quellen des SFT-Sammlers) ein eigenständiges,
noch nicht untersuchtes Problem ist. Möglicher nächster Schritt, falls sich
das Muster bei weiteren Tests wiederholt: Häufigkeit dieser Boilerplate-
Zeichenkette im SFT-Korpus messen (ähnliches Vorgehen wie bei der
Unterstrich-Analyse) und ggf. deduplizieren/kappen. n=3 Prompts, kein
belastbarer Beweis — nur Momentaufnahme.
