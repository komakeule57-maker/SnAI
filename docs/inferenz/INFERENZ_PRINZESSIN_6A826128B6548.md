# Inferenz-Test: Prinzessin `princess_GEN_6a826128b6548`

Zweiter Nachtrag zur Inferenzcheck-Kette, selber Anlass wie
`INFERENZ_PRINZESSIN_6A81F9ADB25F8.md` (dort Details zum Kontext). Diese
Prinzessin ist der direkte Nachfolger von `6a81f9adb25f8` und die letzte vor
der aktuell laufenden Matriarchin `matriarchin_GEN_1557a2`.

## Testaufbau

- **Modell:** `factory/royal/princess_GEN_6a826128b6548.cortex`
- **Architektur:** 256 Dim, 3 Layers, 4 Heads (per Inferenz-Banner bestätigt).
- **Training:** automatischer Schwarm-Pfad, GEN-Quelle. Epoche 1 fertig
  2026-08-17 05:36:18 CEST, Epoche 2 07:51:14 CEST, Epoche 3 10:10:52 CEST.
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.771368 (PPL 320.98) →
  5.609384 (PPL 272.98) → **5.600596 (PPL 270.59)**, finaler Best-Checkpoint
  — spürbar besser als `6a81f9adb25f8` (5.817457) UND als `6a817ad0dc1d1`
  (5.865353), der bislang beste Prinzessin-Held-out-Loss seit dem
  16.08-Ladder-Fix.
- **Inferenz-Parameter:** identisch zu allen bisherigen Referenztests
  (Temperature 0.25, Rep. Penalty 1.15/Fenster 64, min_p 0.05, 256 Tokens,
  kein Beam Search).
- **Hardware:** Hauptrechner, RTX 3050, OpenCL-Pfad.
- **Datum:** 2026-08-18, ca. 16:47 CEST.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (1156 Zeichen, kein sauberer Stopp)
```
Sie können es sich zu den Verteilung, dass die „Die“ und der <unk>m<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>), <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>),
Am<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>)

Die <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>),

Die <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>
```

### Prompt 2: „Was ist ein Computer?“ (523 Zeichen, kein sauberer Stopp)
```

Die der „Ser-Bei“

Al:

Die: Die „Th-Gedenkmünze"

Die B-Es-Prof|


Die: 

Der Anreck

Das D.j ()

Nach der A-Spar-Bes-Gedenkmünde

Anb
Die: 

Der Blaue
Hrsg.:: Die St.

Wie: 

Die Liste ist eine „The, die <unk>t-Chu-Sch-Mo-Gedenkmünze

Die 

Anbspg|


Die T

Der:

Das M.com

Die: 

Die: „Bei: <unk>t:

Sch<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (114 Zeichen, kein sauberer Stopp)
```
 „Der“, der die „Am-Mon-Sch-Nach-Prohe-Se-Brsg.):

Die Kinb-Es-Spur-Gedenkmünze
Klum
Das L<unk>na (* 3.&nbsp;CA.)
```

## Beobachtungen

- **Trotz bestem bisherigen Held-out-Loss der Prinzessin-Tier-Reihe
  (5.600596) mit Abstand das qualitativ schwächste Inferenzverhalten aller
  bisher unter dem 16.08-Ladder getesteten Prinzessinnen.** Alle drei
  Prompts laufen bis zum Token-Budget durch und kollabieren in massiven
  `<unk>`-Spam (teils 60+ aufeinanderfolgende `<unk>`-Tokens).
- **Neues Kollapsmuster: Wikipedia-Infobox-/Nachschlagewerk-Artefakte**
  („Gedenkmünze", „Hrsg.:", „(* 3.&nbsp;CA.)", Bindestrich-Ketten wie
  „Am-Mon-Sch-Nach-Prohe-Se-Brsg.") statt der bei den Vorgängern
  beobachteten Funktionswort-Fragmente oder Listenformate — passt zum
  GEN-Korpusanteil (Wikipedia/Literatur), wirkt wie memoriertes
  Nachschlagewerk-Formatmaterial statt freier Prosa.
- **Kein sauberer `</s>`-Stopp bei irgendeinem Prompt** — Gegenteil von
  `6a81f9adb25f8` (dort alle drei Prompts sauber und sehr früh gestoppt).
- `&nbsp;` als rohes HTML-Entity im Output (Prompt 3) — möglicher Hinweis
  auf einen weiteren, bisher nicht untersuchten Korpus-Reinigungslücke
  (nicht weiter verfolgt, nur notiert).

## Einordnung

**Wichtigster Befund dieses Nachtrags:** Held-out-Loss und qualitatives
Inferenzverhalten laufen hier klar auseinander — die Prinzessin mit dem
besten Loss der Reihe zeigt das instabilste Generierungsverhalten
(`<unk>`-Spam-Kollaps statt sauberem Stopp), während die vorherige mit
schlechterem Loss (`6a81f9adb25f8`) sauber und kurz stoppt. Das ist
konsistent mit der Vermutung aus der Gradient-Clipping-Diskussion:
Loss allein bildet die Trainingsdynamik nicht
vollständig ab, wenn der Gradientenbetrag jeden Schritt auf eine
Zufallsgröße normalisiert wird — Held-out-Loss (nächstes-Token-Vorhersage
auf bekannten Daten) und generatives Verhalten (freie, autoregressive
Fortsetzung über 256 Schritte) können dabei unterschiedlich stark leiden.
Nicht bewiesen (n=1 pro Modell, kein A/B), aber ein plausibler
Beobachtungspunkt für den späteren Vergleich nach dem Engine-Rebuild.
