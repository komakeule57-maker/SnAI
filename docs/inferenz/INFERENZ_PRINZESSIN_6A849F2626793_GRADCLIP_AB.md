# Inferenz-Test: Prinzessin `princess_SFT_6a849f2626793` — erste komplett unter dem gefixten Gradient-Clipping geborene Prinzessin

Erster Prinzessin-Ebene-Datenpunkt nach dem
`libcortex.so`/`libcortex_cuda.so`-Rebuild (Gradient-Clipping-Fix) und dem
vierten kompletten Reset. Rebuild und Reset liefen unbeobachtet (Zeitleiste
rekonstruiert); dieser Inferenztest selbst wurde live durchgeführt.

## Testaufbau

- **Modell:** `factory/royal/princess_SFT_6a849f2626793.cortex`
- **Architektur:** 256 Dim, 3 Layers, 4 Heads (per Inferenz-Banner
  bestätigt) — gleiche Ladder-Generation wie die beiden Bug-Referenzen
  unten.
- **Training:** automatischer Schwarm-Pfad, **SFT-Zweig**
  (Kalibrierungs-Nektar, Net2Net-expandiert aus
  `expanded_royal_6a849f263a4fa`), erste komplette Prinzessin nach dem
  vierten Reset. Epoche 1 fertig 2026-08-18 22:32:09 CEST, Epoche 2
  2026-08-19 00:49:24 CEST, Epoche 3 2026-08-19 03:07:35 CEST
  ("[Queen] ERFOLG: Prinzessin '6a849f2626793' wurde geboren!").
- **Held-out-Loss-Verlauf über 3 Epochen:** 5.915764 (PPL 370.84) →
  5.810403 (PPL 333.75) → **5.794238 (PPL 328.40)**, finaler
  Best-Checkpoint.
- **GradNorm über den kompletten Trainingslauf** (612 geloggte Schritte,
  alle 3 Epochen): Minimum 0.6263, Mittelwert 1.22 — durchgehend im vom
  Audit vorhergesagten Bereich, kein einziger Wert im drei- oder
  vierstelligen Bereich wie vor dem Fix (Minimum 254.8, Median 1303.7).
- **Inferenz-Parameter** (`snai_inference.php`, Standard-Sampling,
  `beam_width` weggelassen): Temperature 0.25, Repetition Penalty 1.15
  (Fenster 64), min_p 0.05, max. 256 neue Tokens — identisch zu allen
  bisherigen Referenztests.
- **Hardware:** Hauptrechner, RTX 3050. **Auffällig:** dieser Lauf ging
  über den CUDA-Pfad ("CUDA Engine eingerastet"), frühere Prinzessin-Tests
  liefen über OpenCL — reines Nebenbeobachtung, keine bekannte Ursache
  untersucht, kein Hinweis auf einen Effekt auf die Ausgabe.
- **Datum:** 2026-08-19, ca. 04:47 CEST.

## Ergebnisse (roh, unbearbeitet)

### Prompt 1: „Erzähl mir etwas über die Natur.“ (522 Zeichen, kein sauberer Stopp)
```
 Die beiden Anstung der Anforderungen der Vergenung des DA-Prakisch und die BD-Mur-Bechränkungen, um die Analyse der Mäfale in den Talen von der Röder zu finden.

Die Autin-Truh. Die Stürte für die Fitation und die Dundung der Wärten, die sich mit einem anderen, dass die Bimatien, die die Anforderungen von der Analyse und die Veranstaltungen, die die Einschränken zu finden.
[SYMBIO] Der Mägelle ist die Schichtung der BS-Begen und die Anforderungen der Fale, die auch die Täfigen und die PED-Markisch und die Anglichen.
```

### Prompt 2: „Was ist ein Computer?“ (766 Zeichen, kein sauberer Stopp)
```
Schuhe: Die Analyse der Einstellung der Lrose-Plie, die in der BD-Begation und die Anforderungen zu einem anderen und die Tägens.

Der Hamo-Marz-Am, die auch das Verbränkmmack, die sich für die Düde. Die Fale, um die Verordnung der Anforderungen zu einer anderen Analyse der Bürgen, dass die Verteilung von den MS- und die Falen, die sie die Bärken, die die Anforderungen der Regels in die Hägelle, die die Schrändern, die die Ankacken.
[SYMBIO] Die der Anbungen ist eine der Tale zu finden.
[SYMBIO] Der Eine ist der DAB-Mil-Pur und die Analyse des BS-Kürfiert werden, dass die kägenung der Ausst.
[SYMBIO] Die Münstigen von den Falen und die Steilung der CD-Tin-Larbal, die die Verforderungen zu einer der Lokie-Amale, um das Analyse der Timien und die Einstellung
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“ (366 Zeichen, kein sauberer Stopp)
```
Nach der KA-Pechu (*&nbsp;m), die sich in der Gobration und den Bäden von der Stadt-Berfale.

Die Togung des Poches ist ein der Troste der Hammer-Schar, dass die Verteilung der Körten der Weikischs.

Der Möder sind der Fall- und in der BA-Murbg.

Die Castung der Ragon-Befal-Bara, der im Jahr 2020 und Sen-Schm.

Die Steilung von der Stadt-Gitische Apatischen Pogn.
```

## Beobachtungen

- **Kein sauberer `</s>`-Stopp bei irgendeinem der drei Prompts** — alle
  drei laufen bis nahe an oder ganz an das 256-Token-Budget (522/766/366
  Zeichen).
- **Kein `<unk>`-Spam-Kollaps** (anders als `6a826128b6548`, der bug-era
  Prinzessin mit dem bis dahin besten Loss der Reihe). Stattdessen ein
  drittes, bisher so nicht dokumentiertes Kollapsmuster: durchgehend
  erfundene, mit Bindestrich zusammengesetzte Pseudo-Komposita
  („DA-Prakisch", „BD-Mur-Bechränkungen", „Hamo-Marz-Am", „KA-Pechu",
  „Ragon-Befal-Bara") — grammatisch näher an echten deutschen
  Wortformen/Flexionsendungen als der Drohnen-Kollaps
  (`INFERENZ_DROHNEN_GRADCLIP_AB.md`), aber ohne erkennbaren Sinn oder
  Prompt-Bezug.
- **Wiederkehrendes Artefakt: `[SYMBIO]`-Tag im generierten Text**
  (Prompt 1 + 2, insgesamt 4×) — offenbar ein Formatierungs-/Systemtoken
  aus dem SFT-Kalibrierungs-Korpus, das hier als literaler Text
  reproduziert statt als Steuerzeichen interpretiert wird. Neuer, noch
  nicht untersuchter Befund, nicht weiter verfolgt (ähnlich dem
  `&nbsp;`-HTML-Entity-Leck, das in `6a826128b6548` schon einmal notiert,
  aber nie nachverfolgt wurde — hier erneut aufgetreten, Prompt 3).
- **Rein orthografisch/morphologisch wirkt der Output "deutscher"** als
  beide Bug-Referenzen unten: vollständige Wörter mit korrekten Endungen
  („Anforderungen", „Veranstaltungen", „Bürgen"), Satzbau mit Subjekt/Verb/
  Nebensatz-Struktur — aber durchgehend inhaltsleer bzw. mit erfundenen
  Fantasiewörtern durchsetzt.

## Vergleich mit den Bug-Referenzen vom 18.08.

**Wichtiger Konfund vorab:** die beiden Referenzen unten sind
**GEN-Prinzessinnen**, die neue ist eine **SFT-Prinzessin** (anderer
Korpuszweig, andere Zielverteilung, Loss-Masking auf Assistenten-Turns
statt Fließtext) — zusätzlich zum Clipping-Fix selbst ein nicht
kontrollierter Unterschied. Kein sauberer isolierter A/B-Vergleich, wie
schon bei allen bisherigen Referenztests in dieser Kette angemerkt.

| | `6a81f9adb25f8` (Bug, GEN) | `6a826128b6548` (Bug, GEN) | **`6a849f2626793` (gefixt, SFT)** |
|---|---|---|---|
| Held-out-Loss (final) | 5.817457 | **5.600596** (bester der drei) | 5.794238 |
| GradNorm | nicht geloggt (vor Nachrüst-Paket #9) | nicht geloggt | **0.63–2.4, Mittel 1.22** |
| Output-Länge | 22–50 Zeichen (weit unter Budget) | volles Budget (nicht exakt gemessen, dokumentiert als „bis zum Token-Budget") | volles/nahes Budget, 366–766 Zeichen |
| Sauberer `</s>`-Stopp | **ja, alle 3 Prompts** | nein, keiner der 3 | nein, keiner der 3 |
| Kollapsmuster | kurze Funktionswort-Fragmente | schwerer `<unk>`-Spam (60+ Ketten) | erfundene Bindestrich-Komposita, `[SYMBIO]`-Leck |

## Einordnung

**Kein eindeutiges Bild — weder klare Bestätigung noch klare Widerlegung
der Hypothese, mit dem SFT/GEN-Konfund als Haupteinschränkung.**
Die neue, gefixte Prinzessin liegt beim Loss zwischen den beiden
Bug-Referenzen, zeigt aber **keinen** der beiden Extremfälle: nicht den
sauberen, kurzen Stopp von `6a81f9adb25f8` (trotz leicht besserem Loss),
aber auch nicht den schweren `<unk>`-Spam-Kollaps von `6a826128b6548`
(trotz ähnlicher Ausschöpfung des Token-Budgets). Das ist konsistent mit
einer **milderen** Ausprägung des in `INFERENZ_PRINZESSIN_6A826128B6548.md`
beschriebenen Loss/Verhalten-Auseinanderlaufens, aber bei n=1 pro
Konfiguration und dem SFT/GEN-Konfund keine belastbare Bestätigung.

**Einziger neuer, eindeutig positiver Befund:** GradNorm ist erstmals auf
Prinzessin-Ebene über einen kompletten 3-Epochen-Lauf durchgehend geloggt
und liegt exakt im vom Audit vorhergesagten Bereich (~1 statt
Hunderte/Tausende) — bestätigt, dass der Fix auf dieser Kapazitätsstufe
technisch genauso greift wie beim Drohnen-Test
(`INFERENZ_DROHNEN_GRADCLIP_AB.md`).

**Weiterhin unklar:** ob die verbleibende Instabilität (kein sauberer
Stopp, Fantasiewort-Kollaps) an Kapazität/Datenmenge liegt (wie auf
Drohnen-Ebene vermutet) oder tatsächlich noch Restanteile der
Loss/Verhalten-Entkopplung sind. Der eigentlich saubere Test dafür bräuchte
eine **GEN**-Prinzessin unter dem Fix als direktes Gegenstück zu
`6a81f9adb25f8`/`6a826128b6548` — die Pipeline hat nach dieser
SFT-Prinzessin bereits mit einer GEN-Prinzessin (`princess_GEN_6a850677ed9b3`,
Stand dieses Tests: Epoche 1 in Arbeit) weitergebaut, die dafür in Frage
käme, sobald sie fertig ist. Der Prinzessin-A/B ist damit **inhaltlich abgeschlossen**
(ein Datenpunkt vorhanden), aber mit offener Anschlussfrage für einen
saubereren Vergleich.
