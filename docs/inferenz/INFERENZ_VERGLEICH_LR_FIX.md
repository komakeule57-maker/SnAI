# Inferenz-Vergleich: GEN-Prinzessin vor/nach dem LR-Boost-Fix

Direkter Vorher/Nachher-Vergleich für die "differenzierte Learning-Rate für
frisch Net2Net-expandierte Parameter". Ergänzt
`docs/inferenz/INFERENZ_BASELINE.md` (die dortige Baseline ist SFT, hier GEN gegen GEN —
sauberer Vergleich, kein Track-Wechsel).

## Testaufbau

| | Vor dem Fix (Baseline) | Nach dem Fix (Growth-Test) |
|---|---|---|
| Modell | `factory/royal/princess_GEN_6a6fa6fbcb954.cortex` | `factory/testruns/princess_growth_test/princess_GEN_growthtest.cortex` |
| Trainings-Loss | 6.67528 | **6.24765** |
| Herkunft | reguläre 3-Drohnen-Verschmelzung (Schwarm-Pipeline) | **1** Drohne direkt expandiert (isolierter Test, 2026-08-04) |
| Architektur | 256 Dim, 2 Layer, 4 Heads | identisch |
| LR-Boost | nein (vor dem Fix) | ja, 3x auf frisch expandierte Bereiche |
| Nectar | 5 Dateien (regulärer Schwarm-Mix) | 4 Dateien (3 GEN + 1 SFT, aus `factory/archive/`) |

**Wichtige Einschränkung:** kein sauber isoliertes Experiment — der Growth-Test
stammt aus einer einzelnen Drohne statt der üblichen 3-fach-Verschmelzung, und
die Nectar-Zusammensetzung weicht leicht ab (eine vor dem Fix schon trainierte Zusammensetzung war
mangels 3 verfügbarer GEN-Drohnen nicht exakt reproduzierbar). Der LR-Boost ist damit nicht die
einzige Variable, die sich geändert hat.

**Parameter:** Temperature 0.25, Repetition Penalty 1.15, max. 256 neue Tokens
(identisch zu `docs/inferenz/INFERENZ_BASELINE.md`).

## Ergebnisse

### Prompt 1: „Erzähl mir etwas über die Natur.“

**Vor dem Fix** (632 Zeichen):
```
Sie „Dnbsp;irt (Bnbsp;.
Aersp;), das der San, die die den Hauptstischen Stadt, die Dmuo, die den Vorlage.
Mon, die Bas, dass die die die die dem Kach der den Angehen und die die die Erlichen Tangen.
Die Kanda, die Leits, die die Fall.
Wrie Gotte der den Unter anderem die die die den „Die der den Poge, die die die den Zeit der einem mit dem Ein dem Mach und sich von der die die Ablich.
[...]
```

**Nach dem Fix** (371 Zeichen):
```
Die Sie „Ane.
Db der S;
Lrer.
B<unk>n (Sn;
19897-Killin, der dass er von der Rans, dem Kieachsti.
Mo und der Anlichen ersten Nuische Bon:
Waryt.
Rolen.
Gm.
Eic,.
T<unk>n, IPB;Hie 3-53–402).
Fie 3.
2001)
[...]
```

### Prompt 2: „Was ist ein Computer?“

**Vor dem Fix** (571 Zeichen):
```
;|1901&SN 3. Die Aus dem Ster der Stadt und die den den Oie, das BLas (Bnbsp;-Dnbsp;).
E<unk>msp;5–199.
„Cnbsp;), die seine einem der Verstisch, die „„Snbsp;.
Gie der Zeit von den Sar.
Molische Rüz.
We, die „Die, in der Batbsp;-To, die der Wand, die die die Srlin und die die ein der die keinen der Tilte der Tie, die Pie und die sich der Keinbe der den die die dem Kuri.
[...]
```

**Nach dem Fix** (353 Zeichen):
```
Drel, die der die die in der die den die im Leiten.
An.
Fie.
S-Bimo, die die Tlbger. Die Solischen ;
Di (P<unk>n:
Hare und Rachung.
En.
Cn).
Bn-Mi,.
Jonl).
K.
191-945–;
Nach ausissbfon,.
[...]
```

### Prompt 3: „Beschreibe einen sonnigen Tag.“

**Vor dem Fix** (546 Zeichen):
```
„Berbsp;<unk>. Die Lolischen Tarel, die der „Die der Stadt und die dem Pal (Sas.
Nach der Mat.
Dnbspen der Gemeinde der Schäiert.
Hie Sa:
Dnbspe, die den Bürte der der Sand, das Vie, in der Film, Sdon, die Wirt, die eine den den Kspen von der Hand, die die Zanes, ein der die Fulers, die Gifen, die der der der Künung des Tag der Verlehns.
[...]
```

**Nach dem Fix** (305 Zeichen):
```
Die, die 1920.
A-Sbaten.
Ge.
Cn-S<unk>;
BN„19947683).
Sn: „Marbalerachs, 3.
Ho (* „Tri.
K.
Pn
Dl-S<unk>-Jy
2005–;
W„L.
181-S; „Son
U)
[...]
```

## Beobachtungen — ehrlich, nicht schöngeredet

**Loss ist eindeutig besser** (6.25 vs. 6.68), aber **die Inferenz sieht nicht
eindeutig besser aus, nur anders**:

- **Positiv:** die vorher dominante Artikel-Wiederholungsschleife
  ("der/die/den" praktisch endlos hintereinander) ist im Nachher-Text spürbar
  seltener — passt zum erhofften Effekt, dass die frische Kapazität nicht mehr
  im selben Trapping-Modus feststeckt.
- **Negativ:** der Nachher-Text zerfällt dafür **früher** in kurze Fragmente
  (305–371 Zeichen vs. 546–632 Zeichen vorher) und enthält deutlich mehr
  **Zahlen-/Jahreszahl-Fragmente** (19897, 978694, 20098884099, 9778363,
  19947683, ...). Das sieht nach einem neuen, anderen Failure-Mode aus: die
  GEN-Archivquelle (Wikipedia-Dump) ist voller Jahreszahlen und
  Klammer-Referenzen — mit 3x höherer effektiver Rate könnte die frische
  Kapazität genau dieses sehr häufige, aber inhaltsarme Oberflächenmuster
  besonders schnell aufgegriffen haben.
- Beide Versionen bleiben bei allen drei Prompts **inhaltlich unbezogen** zum
  jeweiligen Thema (Natur/Computer/sonniger Tag) — keine Verbesserung bei der
  eigentlichen Kohärenz, die laut Perplexity-Diskussion ohnehin eher am
  Datenvolumen hängt als an der Optimierung.

## Einordnung

Der Fix tut nachweislich etwas (niedrigerer Loss, sichtbar andere
Fehler-Signatur), aber **kein eindeutiger Qualitätssprung** bei diesem kleinen,
nicht perfekt isolierten Test. Mögliche nächste Schritte, falls das weiter
verfolgt wird:
- Sauberer A/B-Test mit identischem Nectar und echter 3-Drohnen-Verschmelzung
  (dieser Test hatte beides nicht exakt repliziert).
- Boost-Faktor (aktuell 3x, `GROWTH_LR_BOOST` in `train_worker.php`) evtl.
  moderater ansetzen, falls die Zahlen-Fragmentierung auf zu aggressives
  Lernen der häufigsten Oberflächenmuster hindeutet.
- Test bei der Matriarchin-Stufe wiederholen, sobald die laufenden Läufe
  (Hauptrechner/Laptop) durch sind — bei mehr Trainingsschritten könnte sich
  das Bild anders darstellen als bei der kurzen 3-Epochen-Prinzessin-Kalibrierung.
