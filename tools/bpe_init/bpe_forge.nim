# Dateiname: bpe_forge.nim
# Funktion: Trainiert den BPE-Tokenizer (Pure Nim, High Speed) und exportiert
# ein 100% Huggingface/SentencePiece-kompatibles JSON-Manifest.

#[
  SYMBIO NANO-AI FRAMEWORK
  Modul: Pure Nim Tokenizer Forge (Huggingface Compliant Edition)

  FIX-LOG (V2 - "wirklich" High-Speed):
  1) BUG: MetaSpace war ein literales ASCII-Leerzeichen (0x20), obwohl der
     Kommentar "U+2581" versprach. Jetzt echtes "\u2581" (3 Bytes, E2 96 81) -
     matcht bpe_forge.py und damit das, was SnaiTokenizer.php erwartet.
  2) BUG (der eigentliche Grund fuer "komische Dateigroessen"): der Merge-Loop
     hat bei JEDER Iteration ALLE Woerter neu gezaehlt - O(VocabSize * Korpus).
     Bei einem ueberschaubaren PHP/txt-Korpus reisst das idR ab, lange bevor
     VocabSize erreicht ist (still, ohne Fehlermeldung) - die tatsaechliche
     Vokabelgroesse haengt dann unvorhersehbar am Korpusinhalt, nicht an der
     Konfiguration. Jetzt: inkrementelle Paarzaehlung, die nach einem Merge
     NUR die betroffenen Woerter neu bewertet (Beweis der Aequivalenz zur
     naiven Variante wurde vorab in Python gefuehrt, identische Merge-Sequenz
     und identische finale Tokens auf einem synthetischen Korpus).
  3) Robustheit: Zeichenfolgen aus \s (Leerzeichen/Tab) werden vor der
     Pre-Tokenisierung kollabiert (wie bpe_forge.py's Replace(r"\s+"," ")),
     damit Whitespace-Rauschen nicht Vokabel-Slots frisst, die eigentlich fuer
     Inhalt gedacht sind.
  4) Diagnose statt Stille: wird VocabSize nicht erreicht, sagt das Skript
     explizit warum (keine Paare mehr uebrig) statt einfach kleiner zu stoppen.
]#

import os, strutils, tables, sets, json, unicode, times, math, algorithm

const
  VocabSize = 8192
  MetaSpace = "\u2581"   # E2 96 81 - Gemma/LLaMA-Style Metaspace-Marker
  # Anteil aller Zeichenvorkommen, den das Grundalphabet abdecken muss. Der
  # Rest (seltene Zeichen) geht als UTF-8-Bytes ins Training, statt je einen
  # der 8192 Plaetze zu belegen - ohne Schwelle fraessen die ~7000 seltenen
  # Zeichen des Produktivkorpus fast das ganze Vokabular. Frei gewaehlt.
  CharCoverage = 0.99995

let specialTokens = ["<unk>", "<pad>", "<s>", "</s>", "\n", "[USER]", "[SYMBIO]", "[SYSTEM]"]

type
  TokenId = int32
  Pair = tuple[a, b: TokenId]
  WordData = tuple[tokens: seq[TokenId], count: int]

proc collapseWhitespace(text: string): string =
  # Entspricht bpe_forge.py: Replace(r"\s+", " ") - reduziert Whitespace-Laeufe
  # auf ein einzelnes Leerzeichen, damit "  " und "\t\t" nicht als eigene,
  # von echtem Inhalt verschiedene Woerter im Vokabular landen.
  result = newStringOfCap(text.len)
  var inWs = false
  for r in runes(text):
    let c = $r
    if c == " " or c == "\t" or c == "\r":
      if not inWs: result.add(' ')
      inWs = true
    else:
      result.add(c)
      inWs = (c == "\n")   # \n bleibt als eigenes Zeichen erhalten (Special Token)

proc preTokenize(text: string): seq[string] =
  var res: seq[string] = @[]
  var cur = ""
  for r in runes(text):
    let c = $r
    if c == " " or c == "\n" or c == "\t" or c == "\r":
      if cur != "":
        res.add(cur)
        cur = ""
    elif c in [".", ",", "!", "?", "\"", "'", ":", ";", "(", ")", "[", "]", "{", "}", "-", "_", "=", "+", "<", ">", "/", "\\", "|", "*", "&", "^", "%", "$", "#", "@", "~", "`"]:
      if cur != "":
        res.add(cur)
        cur = ""
      res.add(c)
    else:
      cur &= c
  if cur != "": res.add(cur)
  return res

proc preTokenizeLeer(text: string): seq[string] =
  # Update 2026-10-10: Vortokenisierung GENAU wie SnaiTokenizer::encode sie im
  # Betrieb bildet - getrennt wird nur an Leerzeichen und Zeilenumbruch,
  # Satzzeichen bleiben am Wort, und der Metaspace steht nur vor Woertern,
  # denen ein Leerzeichen vorausgeht (nicht am Zeilenanfang). Der Modus "wort"
  # (preTokenize oben) trennt Satzzeichen als eigene Woerter ab und setzt vor
  # jedes einen Metaspace; Merges wie "▁," oder "▁chinesisch" hinter "(" lernt
  # er aus Wortgrenzen, die der Tokenizer nie bildet. Das Wort traegt seinen
  # Metaspace hier selbst als erstes Zeichen.
  var res: seq[string] = @[]
  var cur = ""
  var lineStart = true
  var curMeta = false
  for r in runes(text):
    let c = $r
    if c == "\n":
      if cur != "" or curMeta: res.add((if curMeta: MetaSpace else: "") & cur)
      cur = ""; curMeta = false; lineStart = true
    elif c == " ":
      if cur != "" or curMeta: res.add((if curMeta: MetaSpace else: "") & cur)
      cur = ""; curMeta = true; lineStart = false
    else:
      cur &= c
  if cur != "" or curMeta: res.add((if curMeta: MetaSpace else: "") & cur)
  return res

proc main() =
  let startTime = cpuTime()
  let inputDir = if paramCount() >= 1: paramStr(1) else: "."
  let outputFile = if paramCount() >= 2: paramStr(2) else: "symbio_vocab.json"
  # Vortokenisierung: "leer" (Default, wie der Tokenizer im Betrieb, siehe
  # preTokenizeLeer) oder "wort" (trennt Satzzeichen ab, reproduziert die
  # Vokabulare vor 2026-10). Gemessen 2026-10-10: "leer" braucht 4,3 % weniger
  # Token und zaehlt dieselben Wortgrenzen, die der Tokenizer bildet.
  let pretok = if paramCount() >= 3: paramStr(3) else: "leer"
  if pretok != "wort" and pretok != "leer":
    echo " Unbekannte Vortokenisierung '", pretok, "' - erlaubt sind 'wort' und 'leer'."
    quit(1)
  echo " [Forge] Vortokenisierung: ", pretok
  var files: seq[string] = @[]

  for path in walkDirRec(inputDir):
    if path.endsWith(".php") or path.endsWith(".txt"):
      files.add(path)

  if files.len == 0:
    echo " Keine Dateien gefunden. Bitte Trainingsdaten (*.php, *.txt) bereitstellen."
    quit(1)

  echo " [Forge] Starte High-Speed BPE Training (HF-Compliant, inkrementell) mit ", files.len, " Dateien..."

  var vocabStr: seq[string] = @[]
  var vocabMap = initTable[string, TokenId]()
  var mergesList: seq[string] = @[]

  proc getId(s: string): TokenId =
    if vocabMap.hasKey(s): return vocabMap[s]
    result = vocabStr.len.TokenId
    vocabStr.add(s)
    vocabMap[s] = result

  for st in specialTokens: discard getId(st)
  let metaId = getId(MetaSpace)
  # Byte-Rueckfall (SentencePiece-Stil): feste IDs direkt hinter den
  # Sondertoken, damit <unk> = 0 und die Sondertoken-IDs unveraendert bleiben.
  var byteIds: array[256, TokenId]
  for b in 0 .. 255:
    byteIds[b] = getId("<0x" & toHex(b, 2) & ">")
  let byteLo = byteIds[0]
  let byteHi = byteIds[255]

  echo "  > Lese Daten, kollabiere Whitespace, pre-tokenisiere und zaehle Frequenzen..."
  var rawWordCounts = initCountTable[string]()
  for f in files:
    if pretok == "leer":
      # Kein Kollabieren: der Tokenizer kollabiert auch nicht. Der Tokenizer
      # stellt dem Text ein Leerzeichen voran, wenn er nicht mit ' ', '[' oder
      # '<' beginnt - hier je Datei nachgebildet.
      var text = readFile(f)
      if text.len > 0 and text[0] != ' ' and text[0] != '[' and text[0] != '<': text = " " & text
      for w in preTokenizeLeer(text):
        rawWordCounts.inc(w)
    else:
      let text = collapseWhitespace(readFile(f))
      for w in preTokenize(text):
        rawWordCounts.inc(MetaSpace & w)

  var charCounts = initCountTable[string]()
  for wordStr, freq in rawWordCounts:
    for r in runes(wordStr):
      if $r != MetaSpace: charCounts.inc($r, freq)
  charCounts.sort()
  var totalChars = 0
  for _, n in charCounts: totalChars += n
  var keptChars = initHashSet[string]()
  var covered = 0
  var droppedKinds = 0
  var droppedOcc = 0
  for c, n in charCounts:
    if covered.float < CharCoverage * totalChars.float:
      keptChars.incl(c)
      covered += n
    else:
      inc droppedKinds
      droppedOcc += n
  echo "  > Grundalphabet: ", keptChars.len, " Zeichen decken ", covered, " von ", totalChars,
       " Vorkommen; ", droppedKinds, " seltene Zeichen (", droppedOcc, " Vorkommen) gehen als Bytes."

  echo "  > Konvertiere Strings in Integer-Arrays (Cache-Locality Optimierung)..."
  # Grundalphabet in fester (sortierter) Reihenfolge anlegen: die IDs der
  # Einzelzeichen hingen sonst an der Iterationsreihenfolge der Wort-Tabelle.
  block:
    var sortedChars: seq[string] = @[]
    for c in keptChars: sortedChars.add(c)
    sortedChars.sort()
    for c in sortedChars: discard getId(c)
  var wordList: seq[WordData] = @[]

  for wordStr, freq in rawWordCounts:
    var toks: seq[TokenId] = @[]
    for r in runes(wordStr):
      let c = $r
      if c == MetaSpace:
        toks.add(metaId)
      elif c in keptChars:
        toks.add(getId(c))
      else:
        for ch in c: toks.add(byteIds[ord(ch)])
    wordList.add((tokens: toks, count: freq))
  # Update 2026-10-10: die Zeichenketten-Tabelle wird ab hier nicht mehr
  # gebraucht. reset statt clear: clear behaelt die Kapazitaet.
  reset(rawWordCounts)

  echo "  > Start-Vokabular: ", vocabStr.len, " Basis-Tokens. Beginne inkrementelle BPE-Merges (Ziel: ", VocabSize, ")..."

  var pairCounts = initCountTable[Pair]()
  # Update 2026-10-10: Wortliste je Paar als seq[int32] statt HashSet[int].
  # Am ganzen Produktivkorpus (1,3 Mrd. Zeichen) brauchte die HashSet-Fassung
  # schon beim Start der Merges ueber 6,7 GB und wurde vom Kernel beendet. Die
  # Liste darf ein Wort mehrfach nennen; lastSeen sorgt dafuer, dass jedes Wort
  # je Merge genau einmal bearbeitet wird (wie vorher ueber die Menge).
  var pairWords = initTable[Pair, seq[int32]]()
  var lastSeen = newSeq[int32](wordList.len)
  for i in 0 ..< lastSeen.len: lastSeen[i] = -1

  proc regPair(p: Pair, widx: int, delta: int) =
    # Byte-Token werden nie gemergt: der Tokenizer setzt sie erst NACH dem BPE
    # fuer unbekannte Zeichen ein, ein Merge darauf waere unerreichbar.
    if (p.a >= byteLo and p.a <= byteHi) or (p.b >= byteLo and p.b <= byteHi): return
    pairCounts.inc(p, delta)
    if delta > 0:
      let lst = addr pairWords.mgetOrPut(p, @[])
      if lst[].len == 0 or lst[][^1] != widx.int32: lst[].add(widx.int32)

  for widx in 0 ..< wordList.len:
    let w = wordList[widx].tokens
    let cnt = wordList[widx].count
    for i in 0 ..< w.len - 1:
      regPair((w[i], w[i+1]), widx, cnt)

  var mergeSteps = 0
  while vocabStr.len < VocabSize:
    var bestFreq = 0
    var bestPair: Pair = (-1.TokenId, -1.TokenId)
    # Update 2026-10-10: Gleichstand wird nach den Token-Zeichenketten
    # entschieden statt nach der Iterationsreihenfolge der Hashtabelle. Vorher
    # hing die Merge-Reihenfolge an Dingen, die nichts mit dem Korpus zu tun
    # haben (gemessen: dieselbe Zaehlung, andere Schluesselform der Woerter ->
    # 57 von 6727 Merges anders).
    for p, freq in pairCounts:
      if freq > bestFreq or (freq == bestFreq and freq > 0 and
          (vocabStr[p.a], vocabStr[p.b]) < (vocabStr[bestPair.a], vocabStr[bestPair.b])):
        bestFreq = freq
        bestPair = p

    if bestFreq <= 0:
      echo "  > [Stop] Keine mergbaren Paare mehr uebrig bei ", vocabStr.len,
           " Tokens (Korpus zu klein/repetitiv fuer Ziel ", VocabSize, ")."
      break

    let strA = vocabStr[bestPair.a]
    let strB = vocabStr[bestPair.b]
    mergesList.add(strA & " " & strB)

    # Update 2026-10-10: zwei verschiedene Merge-Wege koennen dieselbe
    # Zeichenkette ergeben ("ab"+"c" und "a"+"bc"). Vorher bekam sie dann zwei
    # IDs, die JSON-Tabelle aber nur einen Eintrag: das Vokabular hatte weniger
    # als VocabSize Eintraege, waehrend IDs bis VocabSize-1 vergeben waren
    # (gemessen: 8189 statt 8192; getVocabSize() zaehlt die Eintraege). Jetzt
    # wird die vorhandene ID wiederverwendet, der Merge bleibt in der Liste.
    let newStr = strA & strB
    var newId: TokenId
    if vocabMap.hasKey(newStr):
      newId = vocabMap[newStr]
    else:
      newId = vocabStr.len.TokenId
      vocabStr.add(newStr)
      vocabMap[newStr] = newId

    var affected: seq[int32]
    discard pairWords.pop(bestPair, affected)
    for w32 in affected:
      let widx = w32.int
      if lastSeen[widx] == mergeSteps.int32: continue
      lastSeen[widx] = mergeSteps.int32
      let oldToks = wordList[widx].tokens
      let cnt = wordList[widx].count
      let tLen = oldToks.len
      if tLen < 2: continue

      var newToks = newSeqOfCap[TokenId](tLen)
      var isNew = newSeqOfCap[bool](tLen)
      # AUDIT-FIX (aus snai_sync_legacy uebernommen 2026-10-10, BUGFIXES §5.1):
      # prevWasMerge verhindert einen Phantom-Dekrement, wenn dasselbe Paar
      # zweimal direkt hintereinander im selben Wort vorkommt (A,B,A,B). Ohne
      # das zeigt newToks[^1] beim zweiten Treffer auf das gerade verschmolzene
      # Token und dekrementiert ein Paar, das nie gezaehlt wurde.
      var prevWasMerge = false
      var i = 0
      while i < tLen:
        if i < tLen - 1 and oldToks[i] == bestPair.a and oldToks[i+1] == bestPair.b:
          if newToks.len > 0 and not prevWasMerge:
            regPair((newToks[^1], oldToks[i]), widx, -cnt)
          if i + 2 < tLen:
            regPair((oldToks[i+1], oldToks[i+2]), widx, -cnt)
          newToks.add(newId)
          isNew.add(true)
          prevWasMerge = true
          i += 2
        else:
          newToks.add(oldToks[i])
          isNew.add(false)
          prevWasMerge = false
          i += 1
      wordList[widx].tokens = newToks

      # Update 2026-10-10: neu gezaehlt werden nur Paare an Stellen, die DIESER
      # Merge erzeugt hat. Der Vergleich mit newId zaehlte doppelt, sobald
      # newId eine wiederverwendete ID ist (siehe oben) und das Wort das Token
      # schon aus einem frueheren Merge enthaelt. -d:altreg baut die alte
      # Bedingung (nur fuer die Gegenprobe der Schlusspruefung unten).
      for j in 0 ..< newToks.len - 1:
        when defined(altreg):
          let neuHier = newToks[j] == newId or newToks[j+1] == newId
        else:
          let neuHier = isNew[j] or isNew[j+1]
        if neuHier:
          regPair((newToks[j], newToks[j+1]), widx, cnt)

    pairCounts[bestPair] = 0

    inc mergeSteps
    if vocabStr.len mod 1000 == 0:
      echo "  > Fortschritt: ", vocabStr.len, " / ", VocabSize, " Tokens erreicht..."

  # Update 2026-10-10: Schlusspruefung. Die inkrementell gefuehrte Paarzaehlung
  # muss am Ende dieselbe sein wie eine Neuzaehlung ueber alle Woerter; sonst
  # war die Merge-Rangfolge irgendwo falsch. Kostet einen Durchgang.
  block:
    var soll = initCountTable[Pair]()
    for wd in wordList:
      for i in 0 ..< wd.tokens.len - 1:
        let p: Pair = (wd.tokens[i], wd.tokens[i+1])
        if (p.a >= byteLo and p.a <= byteHi) or (p.b >= byteLo and p.b <= byteHi): continue
        soll.inc(p, wd.count)
    var abw = 0
    for p, n in soll:
      if pairCounts.getOrDefault(p) != n: inc abw
    for p, n in pairCounts:
      if n != 0 and not soll.hasKey(p): inc abw
    echo "  > [Pruefung] Paarzaehlung gegen Neuzaehlung: ", abw, " Abweichungen bei ", soll.len, " Paaren."
    if abw > 0:
      echo " Abbruch: die inkrementelle Paarzaehlung ist falsch, das Vokabular wird nicht geschrieben."
      quit(3)

  var addedTokens = newJArray()
  for i, st in specialTokens:
    let tokenObj = %* {
      "id": i,
      "content": st,
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
    addedTokens.add(tokenObj)

  var vocabNode = newJObject()
  for i, s in vocabStr:
    vocabNode[s] = newJInt(i)

  var mergesNode = newJArray()
  for m in mergesList:
    mergesNode.add(newJString(m))

  let rootNode = %* {
    "version": "1.0",
    "truncation": newJNull(),
    "padding": newJNull(),
    "added_tokens": addedTokens,
    "model": {
      "type": "BPE",
      "dropout": newJNull(),
      "unk_token": "<unk>",
      "continuing_subword_prefix": newJNull(),
      "end_of_word_suffix": newJNull(),
      "fuse_unk": false,
      "byte_fallback": true,
      "vocab": vocabNode,
      "merges": mergesNode
    }
  }

  writeFile(outputFile, rootNode.pretty())

  let duration = round(cpuTime() - startTime, 2)
  echo "✅ Vokabular erfolgreich in ", outputFile, " gespeichert (", duration, "s)."
  echo "   > Vocab-Size: ", vocabStr.len, " / Ziel: ", VocabSize
  echo "   > HF-Merges aufgezeichnet: ", mergesList.len
  echo "   > Special-Tokens geschuetzt: ", specialTokens.join(", ")

main()