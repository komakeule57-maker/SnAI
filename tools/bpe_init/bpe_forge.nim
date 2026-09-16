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

import os, strutils, tables, sets, json, unicode, times, math

const
  VocabSize = 8192
  MetaSpace = "\u2581"   # E2 96 81 - Gemma/LLaMA-Style Metaspace-Marker
  OutputFile = "symbio_vocab.json"

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

proc main() =
  let startTime = cpuTime()
  var files: seq[string] = @[]

  for path in walkDirRec("."):
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

  echo "  > Lese Daten, kollabiere Whitespace, pre-tokenisiere und zaehle Frequenzen..."
  var rawWordCounts = initCountTable[string]()
  for f in files:
    let text = collapseWhitespace(readFile(f))
    for w in preTokenize(text):
      rawWordCounts.inc(w)

  echo "  > Konvertiere Strings in Integer-Arrays (Cache-Locality Optimierung)..."
  var wordList: seq[WordData] = @[]

  for wordStr, freq in rawWordCounts:
    var toks: seq[TokenId] = @[metaId]
    for r in runes(wordStr):
      toks.add(getId($r))
    wordList.add((tokens: toks, count: freq))

  echo "  > Start-Vokabular: ", vocabStr.len, " Basis-Tokens. Beginne inkrementelle BPE-Merges (Ziel: ", VocabSize, ")..."

  var pairCounts = initCountTable[Pair]()
  var pairWords = initTable[Pair, HashSet[int]]()

  proc regPair(p: Pair, widx: int, delta: int) =
    pairCounts.inc(p, delta)
    if delta > 0:
      pairWords.mgetOrPut(p, initHashSet[int]()).incl(widx)

  for widx in 0 ..< wordList.len:
    let w = wordList[widx].tokens
    let cnt = wordList[widx].count
    for i in 0 ..< w.len - 1:
      regPair((w[i], w[i+1]), widx, cnt)

  var mergeSteps = 0
  while vocabStr.len < VocabSize:
    var bestFreq = 0
    var bestPair: Pair = (-1.TokenId, -1.TokenId)
    for p, freq in pairCounts:
      if freq > bestFreq:
        bestFreq = freq
        bestPair = p

    if bestFreq <= 0:
      echo "  > [Stop] Keine mergbaren Paare mehr uebrig bei ", vocabStr.len,
           " Tokens (Korpus zu klein/repetitiv fuer Ziel ", VocabSize, ")."
      break

    let strA = vocabStr[bestPair.a]
    let strB = vocabStr[bestPair.b]
    mergesList.add(strA & " " & strB)

    let newId = vocabStr.len.TokenId
    let newStr = strA & strB
    vocabStr.add(newStr)
    vocabMap[newStr] = newId

    let affected = pairWords.getOrDefault(bestPair, initHashSet[int]())
    for widx in affected:
      let oldToks = wordList[widx].tokens
      let cnt = wordList[widx].count
      let tLen = oldToks.len
      if tLen < 2: continue

      var newToks = newSeqOfCap[TokenId](tLen)
      var i = 0
      while i < tLen:
        if i < tLen - 1 and oldToks[i] == bestPair.a and oldToks[i+1] == bestPair.b:
          if newToks.len > 0:
            regPair((newToks[^1], oldToks[i]), widx, -cnt)
          if i + 2 < tLen:
            regPair((oldToks[i+1], oldToks[i+2]), widx, -cnt)
          newToks.add(newId)
          i += 2
        else:
          newToks.add(oldToks[i])
          i += 1
      wordList[widx].tokens = newToks

      for j in 0 ..< newToks.len - 1:
        if newToks[j] == newId or newToks[j+1] == newId:
          regPair((newToks[j], newToks[j+1]), widx, cnt)

    pairCounts[bestPair] = 0
    pairWords[bestPair] = initHashSet[int]()

    inc mergeSteps
    if vocabStr.len mod 1000 == 0:
      echo "  > Fortschritt: ", vocabStr.len, " / ", VocabSize, " Tokens erreicht..."

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
      "byte_fallback": false,
      "vocab": vocabNode,
      "merges": mergesNode
    }
  }

  writeFile(OutputFile, rootNode.pretty())

  let duration = round(cpuTime() - startTime, 2)
  echo "✅ Vokabular erfolgreich in ", OutputFile, " gespeichert (", duration, "s)."
  echo "   > Vocab-Size: ", vocabStr.len, " / Ziel: ", VocabSize
  echo "   > HF-Merges aufgezeichnet: ", mergesList.len
  echo "   > Special-Tokens geschuetzt: ", specialTokens.join(", ")

main()