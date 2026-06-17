# Dateiname: bpe_forge.nim
# Funktion: Trainiert den BPE-Tokenizer (Pure Nim, High Speed) und exportiert
# ein 100% Huggingface/SentencePiece-kompatibles JSON-Manifest.

#[
  SYMBIO NANO-AI FRAMEWORK
  Modul: Pure Nim Tokenizer Forge (Huggingface Compliant Edition)
]#

import os, strutils, tables, json, unicode, times, math

const
  VocabSize = 16384
  MetaSpace = " " # U+2581
  OutputFile = "symbio_vocab.json"

let specialTokens = ["<unk>", "<pad>", "<s>", "</s>", "\n", "[USER]", "[SYMBIO]", "[SYSTEM]"]

type
  TokenId = int32
  Pair = tuple[a, b: TokenId]
  WordData = tuple[tokens: seq[TokenId], count: int]

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

  echo " [Forge] Starte High-Speed BPE Training (HF-Compliant) mit ", files.len, " Dateien..."

  var vocabStr: seq[string] = @[]
  var vocabMap = initTable[string, TokenId]()
  var mergesList: seq[string] = @[] # VETO-FIX: Die essenzielle Regel-Historie!

  proc getId(s: string): TokenId =
    if vocabMap.hasKey(s): return vocabMap[s]
    result = vocabStr.len.TokenId
    vocabStr.add(s)
    vocabMap[s] = result

  for st in specialTokens: discard getId(st)

  echo "  > Lese Daten, pre-tokenisiere und zähle Frequenzen..."
  var rawWordCounts = initCountTable[string]()
  for f in files:
    let text = readFile(f)
    for w in preTokenize(text):
      rawWordCounts.inc(w)

  echo "  > Konvertiere Strings in Integer-Arrays (Cache-Locality Optimierung)..."
  var wordList: seq[WordData] = @[]
  discard getId(MetaSpace)

  for wordStr, freq in rawWordCounts:
    var toks: seq[TokenId] = @[]
    toks.add(getId(MetaSpace)) 
    for r in runes(wordStr):
      toks.add(getId($r))
    wordList.add((tokens: toks, count: freq))

  echo "  > Start-Vokabular: ", vocabStr.len, " Basis-Tokens. Beginne BPE-Merges (Ziel: ", VocabSize, ")..."

  while vocabStr.len < VocabSize:
    var pairCounts = initCountTable[Pair]()
    
    for w in mitems(wordList):
      let tLen = w.tokens.len
      if tLen < 2: continue
      for i in 0 ..< tLen - 1:
        pairCounts.inc((w.tokens[i], w.tokens[i+1]), w.count)

    var bestFreq = 0
    var bestPair: Pair = (-1.TokenId, -1.TokenId)
    
    for p, freq in pairCounts:
      if freq > bestFreq:
        bestFreq = freq
        bestPair = p

    if bestFreq == 0: break

    # Historie protokollieren für das HF "merges" Array
    let strA = vocabStr[bestPair.a]
    let strB = vocabStr[bestPair.b]
    mergesList.add(strA & " " & strB)

    let newId = vocabStr.len.TokenId
    let newStr = strA & strB
    vocabStr.add(newStr)
    vocabMap[newStr] = newId

    for w in mitems(wordList):
      let tLen = w.tokens.len
      if tLen < 2: continue
      
      var newToks = newSeqOfCap[TokenId](tLen)
      var i = 0
      while i < tLen:
        if i < tLen - 1 and w.tokens[i] == bestPair.a and w.tokens[i+1] == bestPair.b:
          newToks.add(newId)
          i += 2
        else:
          newToks.add(w.tokens[i])
          i += 1
      w.tokens = newToks

    if vocabStr.len mod 1000 == 0:
      echo "  > Fortschritt: ", vocabStr.len, " / ", VocabSize, " Tokens erreicht..."

  # ==============================================================================
  # 5. HF KOMPATIBLES JSON MANIFEST BAUEN
  # ==============================================================================
  
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
  echo "   > Vocab-Size: ", vocabStr.len
  echo "   > HF-Merges aufgezeichnet: ", mergesList.len
  echo "   > Special-Tokens geschützt: ", specialTokens.join(", ")

main()