"""
Dateiname: bpe_forge.py
Funktion: Trainiert den BPE-Tokenizer, definiert das Vokabular und schützt 
die speziellen SFT-Steuerungs-Token vor der Fragmentierung.
-------------------------------------------------------------------------
Aufgabe: Trainiert einen Custom BPE-Tokenizer (16384 Tokens) aus eigenen Daten.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.normalizers import Sequence as NormSequence, NFKC, Replace
import glob

# 1. Konfiguration
VOCAB_SIZE = 16384
#VOCAB_SIZE = 8192
OUTPUT_FILE = "symbio_vocab.json"

# Sichere UTF-8 Repräsentation des LLaMA/Gemma Metaspaces
META_SPACE = "\u2581" 

# Trainingsdaten: Alles was wir an Code und Text finden
training_files = glob.glob("**/*.php", recursive=True) + glob.glob("**/*.txt", recursive=True)

if not training_files:
    print(" Keine Dateien gefunden. Bitte Trainingsdaten bereitstellen.")
    exit(1)

print(f"[Forge] Starte BPE Training (Gemma-Style) mit {len(training_files)} Dateien...")

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

tokenizer.normalizer = NormSequence([
    Replace(r"\s+", " "), 
    NFKC()
])

tokenizer.pre_tokenizer = Metaspace(replacement=META_SPACE)

# Das Metaspace MUSS ins Basis-Alphabet, damit eigenständige Leerzeichen eine ID bekommen!
base_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789äöüßÄÖÜ.,!?\"':;(){}[]-_<>@#$%^&*/\\+=|~ \n\t" + META_SPACE

# --- DIE SFT-KORSETT TOKEN ---
# Diese Tokens sind heilig. Der BPE-Algorithmus darf sie niemals zerschneiden.
# Sie dienen dem Netzwerk als absolute, eindeutige State-Trigger (System, Input, Output).
special_tokens = [
    "<unk>",     # 0: Unbekanntes Zeichen
    "<pad>",     # 1: Padding
    "<s>",       # 2: Sequence Start
    "</s>",      # 3: Sequence End / Turn End
    "\n",        # 4: Harter Zeilenumbruch
    "[USER]",    # 5: SFT Input Marker
    "[SYMBIO]",  # 6: SFT Output Marker
    "[SYSTEM]"   # 7: Optionaler System-Prompt Marker
]

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=special_tokens,
    show_progress=True,
    initial_alphabet=list(base_chars)
)

tokenizer.train(training_files, trainer)
tokenizer.save(OUTPUT_FILE)

print(f"✅ Vokabular erfolgreich in {OUTPUT_FILE} gespeichert.")
print(f"   > Vocab-Size: {VOCAB_SIZE}")
print(f"   > Special-Tokens geschützt: {', '.join(special_tokens)}")
