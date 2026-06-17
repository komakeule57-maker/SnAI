# Symbiotic Nano-AI

Proof of Concept: Dieses Framework ist der lebende Beweis, dass man nicht auf Python angewiesen ist, um eine schnelle, hardwareagnostische KI-Architektur zu bauen. Es entkoppelt das neuronale Training komplett von gängigen pip-Abhängigkeiten und Virtual Environments.

Ein dezentrales, ressourceneffizientes Framework für das Training, die Synthese und die Inferenz von Large Language Models (LLMs) auf Consumer-Hardware.

Das System kombiniert die Flexibilität von PHP mit der nackten Performance einer C/Nim-Engine via Zero-Copy FFI (Foreign Function Interface). Es ermöglicht verteiltes Lernen (Federated Averaging) über P2P-Netzwerke und dynamisches Modell-Wachstum durch Network Morphism (Net2Net).

# 🚀 Kern-Features

P2P Schwarm-Training (Federated Averaging): Verteiltes Lernen auf mehreren PCs. Das System synchronisiert Trainingsdaten und verschmilzt lokal trainierte Gewichte vollautomatisch über das lokale Netzwerk. (Hinweis: Dies ist echtes Full-Parameter-Training. Es handelt sich hierbei NICHT um LoRA, DoRA oder andere PEFT-Adapter, sondern um das Verschmelzen vollständiger, dichter Matrizen).

Dezentrale Infrastruktur: Kommunikation erfolgt direkt via UDP/TCP (Gossip-Protokoll), komplett ohne zentralen Server oder Cloud-Abhängigkeit.

Dezentraler Schwarm: P2P-Synchronisation von Tensoren via UDP/TCP ohne zentralen Server.

Network Morphism (Forge): Verlustfreies Upscaling von Basismodellen (z.B. 128 Dimensionen) auf höhere Architekturen (256, 512, 1024+).

Autonome Steuerung: Ein Daemon überwacht Rohdaten, trainiert automatisiert Basis-Drohnen und verschmilzt diese bei kritischer Masse zu größeren Netzen.

Zero-Copy Inferenz: Direkte Speicheradressierung der Tensoren in den RAM/VRAM unter Umgehung des PHP-Overheads.

# 🛠 Systemanforderungen

OS: Linux (Ubuntu/Debian/Arch/Manjaro)

PHP: 8.1 oder höher (CLI-Modus). WICHTIG: In der php.ini muss ffi.enable=true gesetzt sein!

Nim Compiler: Version 1.6+ (zur Kompilierung der JIT-Engine).

OpenCL (Optional): Für GPU-beschleunigte Inferenz und Training.

# 📦 Installation

Repository klonen:

git clone [https://github.com/DeinName/symbio-nano-ai.git](https://github.com/DeinName/symbio-nano-ai.git)
cd symbio-nano-ai


Berechtigungen setzen und Engine kompilieren:

chmod +x symbio.sh
./symbio.sh init


Dieser Befehl erstellt die Ordnerstruktur (/lib, /factory/..) und kompiliert die Nim-Sourcen (core/cortex.nim) zur hochperformanten C-Bibliothek lib/libcortex.so.

# ⚙️ Nutzung

Das Framework kann auf zwei Arten betrieben werden: Manuell über das CLI oder vollautonom im Schwarm-Modus.

Modus 1: Vollautonomer Schwarm (Empfohlen)

Startet den P2P-Gossip-Node und den Autonomie-Daemon.

./symbio.sh swarm


Ablauf:

Lege rohe Textdateien (Nektar) im Format [SYMBIO]\n...Text...\n</s> im Ordner factory/input/ ab.

Der Daemon erkennt die Daten, trainiert eine Basis-Drohne (128-Dim) und legt sie in factory/experts/ ab.

Sobald 3 Drohnen existieren, verschmilzt der Daemon diese automatisch, skaliert sie auf 256-Dim und erschafft ein größeres Modell in factory/royal/.

Modus 2: Manuelle Kontrolle (CLI)

Für gezieltes Training, Upscaling oder Tensor-Verschmelzungen.

Modell trainieren (BPTT):
Befehl:

./symbio.sh cli train <input_text.txt> <output_modell.snai> <ziel_dimensionen>

Beispiel:
./symbio.sh cli train factory/input/data.txt factory/experts/drone.snai 512


Modelle verschmelzen (Federated Merge):

Befehl:

./symbio.sh cli merge <output.snai> <input1.snai> <input2.snai> ...

Beispiel:

./symbio.sh cli merge factory/checkpoints/merged.snai factory/experts/d1.snai factory/experts/d2.snai

Modell skalieren (Net2Net Upscale):

Befehl:

./symbio.sh cli upscale <input.snai> <output.snai> <ziel_dimension>

Beispiel:

./symbio.sh cli upscale factory/checkpoints/merged.snai factory/royal/princess.cortex 256


# 📂 Architektur / Verzeichnisstruktur

/core - Source-Code der Nim/C Tensor-Engine.

/src - Die PHP-Steuerlogik (Daemon, Node, Cluster-Master, CLI).

/lib - Die kompilierte libcortex.so und das Tokenizer-Vokabular.

/factory - Dynamischer I/O-Ordner für das Training.

/factory/input - Rohe Textdaten (.txt) für das Training.

/factory/archive - Verdaute Textdaten (werden für Kalibrierungen wiederverwendet).

/factory/experts - Generierte Basis-Drohnen (128-Dim).

/factory/royal - Synthetisierte Großmodelle 

# ⚖️ Lizenz & Kommerzielle Nutzung

Dieses Projekt ist Open Source und unter der GNU AGPLv3 lizenziert. Das bedeutet, dass du den Code für private, akademische oder Open-Source-Projekte frei nutzen, verändern und weitergeben darfst, solange du dich an die Bedingungen der AGPLv3 hältst (welche vorschreibt, dass abgeleitete Werke ebenfalls unter dieser Lizenz veröffentlicht werden müssen).

Für Unternehmen:
Wenn ihr diesen Code in einem kommerziellen oder proprietären Umfeld (inklusive SaaS) nutzen, integrieren oder verbreiten möchtet, ohne den strengen Open-Source-Zwang der AGPLv3 zu übernehmen, ist eine separate kommerzielle Lizenz erforderlich. Bitte kontaktiert mich für kommerzielle Anfragen.
