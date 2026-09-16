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

CUDA (Optional): Wird genutzt, wenn nvcc beim Bauen gefunden wird; sonst überspringt ./symbio.sh init das CUDA-Backend und OpenCL/CPU tragen allein.

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

Lege rohe Textdateien im Format [SYMBIO]\n...Text...\n</s> im Ordner factory/input/ ab.

Der Daemon erkennt die Daten, trainiert eine Basis-Drohne (128-Dim) und legt sie in factory/experts/ ab.

Sobald 3 Drohnen existieren, verschmilzt der Daemon diese automatisch, skaliert sie auf 256-Dim und erschafft ein größeres Modell in factory/royal/.

Modus 2: Manuelle Kontrolle (CLI)

Für gezieltes Training, Inferenz oder Destillation.

Modell trainieren (BPTT):
Befehl:

./symbio.sh cli train <input_text.txt> <output_modell.snai> [epochen]

Beispiel:
./symbio.sh cli train factory/input/data.txt factory/experts/drone.snai 20

Hinweis: Die Genesis-Dimension liegt fest bei 128 (Drohne); ein Dimensions-Argument gibt es nicht mehr. Höhere Stufen entstehen durch Destillation, nicht durch direktes Training.

Modell befragen (Zero-Copy Inferenz):

Befehl:

./symbio.sh cli infer <modell.snai> "<prompt>" [beam_width]

Beispiel:

./symbio.sh cli infer factory/experts/drone.snai "Die Bahnstrecke"

Ohne beam_width läuft Temperature/min-p-Sampling; ab 2 schaltet Beam-Search ein.

Modell destillieren (Federated Merge + Net2Net Upscale + Kalibrierung):

Befehl:

./symbio.sh cli distill [Prinzessin|Matriarchin|Empress|Demi Goddess|Goddess]

Beispiel:

./symbio.sh cli distill Prinzessin

Die früheren Einzelschritte cli merge und cli upscale gibt es nicht mehr. Verschmelzung, Expansion und die anschliessende Kalibrierung laufen als eine Einheit; die Zielarchitektur je Stufe steht in der EVOLUTION_LADDER in src/royal_distillery.php (Prinzessin 256 Dim / 3 Layer, Matriarchin 512 / 6, Empress 1024 / 12, ...). Für Prinzessin müssen 3 Drohnen in factory/tributes/ liegen - im autonomen Betrieb stellt der Daemon sie selbst dorthin.


# 📂 Architektur / Verzeichnisstruktur

/core - Source-Code der Nim/C Tensor-Engine.

/src - Die PHP-Steuerlogik (Daemon, Node, Cluster-Master, CLI).

/lib - Die kompilierte libcortex.so und das Tokenizer-Vokabular.

/factory - Dynamischer I/O-Ordner für das Training.

/factory/input - Rohe Textdaten (.txt) für das Training.

/factory/archive - Fertig trainierte Textdaten (werden für Kalibrierungen wiederverwendet).

/factory/experts - Generierte Basis-Drohnen (128-Dim).

/factory/royal - Synthetisierte Großmodelle 

# ⚖️ Lizenz & Kommerzielle Nutzung

Dieses Projekt ist Open Source und unter der GNU AGPLv3 lizenziert. Das bedeutet, dass du den Code für private, akademische oder Open-Source-Projekte frei nutzen, verändern und weitergeben darfst, solange du dich an die Bedingungen der AGPLv3 hältst (welche vorschreibt, dass abgeleitete Werke ebenfalls unter dieser Lizenz veröffentlicht werden müssen).

Für Unternehmen:
Wenn ihr diesen Code in einem kommerziellen oder proprietären Umfeld (inklusive SaaS) nutzen, integrieren oder verbreiten möchtet, ohne den strengen Open-Source-Zwang der AGPLv3 zu übernehmen, ist eine separate kommerzielle Lizenz erforderlich. Bitte kontaktiert mich für kommerzielle Anfragen.
