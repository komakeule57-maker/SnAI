<?php
// Dateiname: src/ram_checkpoint_config.php
// Funktion: zentrale Konfiguration fuer RAM-basierte (tmpfs) Ablage der
//           Worker-/Epochen-Checkpoints kleiner Tiers.
//
// WARUM: train_cluster.php schreibt pro Epoche pro Worker einen vollen
// Modell-Checkpoint (chunk_*/worker_*/cp_*_e*.snai) nach factory/checkpoints/
// - reines Wegwerfmaterial, das nach erfolgreichem Merge/Kalibrierung wieder
// verschwindet, aber durch die hohe Schreibfrequenz einen Grossteil der
// SSD-Schreiblast im Projekt verursacht (Nutzer-Sorge: TBW-Verschleiss
// billiger SSDs). Session-Entscheidung 2026-08-16: nur Drohne (128 Dim) und
// Prinzessin (256 Dim) auf tmpfs umziehen - die beiden Tiers, die den
// Grossteil dieser kurzlebigen Checkpoints erzeugen; Matriarchin aufwaerts
// bleibt bewusst auf der SSD (groessere Modelle -> mehr RAM-Bedarf, seltenere
// Epochen -> geringerer Wear-Anteil, siehe Session-Gespraech). /tmp ist auf
// beiden Maschinen bereits tmpfs (verifiziert 2026-08-16), kein Mount-Setup
// noetig.
//
// Bewusst NICHT betroffen: royal_distillery.php's eigene proto_*/expanded_*/
// calib_*-Dateien (dort $this->checkpoint_dir) - deutlich seltenere
// Schreibfrequenz (einmal pro Net2Net-Wachstumsschritt statt einmal pro
// Epoche), Aenderung dort haette mehr Aufrufstellen angefasst als der
// eigentliche Wear-Hotspot rechtfertigt.

define('SNAI_RAM_CHECKPOINT_DIR', '/tmp/snai_edge_checkpoints_ram/');
define('SNAI_RAM_CHECKPOINT_MAX_DIM', 256);

function snai_should_use_ram_checkpoints(int $dim): bool {
    return $dim <= SNAI_RAM_CHECKPOINT_MAX_DIM;
}
