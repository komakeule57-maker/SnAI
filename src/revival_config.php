<?php
/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Neuron-Revival-Konfiguration (Ein-/Ausschalter fuer den Live-Hook)
 * -------------------------------------------------------------------------
 * Bewusst eine JSON-Datei statt einer PHP-Konstante: PHP-Code wird von
 * beehive_core.php/train_cluster.php nur einmal beim Prozessstart geladen
 * (siehe CLAUDE.md/royal_distillery.php-Erfahrung mit dem synthesis.log-Fix -
 * eine Code-Aenderung griff dort erst nach einem Daemon-Neustart). Eine reine
 * Datenwert-Datei wird dagegen bei jedem Epochenende in train_cluster.php
 * frisch von der Platte gelesen (siehe load_revival_config()-Aufruf dort) -
 * "enabled": false in revival_config.json schaltet das Feature also SOFORT
 * ab der naechsten Epoche ab, ohne dass irgendein laufender Prozess neu
 * gestartet werden muss.
 *
 * Feature selbst: siehe diagnose/snai_neuron_revival.php (Erkennung+Revival
 * toter Neuronen per Zeilennorm-Median/MAD-Heuristik) - dieser Config-Loader
 * steuert nur, OB/WIE train_cluster.php es nach jeder Epoche aufruft.
 */

define('REVIVAL_CONFIG_PATH', __DIR__ . '/revival_config.json');

function load_revival_config(): array {
    $defaults = [
        'enabled'  => false,
        'tensors'  => 'w_up,w_q,w_k,w_v,w_o,w_down',
        'scale'    => 0.02,
        'cooldown' => 3,
    ];
    if (!file_exists(REVIVAL_CONFIG_PATH)) return $defaults;

    $raw = @file_get_contents(REVIVAL_CONFIG_PATH);
    $data = $raw !== false ? json_decode($raw, true) : null;
    if (!is_array($data)) return $defaults; // korrupte/kaputte Config -> sicherer Default (aus)

    return array_merge($defaults, $data);
}
