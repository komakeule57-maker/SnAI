<?php
// Dateiname: src/head_config.php
// Funktion: Persistiert num_heads NEBEN dem .cortex-File, weil das Binaerformat
//           selbst den Wert nicht speichert (siehe Warnung unten).
//
// WARUM EINE SIDECAR-DATEI STATT HEADER-ERWEITERUNG:
// num_heads bestimmt headSize = hidden_dim / num_heads, und headSize legt fest,
// wie RoPE rotiert und wie die Attention-Summen segmentiert werden. Das ist in
// den TRAINIERTEN GEWICHTEN von w_q/w_k/w_v faktisch eingebrannt, auch wenn die
// Zahl selbst nirgends in der Datei steht. Laedst du denselben Checkpoint mit
// einer anderen Head-Zahl, bleiben alle Floats unveraendert, aber ihre Bedeutung
// aendert sich komplett - strukturell derselbe Fehler wie der Net2Net-Gain-Bug,
// nur ohne das offensichtliche Loss-Symptom.
// Eine .heads-Sidecar-Datei loest das, ohne das Binaerformat, den Nim-Loader
// oder die Byte-Bilanz-Pruefung anzufassen (E>H: kleinster Eingriff, volle Wirkung).

function head_sidecar_path(string $model_path): string {
    return $model_path . '.heads';
}

/**
 * Zielwert headSize=64 (Standard seit dem Original-Transformer/GPT-2).
 * cortex_init() korrigiert Rundungsfehler ohnehin selbst (erzwingt gerade,
 * >=2), hier reicht die einfache Naeherung.
 */
function suggest_num_heads(int $dim, int $target_head_dim = 64): int {
    if ($dim < 2) return 1;
    $heads = max(1, (int) round($dim / $target_head_dim));
    return min($heads, intdiv($dim, 2));
}

/**
 * Ermittelt die Head-Zahl fuer $model_path.
 *  1. Sidecar vorhanden      -> deren Wert ist bindend, PUNKT. Nie aus $dim
 *     neu geraten, sonst genau der Bug, vor dem diese Datei schuetzt.
 *  2. Kein Sidecar (Altbestand, z.B. eine vor diesem Fix trainierte Datei)
 *     -> das Modell wurde bisher immer mit dem Konstruktor-Default geladen.
 *     Dieser Default ist 1 (siehe tensor_ffi_adapter.php). Wir uebernehmen
 *     also weiterhin 1, warnen aber laut, weil das vermutlich nicht gewollt
 *     war - und schreiben die Sidecar-Datei JETZT, damit der Zustand ab
 *     sofort fixiert ist und nicht bei jedem Lauf neu geraten wird.
 */
function resolve_num_heads(string $model_path, int $dim): int {
    $sidecar = head_sidecar_path($model_path);

    if (file_exists($sidecar)) {
        $heads = (int) trim(file_get_contents($sidecar));
        if ($heads >= 1) return $heads;
        echo "  > ⚠️ [Heads] Sidecar '$sidecar' ist unlesbar, falle auf Legacy-Wert zurueck.\n";
    }

    $legacy_heads = 1; // = bisheriger Konstruktor-Default in tensor_ffi_adapter.php
    echo "  > ⚠️ [Heads] Keine Sidecar-Datei fuer '" . basename($model_path) . "' gefunden.\n";
    echo "  >    Dieses Modell wurde bislang implizit mit $legacy_heads Head(s) geladen\n";
    echo "  >    (headSize=" . intdiv($dim, $legacy_heads) . "). Das wird JETZT festgeschrieben,\n";
    echo "  >    um es NICHT rueckwirkend zu aendern (wuerde bereits gelernte\n";
    echo "  >    Attention-Gewichte zerstoeren - selbe Fehlerklasse wie der\n";
    echo "  >    Net2Net-Gain-Bug). Fuer mehr Heads: gezielt migrieren, siehe\n";
    echo "  >    migrate_heads() weiter unten in dieser Datei.\n";

    write_num_heads($model_path, $legacy_heads);
    return $legacy_heads;
}

function write_num_heads(string $model_path, int $num_heads): void {
    file_put_contents(head_sidecar_path($model_path), (string) $num_heads);
}

/**
 * Bewusster, expliziter Wechsel der Head-Zahl fuer ein Modell.
 * Nur fuer zwei legitime Faelle gedacht:
 *   a) ein FRISCH initialisiertes Modell, das noch nie trainiert hat
 *      (kein gelerntes Wissen, das zerstoert werden koennte)
 *   b) ein bewusster Cut: "ich nehme den Qualitaetsverlust in Kauf und
 *      trainiere die Attention ab jetzt neu ein"
 * Ruft man diese Funktion NICHT auf, bleibt resolve_num_heads() stur beim
 * zuletzt festgeschriebenen Wert - das ist Absicht.
 */
function migrate_heads(string $model_path, int $new_heads, bool $i_understand_the_risk = false): void {
    if (!$i_understand_the_risk) {
        die("🚨 migrate_heads(): \$i_understand_the_risk muss explizit true sein.\n" .
            "   Ein Wechsel der Head-Zahl auf einem trainierten Checkpoint aendert\n" .
            "   die Bedeutung der Attention-Gewichte, nicht ihre Werte.\n");
    }
    $old = file_exists(head_sidecar_path($model_path))
        ? trim(file_get_contents(head_sidecar_path($model_path)))
        : "1 (implizit, kein Sidecar)";
    echo "  > 🔧 [Heads] Migriere '$model_path': $old -> $new_heads Heads.\n";
    write_num_heads($model_path, $new_heads);
}
