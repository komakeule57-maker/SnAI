<?php
// Dateiname: src/growth_config.php
// Funktion: Persistiert die Net2Net-Expansionsgrenze (alte Dim/FFN/Layer-Zahl)
//           NEBEN dem .apex/.snai-File, analog zu head_config.php's .heads-Sidecar.
//
// WARUM: nach einer Net2Net-Expansion (upscale_matrix_stream() in
// royal_distillery.php) besteht ein Grossteil jeder quadratisch wachsenden
// Matrix aus frisch verrauschten, nicht gelernten Werten (bei einer
// Dim-Verdopplung bis zu 75% der Zellen - siehe docs/ROADMAP.md, Abschnitt
// "Differenzierte Learning-Rate"). Damit der Adam-Optimizer diese Bereiche mit
// einer hoeheren effektiven Lernrate schneller einspielen kann, muss die
// Grenze (old_dim/old_ffn/old_layers) ueber alle Epochen einer Synthese hinweg
// bekannt bleiben - das Binaerformat selbst speichert sie nicht.

function growth_sidecar_path(string $model_path): string {
    return $model_path . '.growth';
}

function write_growth_sidecar(string $model_path, int $old_dim, int $old_ffn, int $old_layers): void {
    file_put_contents(growth_sidecar_path($model_path), json_encode([
        'old_dim' => $old_dim,
        'old_ffn' => $old_ffn,
        'old_layers' => $old_layers,
    ]));
}

/**
 * Liefert [old_dim, old_ffn, old_layers] oder null, falls keine Sidecar-Datei
 * existiert (z.B. Modell wurde nicht in dieser Session expandiert, oder ist
 * ein Genesis-Modell ohne Expansionshistorie). null bedeutet fuer den Aufrufer:
 * kein LR-Boost anfordern, exakt das bisherige Verhalten.
 */
function resolve_growth_boundary(string $model_path): ?array {
    $sidecar = growth_sidecar_path($model_path);
    if (!file_exists($sidecar)) return null;
    $data = json_decode(file_get_contents($sidecar), true);
    if (!is_array($data) || !isset($data['old_dim'], $data['old_ffn'], $data['old_layers'])) return null;
    return [(int)$data['old_dim'], (int)$data['old_ffn'], (int)$data['old_layers']];
}

function propagate_growth_sidecar(string $src_model, string $dst_model): void {
    $src = growth_sidecar_path($src_model);
    if (!file_exists($src)) return;
    @copy($src, growth_sidecar_path($dst_model));
}

function move_growth_sidecar(string $src_model, string $dst_model): void {
    $src = growth_sidecar_path($src_model);
    if (!file_exists($src)) return;
    @rename($src, growth_sidecar_path($dst_model));
}
