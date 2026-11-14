<?php
// Dateiname: src/tensor_utils_ffi.php
// Funktion: FFI-Bruecke zu snai_tensor_utils.nim (libtensorutils.so).
//           Ersetzt die PHP-seitigen pack()/unpack()-Schleifen in
//           royal_distillery.php::merge_tributes()/upscale_matrix_stream()
//           durch native Aufrufe - gleiches Verhalten (inkl. der drei
//           Fuellwert-Faelle beim Upscale: Adam->0.0, Gewichte->Rauschen um
//           0.0, RMSNorm-Gains->Rauschen um 1.0 via pad_center), aber ohne
//           den Umweg ueber PHP-Arrays pro einzelnem Float.
//
// AUDIT-FIX: dieselbe libgomp-Exit-Race wie bei tensor_ffi_adapter.php /
// snai_inference.php (siehe deren Kommentare) - snai_tensor_utils.nim nutzt
// ebenfalls OpenMP ("||"-Schleifen). Ohne PASSIVE-Policy segfaultet der
// Prozess beim Beenden kurz nach dem letzten Kernel-Aufruf, weil libgomp
// seinen Worker-Thread-Pool noch im Uebergang in den Idle-Zustand hat.
// MUSS vor dem FFI::cdef()-Laden von libtensorutils.so gesetzt sein.
putenv('OMP_WAIT_POLICY=PASSIVE');

class SnaiTensorUtils {
    const SNAI_TYPE_FP32 = 1;
    const SNAI_TYPE_FP16 = 2;

    private static ?FFI $ffi = null;

    private static function ffi(): FFI {
        if (self::$ffi === null) {
            $lib_path = __DIR__ . '/../lib/libtensorutils.so';
            self::$ffi = FFI::cdef("
                void snai_tensor_merge(void* out_buf, void** in_buffers, int num_models, size_t elements, int dtype);
                void snai_tensor_upscale(void* in_buf, void* out_buf, int old_r, int old_c, int new_r, int new_c,
                                         float scale, float noise_factor, float pad_center, int is_adam, int dtype);
            ", $lib_path);
        }
        return self::$ffi;
    }

    /**
     * Mittelt N gleich lange Rohdaten-Buffer (Float32- oder Float16-Bytes)
     * elementweise. $buffers: Array von PHP-Binaerstrings, gleiche Laenge,
     * Laenge muss ein Vielfaches der Elementgroesse sein (4 Byte FP32 /
     * 2 Byte FP16). Rueckgabe: gemittelter Buffer, gleiche Laenge.
     */
    public static function merge(array $buffers, int $dtype = self::SNAI_TYPE_FP32): string {
        $num_models = count($buffers);
        if ($num_models === 0) return '';

        $bytes_per_elem = ($dtype === self::SNAI_TYPE_FP16) ? 2 : 4;
        $len = strlen($buffers[0]);
        $elements = intdiv($len, $bytes_per_elem);
        if ($elements === 0) return '';

        $ffi = self::ffi();

        // AUDIT-FIX: FFI::cast("void*", $chardata) auf ein char[N]-Array
        // interpretiert die Array-Repraesentation selbst als Zeiger-Wert
        // (Muell) statt einen Zeiger AUF die Daten zu liefern - SIGSEGV bei
        // der ersten echten Nutzung. Richtig ist FFI::addr() auf das erste
        // Element, das liefert einen echten char*/void* auf den Puffer.
        $in_cdata = [];
        $in_ptrs = $ffi->new("void*[{$num_models}]");
        for ($i = 0; $i < $num_models; $i++) {
            $in_cdata[$i] = $ffi->new("char[{$len}]"); // owned=true (Default): PHP raeumt selbst auf
            FFI::memcpy($in_cdata[$i], $buffers[$i], $len);
            $in_ptrs[$i] = FFI::cast("void*", FFI::addr($in_cdata[$i][0]));
        }

        $out_cdata = $ffi->new("char[{$len}]");
        $ffi->snai_tensor_merge($out_cdata, FFI::cast("void**", $in_ptrs), $num_models, $elements, $dtype);

        return FFI::string($out_cdata, $len);
    }

    /**
     * Expandiert eine Matrix (Net2Net-Upscale). $in_buffer: alte Matrix,
     * roh, old_r*old_c Elemente. Rueckgabe: neue Matrix, roh, new_r*new_c
     * Elemente. $pad_center: neutrales Element fuer frisch expandierte
     * Zellen (0.0 fuer Gewichte, 1.0 fuer RMSNorm-Gains) - wird bei
     * $is_adam=true ignoriert (Adam-Zustand bekommt immer exakt 0.0).
     */
    public static function upscale(
        string $in_buffer, int $old_r, int $old_c, int $new_r, int $new_c,
        float $scale, float $noise_factor, float $pad_center, bool $is_adam,
        int $dtype = self::SNAI_TYPE_FP32
    ): string {
        $bytes_per_elem = ($dtype === self::SNAI_TYPE_FP16) ? 2 : 4;
        $ffi = self::ffi();

        $in_len = strlen($in_buffer);
        $in_cdata = $ffi->new("char[{$in_len}]"); // owned=true (Default): PHP raeumt selbst auf
        if ($in_len > 0) FFI::memcpy($in_cdata, $in_buffer, $in_len);

        $out_len = $new_r * $new_c * $bytes_per_elem;
        $out_cdata = $ffi->new("char[{$out_len}]");

        $ffi->snai_tensor_upscale(
            $in_cdata, $out_cdata, $old_r, $old_c, $new_r, $new_c,
            $scale, $noise_factor, $pad_center, $is_adam ? 1 : 0, $dtype
        );

        return FFI::string($out_cdata, $out_len);
    }
}
