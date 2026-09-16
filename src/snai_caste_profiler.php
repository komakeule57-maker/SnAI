<?php
/**
 * SYMBIO NANO-AI FRAMEWORK (Code-4372)
 * Modul: Hardware Profiler & Caste System
 * -------------------------------------------------------------------------
 * Aufgabe: System-Ressourcen auslesen und biologische Kaste zuweisen.
 * FIX: RAM-Toleranz für Linux-Kernel-Reservierungen angepasst.
 */

class SnaiHardwareProfiler {
    
    public static function get_profile(): array {
        $os = strtoupper(substr(PHP_OS, 0, 3));
        $cores = 1;
        $ram_mb = 1024; // Fallback

        // 1. CPU Cores auslesen
        if ($os === 'WIN') {
            $cores = (int)@shell_exec('echo %NUMBER_OF_PROCESSORS%');
        } else {
            $cores = (int)@shell_exec('nproc 2>/dev/null'); // Linux
            if (!$cores) $cores = (int)@shell_exec('sysctl -n hw.ncpu 2>/dev/null'); // Mac Fallback
        }

        // 2. RAM auslesen
        if ($os === 'WIN') {
            $ram_out = @shell_exec('wmic os get TotalVisibleMemorySize /Value 2>nul');
            if (preg_match('/=(\d+)/', $ram_out, $m)) $ram_mb = (int)($m[1] / 1024);
        } else {
            if (is_readable('/proc/meminfo')) {
                $meminfo = file_get_contents('/proc/meminfo');
                if (preg_match('/MemTotal:\s+(\d+)\s+kB/', $meminfo, $m)) $ram_mb = (int)($m[1] / 1024);
            } else {
                $ram_bytes = (int)@shell_exec('sysctl -n hw.memsize 2>/dev/null'); // Mac Fallback
                if ($ram_bytes) $ram_mb = (int)($ram_bytes / 1024 / 1024);
            }
        }

        // Sicherheits-Fallbacks
        if ($cores <= 0) $cores = 1;
        if ($ram_mb <= 0) $ram_mb = 1024;

        // 3. Kasten-Logik (Die Biologie des Myzeliums)
        $caste = 'SCOUT'; // Standard: Schwache Hardware -> Darf nur jagen und funken.
        
        // Ab 2 Kernen und 2GB RAM -> Darf FP16 trainieren
        if ($cores >= 24 && $ram_mb >= 20480) {
            $caste = 'WORKER'; 
        }
        
        // Ab 4 Kernen und ~6GB RAM (6000 MB toleriert Linux/GPU Reservierungen bei 8GB Riegeln) -> Royal Distillery
        if ($cores >= 40 && $ram_mb >= 60000) {
            $caste = 'QUEEN_ELIGIBLE'; 
        }

        return [
            'os' => PHP_OS,
            'cores' => $cores,
            'ram_mb' => $ram_mb,
            'caste' => $caste
        ];
    }
}