<?php
/**
 * Dateiname: src/korpus_sanitize.php
 * Funktion:  Gemeinsame Korpus-Bereinigung fuer ALLE Schreiber nach
 *            factory/input.
 * =========================================================================
 * WARUM DAS EINE GEMEINSAME DATEI IST
 *
 * Am 01.09. wurde der Bestand einmal vollstaendig bereinigt
 * (retrofit_korpus_20260901.php: 17.553 Dateien, 38,5 Mio. Zeichen,
 * 17,7 Mio. Entities). Zwei Stunden spaeter war er teilweise wieder
 * verunreinigt - der Retrofit reinigt den Vorrat, nicht die Quelle.
 *
 * Es gibt DREI Schreiber nach factory/input, und sie wurden nacheinander
 * gefunden statt auf einmal:
 *
 *   1. snai_gossip_node.php  - Import vom Laptop, der keinen Code-Sync hat
 *                              und daher dauerhaft altes Material liefert.
 *                              Abgedichtet am 01.09. 12:32.
 *   2. collect_definitions.php - schreibt LOKAL, geht am Node vorbei.
 *                              Erst am 01.09. abends aufgefallen, weil eine
 *                              Datei von 14:05 im Nachlauf auftauchte,
 *                              obwohl der Node laengst bereinigte. Gemessen
 *                              0,022 % unk.
 *   3. collect_gemma.php     - schreibt lokal, ist aber gemessen sauber
 *                              (0,000 % unk ueber 163.863 Token, weil das
 *                              Material per API erzeugt und nicht aus
 *                              Wikipedia gezogen wird).
 *
 * Der Gossip-Node hatte die Tabelle zuerst inline. Eine zweite Kopie in
 * collect_definitions.php waere garantiert auseinandergelaufen - genau die
 * Sorte Divergenz, die den Clipping-Fix vom 18.08. in nur einer von vier
 * Projektkopien hat landen lassen. Deshalb steht sie ab hier EINMAL.
 *
 * DIE ZEICHENKARTE SELBST kommt aus diagnose/ersetzungskarte.json und ist
 * EMPIRISCH gebaut: diagnose/build_ersetzungskarte.php prueft jedes im
 * Korpus vorkommende Zeichen gegen den echten Tokenizer (allein und im
 * Wortinneren) und nimmt nur auf, was nachweislich unk erzeugt. Fehlt die
 * Datei, bereinigt diese Funktion nur Entities und Sequenzen und meldet das
 * ueber korpus_sanitize_bereit() - sie bricht NICHT ab, denn ein Sammler
 * soll nicht sterben, weil eine Diagnosedatei fehlt.
 *
 * Nutzung:
 *   require_once __DIR__ . '/korpus_sanitize.php';
 *   $text = korpus_bereinigen($text);
 */
declare(strict_types=1);

/**
 * Baut die Ersetzungstabelle einmal und haelt sie.
 * Rueckgabe: array<string,string> fuer strtr().
 */
function korpus_sanitize_tabelle(): array
{
    static $tabelle = null;
    if ($tabelle !== null) return $tabelle;

    // Buchstabenpaare und Entities. strtr() nimmt immer die LAENGSTE passende
    // Zeichenkette, "a\u{0364}" gewinnt also von selbst gegen das einzelne
    // "\u{0364}" - die Reihenfolge muss nicht gestaffelt werden.
    $fest = [
        // Fruehneuhochdeutscher Druck: aͤ oͤ uͤ stehen fuer die Umlaute.
        "a\u{0364}"=>'ä', "o\u{0364}"=>'ö', "u\u{0364}"=>'ü',
        "A\u{0364}"=>'Ä', "O\u{0364}"=>'Ö', "U\u{0364}"=>'Ü',
        // Zerlegtes Unicode (NFD) statt vorkomponiert.
        "a\u{0308}"=>'ä', "o\u{0308}"=>'ö', "u\u{0308}"=>'ü',
        "A\u{0308}"=>'Ä', "O\u{0308}"=>'Ö', "U\u{0308}"=>'Ü',
        "s\u{0327}"=>'s', "t\u{0327}"=>'t',
        // HTML-Entities. 17,4 Mio. &nbsp; lagen im Bestand, 539 je MB in GEN.
        '&nbsp;'=>' ', '&#160;'=>' ', '&thinsp;'=>' ', '&emsp;'=>' ',
        '&ensp;'=>' ', '&#8239;'=>' ', '&#8201;'=>' ',
        '&shy;'=>'', '&#173;'=>'', '&lrm;'=>'', '&rlm;'=>'',
        '&zwnj;'=>'', '&zwj;'=>'',
        '&#8209;'=>'-', '&ndash;'=>'-', '&mdash;'=>'-',
        '&quot;'=>'"', '&apos;'=>"'", '&lt;'=>'<', '&gt;'=>'>', '&amp;'=>'&',
        // Bekanntes Korpusartefakt aus dem Truncation-Bug (gefixt 15.08.,
        // kann ueber ungepatchte Sammler weiterhin hereinkommen).
        '... [Gekürzt]'=>'',
    ];

    $karte = [];
    $pfad = __DIR__ . '/diagnose/ersetzungskarte.json';
    if (is_file($pfad)) {
        $j = json_decode((string)file_get_contents($pfad), true);
        if (is_array($j['karte'] ?? null)) $karte = $j['karte'];
    }

    // $fest links: bei Schluesselkollision gewinnt die feste Tabelle.
    $tabelle = $fest + $karte;
    return $tabelle;
}

/** Steht die empirische Zeichenkarte zur Verfuegung? Fuer Logmeldungen. */
function korpus_sanitize_bereit(): bool
{
    return is_file(__DIR__ . '/diagnose/ersetzungskarte.json');
}

/**
 * Bereinigt Korpustext. Verlustbehaftet und beabsichtigt:
 * ersetzt Zeichen, die der Tokenizer nicht darstellen kann, durch solche,
 * die er darstellen kann. Der Text bleibt vollstaendig - es wird nichts
 * gefiltert, nur umgeschrieben.
 */
function korpus_bereinigen(string $text): string
{
    return strtr($text, korpus_sanitize_tabelle());
}
