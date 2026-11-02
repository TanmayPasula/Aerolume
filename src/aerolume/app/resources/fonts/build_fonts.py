"""Rebuild the bundled TrueType fonts from the upstream WOFF2 files (maintainers only).

Why this exists: Qt's font database (``QFontDatabase.addApplicationFont``)
cannot read WOFF2 on any platform this PySide6 build supports - it returns -1
for every file - so the web fonts the Atkinson Hyperlegible projects publish
through Fontsource have to be converted to plain TrueType before the app can
use them. Variable fonts are also instanced to static weights, because static
faces match by weight identically on the Windows, offscreen (tests) and
FreeType font back ends, while a variable face's weight axis is honoured by
some of them only.

The conversion is lossless for the glyph data: WOFF2's table transforms are
undone by fontTools exactly as a browser would, and the instancer pins the
``wght`` axis at the named weights. Nothing is renamed or redrawn, so under
the SIL Open Font Licence this is a change of format, not a Modified Version.

Usage (needs fontTools and either the ``brotli`` module or Node.js >= 12 on
PATH for the Brotli decompression WOFF2 uses)::

    python build_fonts.py <fontsource-variable/atkinson-hyperlegible-next>
                          <fontsource-variable/atkinson-hyperlegible-mono>

It writes ``AtkinsonHyperlegibleNext-<Weight>.ttf`` and
``AtkinsonHyperlegibleMono-<Weight>.ttf`` next to this file and prints a
SHA-256 for each input and output, which README.md records.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (output family stem, Fontsource package folder index, weights to instance)
FAMILIES = (
    ("AtkinsonHyperlegibleNext", 0, {"Regular": 400, "Medium": 500, "SemiBold": 600, "Bold": 700}),
    ("AtkinsonHyperlegibleMono", 1, {"Regular": 400, "SemiBold": 600}),
)
SUBSETS = ("latin", "latin-ext")
#: the family name every weight carries (the instancer names the Mono family after its ExtraLight
#: default instance, and splits SemiBold / Medium into families of their own)
FAMILY_NAMES = {"AtkinsonHyperlegibleNext": "Atkinson Hyperlegible Next",
                "AtkinsonHyperlegibleMono": "Atkinson Hyperlegible Mono"}


def _install_brotli_shim() -> None:
    """Give fontTools a ``brotli.decompress`` backed by Node's zlib when the module is missing."""
    try:
        import brotli  # noqa: F401
        return
    except ImportError:
        pass

    def decompress(data: bytes) -> bytes:
        js = ("const z=require('zlib');const c=[];process.stdin.on('data',d=>c.push(d));"
              "process.stdin.on('end',()=>process.stdout.write(z.brotliDecompressSync(Buffer.concat(c))));")
        return subprocess.run(["node", "-e", js], input=data, capture_output=True, check=True).stdout

    shim = types.ModuleType("brotli")
    shim.decompress = decompress            # type: ignore[attr-defined]
    sys.modules["brotli"] = shim


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _static(data: bytes, wght: int):
    """A WOFF2 variable font as a static TrueType instance at weight ``wght``."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(io.BytesIO(data), recalcTimestamp=False)
    font.flavor = None                      # write plain TrueType
    static = instancer.instantiateVariableFont(font, {"wght": wght}, updateFontNames=True)
    buf = io.BytesIO()
    static.save(buf)                        # a clean, fully compiled font to merge into
    buf.seek(0)
    return TTFont(buf, recalcTimestamp=False)   # keep head.modified: the build is reproducible


def _merge(base, extra) -> int:
    """Add the glyphs (outline, advance, cmap) of ``extra`` that ``base`` lacks; returns how many.

    Both are static instances of the same design cut into Fontsource's unicode
    subsets, so a glyph name means the same glyph in both. Every table of
    ``base`` is decompiled first: fontTools reads tables lazily by glyph index,
    and the glyph order changes here. Kerning pairs between the added
    (Latin Extended) glyphs are not carried over; the Latin subset keeps its own.
    """
    for tag in list(base.keys()):
        base[tag]
    order = base.getGlyphOrder()
    have = set(order)
    new = [g for g in extra.getGlyphOrder() if g not in have]
    # composite glyphs need their components: pull those in too
    glyf_x = extra["glyf"]
    pending = list(new)
    while pending:
        g = pending.pop()
        gl = glyf_x[g]
        if gl.isComposite():
            for comp in gl.components:
                if comp.glyphName not in have and comp.glyphName not in new:
                    new.append(comp.glyphName)
                    pending.append(comp.glyphName)
    if not new:
        return 0
    order = order + new
    base.setGlyphOrder(order)
    glyf = base["glyf"]
    for g in new:
        glyf.glyphs[g] = glyf_x[g]
        base["hmtx"].metrics[g] = extra["hmtx"].metrics[g]
    glyf.glyphOrder = order
    base["maxp"].numGlyphs = len(order)
    if "post" in base and hasattr(base["post"], "extraNames"):
        base["post"].extraNames = []
        base["post"].mapping = {}
    for table in base["cmap"].tables:
        if table.isUnicode():
            for sub in extra["cmap"].tables:
                if sub.isUnicode():
                    for cp, g in sub.cmap.items():
                        if g in have or g in new:
                            table.cmap.setdefault(cp, g)
    return len(new)


def _set_names(font, family: str, style: str, weight: int) -> None:
    """One family name for every weight, the style as the subfamily, matching style bits.

    Qt matches faces by family name plus the OS/2 weight class on every font
    back end; separate per-weight families ("... SemiBold") would make
    ``QFont.Weight.DemiBold`` fall back to a synthesised bold of Regular.
    """
    name = font["name"]
    full = family if style == "Regular" else f"{family} {style}"
    ps = family.replace(" ", "") + "-" + style
    for rec_id, text in ((1, family), (2, style), (4, full), (6, ps), (16, family), (17, style)):
        name.setName(text, rec_id, 3, 1, 0x409)
        name.setName(text, rec_id, 1, 0, 0)
    os2 = font["OS/2"]
    os2.usWeightClass = weight
    sel = os2.fsSelection & ~0b1100001          # clear ITALIC, BOLD, REGULAR
    if style == "Regular":
        sel |= 1 << 6
    elif style == "Bold":
        sel |= 1 << 5
    os2.fsSelection = sel
    font["head"].macStyle = 1 if style == "Bold" else 0


def build(next_dir: Path, mono_dir: Path, out: Path = HERE) -> list[tuple[str, str]]:
    """Convert, instance and merge both families; returns (file name, sha256) for inputs and outputs."""
    _install_brotli_shim()
    record: list[tuple[str, str]] = []
    for stem, idx, weights in FAMILIES:
        pkg = (next_dir, mono_dir)[idx]
        slug = "atkinson-hyperlegible-next" if idx == 0 else "atkinson-hyperlegible-mono"
        datas = []
        for sub in SUBSETS:
            src = pkg / "files" / f"{slug}-{sub}-wght-normal.woff2"
            data = src.read_bytes()
            record.append((src.name, _sha(data)))
            datas.append(data)
        for style, wght in weights.items():
            base = _static(datas[0], wght)
            for data in datas[1:]:
                _merge(base, _static(data, wght))
            _set_names(base, FAMILY_NAMES[stem], style, wght)
            target = out / f"{stem}-{style}.ttf"
            base.save(str(target))
            record.append((target.name, _sha(target.read_bytes())))
    return record


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    for name, digest in build(Path(sys.argv[1]), Path(sys.argv[2])):
        print(f"{digest}  {name}")
