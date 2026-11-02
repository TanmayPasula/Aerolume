# Bundled fonts

Aerolume's interface uses **Atkinson Hyperlegible Next** and its numbers and logs use
**Atkinson Hyperlegible Mono**. Both were designed for the Braille Institute for readers
with low vision: 0 / O, 1 / l / I and 5 / S are drawn so they cannot be mistaken for one
another, which matters in a program whose output is numbers.

| file | face | weight |
|---|---|---|
| `AtkinsonHyperlegibleNext-Regular.ttf` | Atkinson Hyperlegible Next | 400 |
| `AtkinsonHyperlegibleNext-Medium.ttf` | Atkinson Hyperlegible Next | 500 |
| `AtkinsonHyperlegibleNext-SemiBold.ttf` | Atkinson Hyperlegible Next | 600 |
| `AtkinsonHyperlegibleNext-Bold.ttf` | Atkinson Hyperlegible Next | 700 |
| `AtkinsonHyperlegibleMono-Regular.ttf` | Atkinson Hyperlegible Mono | 400 |
| `AtkinsonHyperlegibleMono-SemiBold.ttf` | Atkinson Hyperlegible Mono | 600 |

## Licence

SIL Open Font License 1.1: `OFL-AtkinsonHyperlegibleNext.txt` and
`OFL-AtkinsonHyperlegibleMono.txt` (copied unchanged from the upstream packages).
Copyright 2020-2024 The Atkinson Hyperlegible Next Project Authors and The Atkinson
Hyperlegible Mono Project Authors (https://github.com/googlefonts/atkinson-hyperlegible-next,
https://github.com/googlefonts/atkinson-hyperlegible-next-mono). No Reserved Font Name is
declared. The fonts may be used, bundled and redistributed with the program; they may not
be sold on their own.

## Provenance and why these are TrueType files

The sources are the Fontsource packages `@fontsource-variable/atkinson-hyperlegible-next`
and `@fontsource-variable/atkinson-hyperlegible-mono`, version 5.3.0: variable-weight WOFF2
files cut into a `latin` and a `latin-ext` subset.

Qt cannot use them as published: `QFontDatabase.addApplicationFont` returns -1 for every
WOFF2 file on both the Windows and the offscreen platforms of this PySide6 6.11 build
(measured; Qt reads TrueType, OpenType and WOFF 1 only). `build_fonts.py` therefore

1. decompresses each WOFF2 file and reverses its table transforms (fontTools; the Brotli
   stream is decoded by the `brotli` module or, when that is not installed, by Node.js'
   built-in zlib),
2. pins the `wght` axis at 400 / 500 / 600 / 700 (Next) and 400 / 600 (Mono), because static
   faces match by weight the same way on every Qt font back end,
3. merges the `latin-ext` glyphs into the `latin` face (329 code points in Next, 327 in
   Mono; kerning between two Latin Extended letters is not carried over),
4. gives every weight the same family name and sets the OS/2 weight class and style bits.

The glyph outlines are not changed. The build is reproducible (the font timestamps are
kept), so running it again gives the checksums below.

The subsets contain no Greek letters, superscripts beyond ³ or arrows. Qt draws those
glyph by glyph from the platform UI font (Segoe UI on Windows), which `theme.py` lists
after the bundled face in every font stack.

## Rebuilding

```
python build_fonts.py <node_modules>/@fontsource-variable/atkinson-hyperlegible-next \
                      <node_modules>/@fontsource-variable/atkinson-hyperlegible-mono
```

## SHA-256

```
18b2a1a39a2fa298b0ba5390aca68462669826c90925656f1c1f6796e0e1bbaf  atkinson-hyperlegible-next-latin-wght-normal.woff2 (source)
7dae0c6c66af1aec82e096186aeb1f0e6fa36ab8061ed65422f2b7daf4dd93f3  atkinson-hyperlegible-next-latin-ext-wght-normal.woff2 (source)
2706b1ee4f452e744ea91f7e4908cbde9c5d35521bf5ffffc71a382a2de89613  atkinson-hyperlegible-mono-latin-wght-normal.woff2 (source)
3fa1fa7e990377c87018d62f356cc42b8db39fd88cee5724431e9f0d63b7f32f  atkinson-hyperlegible-mono-latin-ext-wght-normal.woff2 (source)
918ef30d506dec45919d55cb72f21ded76d7eeb732587bea256fab16047be867  AtkinsonHyperlegibleNext-Regular.ttf
f14057177cd79c48488ca65d0c655812fa2d16f4cc835d9145c340e630ac4dab  AtkinsonHyperlegibleNext-Medium.ttf
2fb39577b32d58d36089fb1eb04cda4925f0383b759863ac5f468c81cfbced6e  AtkinsonHyperlegibleNext-SemiBold.ttf
716b93a07bfd09825c79773483b77844c358f2b6f4a051776a931eff42dbc4bb  AtkinsonHyperlegibleNext-Bold.ttf
0bf8f1cdd2c581ebd9b484de50876fe6babd904cee00cde8c67d4a74eb87f195  AtkinsonHyperlegibleMono-Regular.ttf
cc943f8bba282d71f3b8cede6174d41446040cc2b521ab3eab316d808811ff22  AtkinsonHyperlegibleMono-SemiBold.ttf
```
