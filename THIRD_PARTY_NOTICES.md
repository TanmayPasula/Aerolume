# Third-party notices

## mfoil

`third_party/mfoil/mfoil.py` is mfoil v2023-06-28 by Krzysztof J. Fidkowski
(University of Michigan), an XFOIL-class viscous–inviscid airfoil solver,
distributed under the MIT licence (full text in `third_party/mfoil/LICENSE`
and in the file header). Obtained from https://github.com/S-Lykles/mfoil
(`mfoil_original.py`), a verbatim copy of the original release.
Reference: K. J. Fidkowski, "A coupled inviscid–viscous airfoil analysis solver,
revisited", AIAA Journal, 2022.

The copy the package imports, with Aerolume's robustness patches, is
`src/aerolume/solvers/airfoil/_mfoil.py`; it carries the same copyright notice and
MIT licence text in its header, and its local modifications are listed there.

## Atkinson Hyperlegible Next and Atkinson Hyperlegible Mono

The interface fonts in `src/aerolume/app/resources/fonts/` are Atkinson Hyperlegible Next
(regular, medium, semibold, bold) and Atkinson Hyperlegible Mono (regular, semibold),
Copyright 2020-2024 The Atkinson Hyperlegible Next Project Authors and The Atkinson
Hyperlegible Mono Project Authors (https://github.com/googlefonts/atkinson-hyperlegible-next,
https://github.com/googlefonts/atkinson-hyperlegible-next-mono), licensed under the SIL Open
Font License 1.1 (full texts next to the fonts: `OFL-AtkinsonHyperlegibleNext.txt`,
`OFL-AtkinsonHyperlegibleMono.txt`). No Reserved Font Name is declared. They were converted
from the Fontsource 5.3.0 WOFF2 packages to static TrueType faces with unchanged glyph
outlines by `build_fonts.py` in the same folder; its README records the steps and checksums.
The application registers them at start-up (`aerolume.app.theme.ensure_fonts`, through
`QFontDatabase`) and falls back to Segoe UI / Consolas, glyph by glyph or entirely, when a
face is missing. The website sets the same two families from the original WOFF2 files, with
their licences beside them in `website/assets/fonts/`.

## Packages of the Windows build

The installer and the portable folder also contain Python and the packages Aerolume runs on
(PyTorch with the NVIDIA CUDA libraries, NumPy, SciPy, PySide6 / Qt, VTK, pyqtgraph,
Matplotlib and their dependencies). `packaging/build.py` (`buildlib.collect_licenses`) fills
the build's `licenses` folder, and `licenses/README.txt` lists every component with its
version and licence:

* **Python packages**: the licence files each distribution ships, in `licenses/<name>/`.
  Nested files keep their folders, so the licences of the open-source projects compiled into
  PyTorch are in `licenses/torch/third_party/` (95 files for torch 2.14.1).
* **Qt for Python (PySide6, shiboken6)**: used under the GNU LGPL v3. The wheels carry only
  `LicenseRef-Qt-Commercial.txt`, so the build adds the LGPL-3.0 and GPL-3.0 texts from
  `packaging/licenses/`. Each folder also gets a README that says how to replace the Qt
  libraries in `_internal/PySide6` and where The Qt Company publishes the source of the
  bundled versions.
* **VTK's vendored libraries**: VTK's wheel ships only VTK's own licence, but `_internal/vtk.libs`
  also holds VTK's builds of 20 other libraries: zlib, libpng, libjpeg-turbo, libtiff, FreeType,
  expat, liblzma (XZ Utils), lz4, pugixml, JsonCpp, KISS FFT, Verdict, scnlib (with NanoRange),
  MetaIO, DICOMParser, KWSys, token, Loguru, glad and {fmt}. `licenses/vtk/third_party/` holds
  each one's licence text, copied verbatim from the VTK 9.7.1 source tree, and a README that
  names the DLL, the licence and the upstream version of each. Portions of this software are
  copyright © The FreeType Project (www.freetype.org). All rights reserved. This software is
  based in part on the work of the Independent JPEG Group.
* **Third-party code inside Qt**: Qt's DLLs and plugins contain HarfBuzz, FreeType, libpng,
  libjpeg-turbo, libtiff, libwebp, md4c, PCRE2, zlib, double-conversion, TinyCBOR, the Unicode
  data and more; `opengl32sw.dll` is Mesa's llvmpipe software renderer (Mesa 11.2.2 with
  LLVM 3.6.2). `licenses/PySide6_Essentials/third_party/README.txt` lists each with its version,
  use, licence and copyright holders, as Qt's own attribution data (`qt_attribution.json` of
  qtbase, qtimageformats and qtsvg 6.11.2) records them, and the folder holds the licence texts
  from Qt's sources and Mesa's and LLVM's licences. The entries for platforms and Qt modules
  that are not bundled (Wayland, X11, macOS, Android, Qt SQL, Qt D-Bus) are named, not copied.
* **CPython**: `licenses/python/` holds the `LICENSE.txt` and the documentation's
  `license.html` of the Python installation the build was made from. They cover the PSF
  licence and the libraries the interpreter contains: OpenSSL (Apache-2.0), libffi, expat,
  bzip2, zlib and libmpdec. SQLite and liblzma are in the public domain.
* **NVIDIA CUDA and cuDNN runtime libraries** (cuBLAS, cuDNN 9, cuFFT, cuRAND, cuSOLVER,
  cuSPARSE, NVRTC, nvJitLink, CUPTI, NVTX and others in `_internal/torch/lib`): proprietary
  NVIDIA software, redistributed under the NVIDIA CUDA Toolkit EULA and the cuDNN Software
  License Agreement. `licenses/nvidia-cuda/NOTICE.txt` lists each file with the version and
  copyright recorded in it and gives the agreements' addresses. Neither agreement requires
  its text to be shipped, and the texts are not included.
* **Intel OpenMP runtime** (`libiomp5md.dll`, `libiompstubs5md.dll` in `_internal/torch/lib`):
  proprietary Intel software that comes with PyTorch. `licenses/intel-openmp/NOTICE.txt`
  names Intel's End User License Agreement for Developer Tools and its address. The
  agreement's text is not included.

`packaging/licenses/README.md` records where each vendored text comes from.
