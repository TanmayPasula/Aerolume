# Licence texts and notices for the Windows build

`buildlib.collect_licenses` copies the licence files that each bundled Python distribution
ships into the app's `licenses` folder. It then adds the files in this folder for the parts
of the bundle whose distributions ship no usable licence text:

| File | Goes to | What it is |
|---|---|---|
| `LGPL-3.0.txt` | `licenses/PySide6_Essentials/`, `licenses/shiboken6/` | GNU Lesser General Public License v3, verbatim. Qt for Python's wheels carry only `LicenseRef-Qt-Commercial.txt`, while LGPL-3.0 section 4 requires this text with the replaceable Qt libraries. |
| `GPL-3.0.txt` | the same two folders | GNU General Public License v3, verbatim. The LGPL-3.0 incorporates it, and section 4(b) requires it too. |
| `qt-for-python.txt` | `licenses/PySide6_Essentials/README.txt` | Notice written for Aerolume: the licence chosen, how to replace the libraries and where the source is. The build fills in the bundled versions. |
| `python-runtime.txt` | `licenses/python/README.txt` | Notice written for Aerolume, next to the `LICENSE.txt` and `license.html` that the build copies from the Python installation it runs on (`sys.base_prefix`). |
| `nvidia-cuda.txt` | `licenses/nvidia-cuda/NOTICE.txt` | Notice written for Aerolume for the NVIDIA libraries that PyTorch ships in `torch/lib`. The build adds each file with the version and copyright recorded in it. |
| `intel-openmp.txt` | `licenses/intel-openmp/NOTICE.txt` | The same for Intel's OpenMP runtime (`libiomp5md.dll`, `libiompstubs5md.dll`). |
| `vtk/` | `licenses/vtk/third_party/` | The licence texts of the libraries VTK builds into its own DLLs in `vtk.libs` (`vtkzlib-9.7.1.dll`, `vtkfreetype-...`, ...), one folder per DLL, and `components.json` (DLL, name, SPDX licence, upstream path and tag). The build copies the folders of the DLLs it bundles and writes a README from `components.json`. |
| `qt/` | `licenses/PySide6_Essentials/third_party/` | `components.json`: the entries of Qt's own `qt_attribution.json` files (qtbase, qtimageformats, qtsvg) that apply to the bundled Windows modules and plugins, with the licence files they name kept at their paths in Qt's sources (`qt/qtbase/src/3rdparty/harfbuzz-ng/COPYING`, `qt/qtbase/LICENSES/MIT.txt`, ...); the entries that do not apply are named under `not_applicable`. `qt/mesa-llvmpipe/` holds Mesa's and LLVM's licences for `opengl32sw.dll`. |
| `UPSTREAM.sha256` | (not shipped) | SHA-256 of every file under `vtk/` and `qt/`. |

The GPL and LGPL texts are the Free Software Foundation's files
(https://www.gnu.org/licenses/gpl-3.0.txt and https://www.gnu.org/licenses/lgpl-3.0.txt). They
were copied byte for byte from the copies that GNU software ships: GCC's `COPYING3` and GNU
MPC's `COPYING.LESSER`. `tests/test_packaging.py` pins their SHA-256 values, so an accidental
edit fails the tests:

| File | SHA-256 |
|---|---|
| `GPL-3.0.txt` | `8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903` |
| `LGPL-3.0.txt` | `da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768` |

The files under `vtk/` and `qt/` were copied byte for byte from the upstream sources at the
bundled versions: VTK `v9.7.1` (https://github.com/Kitware/VTK, `ThirdParty/<name>/vtk<name>/`
and `Utilities/`), Qt `v6.11.2` (https://github.com/qt/qtbase, qtimageformats and qtsvg), Mesa
`mesa-11.2.2` (`docs/license.html`, https://gitlab.freedesktop.org/mesa/mesa) and LLVM
`llvmorg-3.6.2` (`llvm/LICENSE.TXT`, https://github.com/llvm/llvm-project); the Mesa and LLVM
versions are the ones `opengl32sw.dll` names in its strings. `UPSTREAM.sha256` lists their
SHA-256 values and `tests/test_packaging.py` pins that list's own SHA-256, so an edit to a text
fails the tests until both are updated on purpose. When VTK or Qt is upgraded, fetch the same
files at the new tags, regenerate `components.json` and `UPSTREAM.sha256`, and update the pin:
`buildlib.collect_vtk_third_party` refuses to build when `vtk.libs` holds a third-party DLL that
`vtk/components.json` does not list.

NVIDIA's and Intel's agreements are not copied here. NVIDIA's distribution requirements do
not ask for the agreement text to be shipped. Intel's agreement is published as a PDF on its
website. Each notice names the agreement and gives its address. To ship a verbatim copy of
either agreement as well, add it to this folder and to `collect_licenses`.

Every file in this folder is a build input (`buildlib.source_files`), so changing one makes
the built app stale until the next full build.
