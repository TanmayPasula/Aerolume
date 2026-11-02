# Examples

## Example projects

The application's **File ▸ Open example** and the Start page list the example projects in
`src/aerolume/examples/projects/`. They live inside the package so that they are installed
with it; the app copies one to `Documents/Aerolume/Examples` before opening it, so the
shipped copy is never changed.

| project | contents | size |
|---|---|---|
| Sedan in 3D | a generic sedan (`aerolume.geometry3d.sedan()`, wheels spinning) at 30 m/s on a moving road, Draft 3D CFD: surface pressure, the mean flow volume (streamlines, vortex cores, slices) and the smoke snapshot of the last time step. Its volumes are block-averaged 2× along the flow to keep the example small (the result's notes say so) | 8.0 MB |
| Tapered wing | NACA 2412 wing, span 3 m, taper 0.5, 3° dihedral, 2° washout, at 25 m/s: viscous vortex-lattice polar (−4° to 12°), span loading, induced drag | 0.1 MB |
| Wing section NACA 2412 | NACA 2412, chord 1 m: viscous polar at Re 10⁶ (−4° to 14°, 19 points) and a Draft CFD run at 4° (capped Reynolds number) | 4.4 MB |
| Sedan vs fastback | sedan and fastback templates at 30 m/s on a moving road, Draft CFD, ready for the Compare page | 12.9 MB |
| Heated cylinder | forced convection from a 10 mm cylinder 10 K above a 15 °C air stream at Re 40, buoyancy off (`al.heated_cylinder_case`), Draft CFD: the mean Nusselt number with its standard error against Lange, Durst & Breuer (1998), the temperature field and the heat balance | 4.6 MB |
| Heated cavity Ra 1e5 | the differentially heated square cavity at Ra 10⁵, Pr 0.71 (`al.cavity_case`), Draft CFD: the hot wall's mean Nusselt number next to de Vahl Davis (1983), Nu = 4.509, a validation case | 0.4 MB |
| Room with a fan and a heater | a 4 m × 3 m room in 2D with a 1 N/m thrust fan, a 60 W/m floor heater, a supply jet, a vent and two smoke emitters, Draft CFD. An illustration of the placed-physics tools, not a validation: a real room is turbulent, which the Draft grid does not resolve, and in 2D the room's speeds are qualitative (the run says both) | 2.1 MB |

The Start page lists the 3D examples first. Every result in them is real, produced by
`make_example_projects.py` with the defaults and the run function the Run button uses.
Rebuild them after a change to the solver, the planner or the result format (a few
minutes on a GPU):

```
python examples/make_example_projects.py              # all
python examples/make_example_projects.py --only 3d    # the 3D car and the wing
python examples/make_example_projects.py --only heat  # the heated cylinder, the cavity and the room
```

## Example geometry

`geometry/` holds one file per import format, for trying the Import dialog and for the
tests: airfoil coordinates in Selig (`naca2412_selig.dat`) and Lednicer
(`naca23012_lednicer.dat`) order, a car profile as SVG, DXF and a PNG silhouette, and a
two-element wing as CSV points. `geometry/make_examples.py` regenerates them.
