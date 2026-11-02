"""Viscous-inviscid polars of a NACA 2412 at three Reynolds numbers (about a minute)."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

airfoil = project.add_naca("2412", chord=1.0)
polars = project.sweep(airfoil, analysis="polar", re=[2e5, 5e5, 1e6], alpha=(-4, 14, 1))
print(polars)                                  # Cl,max and best L/D per Reynolds number

best = polars.best("ld_max")
print(f"Best L/D {best.metrics['ld_max']:.1f} at Re {best.re_sim:.3g}, alpha {best.metrics['alpha_ld_max']:g} deg")
al.show(polars.plot("alpha", "cl"))            # lift curves, one per Reynolds number
al.show(polars.plot("cd", "cl"))               # drag polars
