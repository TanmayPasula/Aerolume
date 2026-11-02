"""2D CFD of a NACA 0012 at four angles (Draft quality): lift and drag with error bars."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

airfoil = project.add_naca("0012")
# Airfoils start, as in the app, in the capped-Reynolds mode: no affordable 2D grid resolves the
# boundary layer at Re 2e6, so the run is made at a low Reynolds number and the trends are what to
# read (re_sim against re_real below). project.run_polar(airfoil, re=2e6) gives real-Re numbers.
sweep = project.sweep(airfoil, alpha=[0, 2, 4, 6], speed=30, quality="draft")
print(sweep)

al.show(sweep.plot("alpha", ["cl", "cd"]))     # error bars: +-1 standard error of the mean
best = sweep.best("ld")
print(f"Best L/D {best.ld:.2f} at alpha {best.alpha:g} deg "
      f"(Re {best.re_sim:.3g} simulated, {best.re_real:.3g} real)")
