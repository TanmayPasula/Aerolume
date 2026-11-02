"""Natural convection in a heated square cavity at Ra 1e4, checked against de Vahl Davis (1983)."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

case = al.cavity_case(ra=1e4)                  # air (Pr 0.71) in a 10 cm box: hot left wall, cold right wall
print(case)
print(*case.notes, sep="\n")                   # the wall temperatures that give Ra 1e4, the free-fall speed
print(project.plan(case))                      # grid, time step, Ra, Pr, Gr and the estimated time
result = project.run_case(case, quality="draft")
print(result)

check = case.check(result)                     # mean Nusselt number of the hot wall against the paper
print(f"Nu {check['measured']:.4g} +- {check['stderr']:.2g} (standard error), de Vahl Davis "
      f"{check['reference']:g}: {check['difference_pct']:+.1f} %")
# The standard error is the averaging uncertainty only. The grid error needs three grids:
# project.grid_study(case, quantity="nusselt") gives the observed order and the GCI.
try:
    al.show(result.plot("field", field="temperature"))
except al.AerolumeError as exc:
    print(exc)
