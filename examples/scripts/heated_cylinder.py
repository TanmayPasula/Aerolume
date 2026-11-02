"""Forced convection from a heated cylinder at Re 40, against the Lange et al. (1998) correlation."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

case = al.heated_cylinder_case(re=40)          # a 10 mm cylinder held 10 K above the stream, buoyancy off
print(case)
result = project.run_case(case, quality="draft")
print(result)

check = case.check(result)
print(f"Nu {check['measured']:.4g} +- {check['stderr']:.2g}, Lange et al. {check['reference']:.4g}: "
      f"{check['difference_pct']:+.1f} %")
try:
    al.show(result.plot("field", field="temperature"))
except al.AerolumeError as exc:
    print(exc)

GRID_STUDY = False                             # True: three grids 1.5x apart, Richardson extrapolation, GCI
if GRID_STUDY:
    study = project.grid_study(case, quality="draft", quantity="nusselt")
    print(study)                               # observed order, extrapolated Nu and the GCI band
    al.show(study.plot("nusselt"))
