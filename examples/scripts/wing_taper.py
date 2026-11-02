"""Wings of equal span and area, taper 0.3 to 1.0 (vortex lattice, seconds)."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

SPAN, AREA = 2.0, 0.48                         # m, m2 (aspect ratio 8.3)


def wing(taper):
    root = 2 * AREA / (SPAN * (1 + taper))
    return project.add_wing(span=SPAN, root_chord=root, tip_chord=root * taper, airfoil="2412")


study = project.sweep(wing, taper=[0.3, 0.45, 0.6, 0.8, 1.0], speed=20, alpha=(-2, 10, 1), viscous=False)
print(study)

al.show(study.plot("taper", ["oswald", "cl_alpha_per_deg"]))
best = study.best("oswald")
print(f"Highest span efficiency: {best.design.name} (e = {best.metrics['oswald']:.3f})")
