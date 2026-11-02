"""3D smoke around a sphere at Re 300, where the wake sheds vortices: two coloured smoke rakes."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

sphere = project.add_sphere(0.2)               # centred at the origin; the flow is along +x
# two rakes 1.5 diameters upstream: one across the height (white), one across the width (amber).
# Each colour is its own smoke channel (up to four).
rakes = [
    {"kind": "rake", "position": (-0.3, 0.0, -0.08), "end": (-0.3, 0.0, 0.08), "count": 7, "color": "#f5f5f5"},
    {"kind": "rake", "position": (-0.3, -0.08, 0.0), "end": (-0.3, 0.08, 0.0), "count": 7, "color": "#f0a040"},
]
if "emitters" not in al.supported_physics(3):
    print("This build's 3D solver has no placed emitters yet: the run uses its default smoke rake.")
    rakes = None
result = project.run_cfd(sphere, re=300, quality="draft", emitters=rakes)
# Draft is a grid for pictures: it read CD 0.75 here, against 0.656 computed by Johnson & Patel
# (1999) at Re 300; quality="standard" and a longer duration are for numbers.
print(result)

for kind in ("smoke", "streamlines", "vortices"):
    try:
        al.show(result.plot(kind))
    except al.AerolumeError as exc:
        print(f"{kind}: {exc}")
try:
    print(result.export(project.path / "exports" / "sphere_smoke.vti", what="smoke"))   # for ParaView
except al.AerolumeError as exc:
    print(exc)
