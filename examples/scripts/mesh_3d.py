"""Import a mesh, run a Draft 3D simulation, and look at the flow around the body.

Set MESH to your STL, OBJ or PLY file. Without one, a 0.2 m sphere runs at Re 100,
a validated case with a known answer, so the first run gives a number to check.
"""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()
from pathlib import Path

MESH = Path("my_car.stl")                      # <- your mesh file
SPEED = 30.0                                   # m/s, for your mesh

if MESH.is_file():
    body = project.import_mesh(MESH, units="auto", kind="vehicle")
    flow = {"speed": SPEED}
else:
    # Re 100 is inside the solver's validated range (Re 20 to 300): the standard drag
    # curve gives CD 1.09, and the Draft grid (20 cells across the sphere) reads about 1.16.
    print(f"{MESH} not found: using a 0.2 m sphere at Re 100 instead.")
    body = project.add_sphere(0.2)
    flow = {"re": 100}                         # the speed follows from the diameter

print(project.plan(body, **flow))              # cells, estimated run time, planner warnings
result = project.run_cfd(body, quality="draft", **flow)
print(result)

al.show(result.plot("surface", field="cp"))    # pressure on the surface
al.show(result.plot("streamlines"))            # 3D streamlines through the flow around it
try:
    al.show(result.plot("smoke"))              # the smoke cloud (when the run recorded it)
except al.AerolumeError as exc:
    print(exc)
