"""Pictures of the newest 3D CFD result in this project: streamlines, smoke, vortices."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

results = [r for r in project.results() if r.kind == "cfd3d"]
if not results:
    print("No 3D CFD result in this project yet: run the snippet 'Import a mesh and run 3D CFD' first.")
else:
    result = results[-1]
    print(result)
    for kind in ("streamlines", "smoke", "vortices"):
        try:
            al.show(result.plot(kind))
        except al.AerolumeError as exc:
            print(f"{kind}: {exc}")
    result.export(project.path / "exports" / "surface.vtp")    # for ParaView
