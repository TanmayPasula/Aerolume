"""Sedan against fastback at 30 m/s on a moving road (2D CFD, Draft quality)."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

sedan = project.add_template("sedan")
fastback = project.add_template("fastback")
runs = [project.run_cfd(design, speed=30) for design in (sedan, fastback)]

comparison = project.compare(runs)             # the Compare page's table, with significance
print(comparison)
al.show(comparison.plot(["cd", "cl"]))
