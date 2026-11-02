"""A PDF report of the three newest results in this project."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

results = project.results()[-3:]
if not results:
    print("No results yet: run a simulation first.")
else:
    path = project.report(results, title="Script report")
    print(f"Report written to {path}")
