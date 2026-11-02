"""A fan blowing in a closed 4 m x 3 m room (2D): its delivered thrust, and the circulation it drives."""
import aerolume as al

# In the app and with `aerolume-cli script` the open project is bound as `project`;
# run on its own, the script works in a temporary project.
project = globals().get("project") or al.new_project()

case = al.fan_room_case(thrust=1.0)            # one 0.5 m fan a third of the way along, 1 N per metre of span
print(case)
smoke = [{"kind": "rake", "position": (0.4, 0.4), "end": (0.4, 2.6), "count": 8}]    # to see the circulation
result = project.run_case(case, quality="draft", emitters=smoke)
print(result)
# In 2D a closed room's speeds are qualitative and grid-dependent (no vortex stretching; theory_ns2d.md
# section 18.4): the thrust is exact and the flow pattern meaningful, but use dimension=3 for numbers.
print("Note:", result.warnings[0] if result.warnings else case.notes[0])

for fan in result.fans:                        # thrust, volume flow, mean induced velocity, ideal power
    print(", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in fan.items()))
# A consistency check, not a validation: thrust mode imposes the thrust, so this shows the fan delivers what
# was set. Momentum theory does not apply in a closed room (the fan recirculates its own wake).
check = case.check(result)
print(f"Delivered {check['measured']:.4g} N/m against {check['reference']:g} N/m set: "
      f"{check['difference_pct']:+.1f} % ({check['kind']})")
try:
    al.show(result.plot("field", field="speed"))    # with streamlines: the recirculation in the room
except al.AerolumeError as exc:
    print(exc)
