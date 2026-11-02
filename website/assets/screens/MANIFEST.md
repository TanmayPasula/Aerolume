# Website screenshots

The home page's "The application" section is generated from the table below by
`website/build.py`. It shows only the rows whose `assets/screens/<file>` exists (the
first one at full width, the rest in a grid) and leaves the whole section out while
none do, so the site never ships a mock-up or a wall of placeholders in place of the
real application.

How to take them (retake them after any visible change to the application):
`python website/tools/make_screens.py --work <scratch folder>` builds the project
below with real Draft runs, takes every screenshot offscreen and copies them here and to
`docs/img/ui_*.png`. By hand, the rules are:

* Light theme, default font size, window 1600 × 1000 logical pixels, device pixel
  ratio 1 (the site scales them down). PNG, no window shadow.
* Render with `QWidget.grab().save(path)` on the main window, or a real screen capture;
  never edit pixels or compose several windows into one image.
* Use the stated project state so the captions stay true. Keep the file names: the
  build matches them exactly. Re-run `python website/build.py` afterwards.

| file | page | state | caption |
|---|---|---|---|
| geometry.png | Geometry | NACA 2412 template, chord 1 m, inspector showing Source and References | Geometry: a NACA 2412 section with its reference length, moment point and diagnostics. |
| setup.png | Setup | Fastback template, 30 m/s, moving ground on, Estimate table visible | Setup: the planner's estimate (Re real and simulated, grid, time step, run time) next to the property grid. |
| simulate.png | Simulate | NACA 2412 at 8 deg running on CUDA, vorticity field, Cd/Cl monitors | Simulate: live vorticity with the colour bar, run progress and the force monitors. |
| results.png | Results | Finished NACA 2412 run, Summary tab | Results: coefficient table with mean, standard error and range, and the run's validity line. |
| compare.png | Compare | Three fastback variants (rear slope 12, 17, 22 deg) | Compare: drag and lift of three rear-slope variants with deltas against the baseline. |
| report.png | Report | Report form with the NACA 2412 run selected and one generated PDF | Report: section selection and the list of generated PDF and HTML reports. |
| results3d.png | Results | Bundled example "Sedan in 3D" (Draft, 30 m/s, moving road), Fields tab of the 3D result | Results: in 3D, the stored smoke and the mean-flow streamlines around a car on a moving road, from a Draft run. |
| heat.png | Results | Bundled example "Heated cylinder" (Re 40, 10 K above the stream, Draft), Heat transfer tab | Results: heat transfer of the cylinder, its Nusselt number with its standard error, the heat balance and the temperature field. |
| physics.png | Setup | Bundled example "Room with a fan and a heater", Setup page with the placed items in the preview | Setup: placed physics, a fan, a heater, a supply jet, a vent and two smoke emitters in a closed room. |
| explore.png | Simulate | The same room in explore mode, Live workspace, a wall drawn with the brush into the running flow | Simulate: in explore mode, a wall drawn into the running flow of the room, beside the solver's live diagnostics. |
| script.png | Script | The same room, Scripting workspace, a short script reading the fan report and plotting the temperature | Script: the scripting API reads the open project's results and plots them inline. |
