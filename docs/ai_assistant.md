# Experiment assistant

The **Assistant** panel plans, runs and analyses aerodynamic experiments from a goal written in
plain language:

* "Find the angle of attack with the best L/D of NACA 2412 at 20 m/s using the polar analysis."
* "Compare the drag of the 2D Ahmed body at slant angles of 20°, 25° and 30° at 40 m/s (draft quality)."
* "How does the aspect ratio (6, 8, 10) of a 1.2 m² rectangular wing change its induced drag at 25 m/s?"
* "Estimate the cost of a yaw sweep of the selected 3D design before running anything."
* "Run the heated cavity at Ra 10⁴ and tell me how close its Nusselt number is to the benchmark."

It works only through the same tools as the rest of Aerolume - the scripting API
([scripting.md](scripting.md)), which drives the polar, wing and CFD solvers - so every design it
creates and every run it starts is an ordinary design and simulation of your project: they appear
in the Project tree, run in the same queue as runs you start by hand, open on the Results page and
go into reports. Each experiment is logged in the project (see [Experiment logs](#experiment-logs)).

Two kinds of model can drive it:

| | Claude (Anthropic API) | Local model (Ollama) |
|---|---|---|
| Runs | on Anthropic's servers | on this computer (GPU or CPU) |
| Needs | `pip install aerolume[ai]`, an API key or an `ant auth login` profile | [Ollama](https://ollama.com) and a model with tool calling, e.g. `qwen3:8b` |
| Data | goals, design and simulation names, tool results go to Anthropic | nothing leaves the computer |
| Planning | multi-step studies, sweeps, comparisons, caveats | direct questions (one polar, one wing, one CFD run) |
| Cost | per token (prompt caching keeps repeated context cheap) | free; about 6 GB of GPU memory while it answers |

With the provider set to **Automatic** (the default) Aerolume uses Claude when credentials are
available and the local model otherwise. A third way needs no model in Aerolume at all: Claude
Desktop or Claude Code drive the same tools through the `aerolume-mcp` server
([mcp.md](mcp.md)).

## Setting up

### Claude

1. Install the optional dependency: `pip install aerolume[ai]` (the official `anthropic` SDK).
2. Give it credentials, in order of preference:
   * the `ANTHROPIC_API_KEY` environment variable, or an `ant auth login` profile (the SDK's own
     credential chain; Aerolume stores nothing), or
   * **Edit ▸ Preferences ▸ Assistant ▸ API key**. The key is saved only in your user configuration
     folder (`%APPDATA%\Aerolume\assistant_key` on Windows, `~/.config/aerolume/assistant_key`
     elsewhere, or under `AEROLUME_CONFIG_DIR`), in a file of its own with owner-only permissions
     on Linux and macOS. It is never written into a project, an experiment log or the settings
     file, never logged, and the dialog only ever shows it masked (`sk-ant-…a1b2`). **Remove saved
     key** deletes it.
3. Pick the model in the same tab. The default is `claude-opus-5-5` at effort `high`; lower effort
   answers faster and costs less, higher effort plans more thoroughly.

Requests use adaptive thinking (the panel shows the model's reasoning summary under
**Reasoning**), streaming, and Anthropic's server-side fallback: if a safety classifier declines a
request, the API re-runs it on the recommended fallback model and the panel says so. A request that
is declined all the same ends with a plain message, and nothing it asked for is run.

### Local model (Ollama)

1. Install Ollama from [ollama.com](https://ollama.com) and start it.
2. Pull a model that supports tool calling: `ollama pull qwen3:8b` (5.2 GB on disk; 7.8 GB in
   memory with the 16,384-token context Aerolume uses, 6.3 GB of it on an 8 GB GPU).
3. **Preferences ▸ Assistant**: choose *Local model (Ollama)*. The tab lists the installed models
   that can call tools (code-completion models without tool calling are left out); the address is
   `http://localhost:11434` unless Ollama runs elsewhere.

Options:

* **Reasoning** lets a reasoning model (qwen3) think before each answer: better plans, several
  times slower. Off by default.
* **Context** is the model's context window, 16,384 tokens by default. An experiment with one run
  filled 8,192 (7,309 tokens after the plan, the tool schemas, the project overview, the run and two
  corrections), after which Ollama drops the oldest messages -- the plan and results the answer is
  checked against. 16,384 puts qwen3:8b at 7.8 GB, 80 % of it on an 8 GB GPU; generation is slower
  once the window fills (measured 14.7 tokens/s after a 12k-token prompt, against 42.7 after a 4k
  one at 8,192, RTX 4060 Ti). 8,192 keeps the model entirely on the GPU if you prefer speed. A
  settings file saved before this default with the old 8,192 is moved to 16,384 once; a value you
  choose afterwards stays. The panel warns once per experiment when the window is nearly full.

A model loaded by Ollama keeps its GPU memory for some minutes. Before the assistant starts a GPU
CFD run it asks Ollama to unload the model so the solver gets the memory; the next answer reloads
it in a few seconds.

## Using the panel

Open it with the **Assistant** button in the toolbar, **View ▸ Assistant panel** (Alt+3) or
**Tools ▸ Experiment assistant** (Ctrl+Shift+A). A project must be open: experiments are recorded
in it.

* Type the goal and press **Run** (Ctrl+Enter). The design and simulation selected in the window
  are passed along, so "this airfoil" and "the selected design" mean what is on screen.
* The transcript shows your goal, the assistant's text, and one row per tool call with its status
  (running with progress, done, failed, declined, stopped) and a one-line summary. The arrow on a
  row shows its exact input and outcome. The plan appears under the *Record plan* row; figures
  appear inline and open full size on a click.
* Runs started by the assistant show in the Project tree and the status bar like any other; the
  Simulate page shows them live.
* **Stop** ends the request at once: the model's answer, the running simulation (it keeps its
  partial result, marked stopped) and any pending approval.
* A follow-up ("and at 30 m/s?") continues the same experiment and log. **New experiment** starts a
  new conversation and a new log.
* **Open log** at the end of a request shows the experiment log.
* Under the answer, **Checked statements** lists every sentence with a number or a claimed action
  and what the claims check found (see [How it stays honest](#how-it-stays-honest)). It is folded
  when everything was verified and open when something was not.

### Before anything is sent

With Claude, or an Ollama server on another computer, the first request of an experiment shows a
card first: *Send this experiment to Anthropic (Claude API)?*, listing what is sent (the goal, the
design and simulation names, the tools' results) and what is not (meshes, flow fields, figures,
project files). Nothing leaves the computer until you press **Send**; **Do not ask again for this
destination** remembers the answer in Preferences. A local Ollama on this computer never asks,
because nothing leaves it.

### Suggestions

When a run should be set up differently (a finer grid, a longer duration after a run that did not
converge) the assistant proposes it with `suggest_settings` instead of changing anything. The
panel shows a card listing every change, old value → new value. Nothing changes until you click
**Apply**:

* each value was checked by the tool against the setting's range, and is checked again by the
  scripting API when you apply it (an unknown setting or an out-of-range value is shown struck out
  with the reason and is never applied);
* a simulation that already has a result is copied, so the result never sits under settings it
  was not computed with;
* **Undo** takes the whole suggestion back in one step (the copy is removed, or the settings are
  restored); **Dismiss** leaves everything as it is.

The model is told that a suggestion is not a change, and an answer that says it "changed" or "set"
a setting is sent back (see *Claims must match what was done*).

### Approvals

Some actions wait for your answer in a framed card with **Run** / **Delete** and **Decline**:

* runs whose **estimated** time is above the threshold in Preferences (default 120 s; 0 asks for
  every run) - the card shows the estimate and what will run, for example "15 m/s · α −4° to 14°,
  step 1° (19 angles) · VLM, viscous strips" for a wing, or the quality, cells, Reynolds numbers
  and device for CFD;
* runs whose cost **cannot be estimated** ("cost unknown"): an estimate that fails never lets a run
  start unasked. A call whose own set-up is wrong is not put to you at all: the planner's reason
  (for example "has heated bodies, but heat transfer is off") goes back to the model, which fixes
  its call, and nothing is run or recorded;
* **all Python code** (`run_python`): the card shows the exact script, wrapped so no line runs past
  its edge; a run of spaces inside a line is shown as «160 spaces» and an invisible character (a
  bidirectional override, a zero-width mark) as «U+202E», with a warning above the script. An
  approved script runs with your rights, so read it to the end. If it changes the approval rules
  themselves, they are put back at once and the call is reported as an error;
* **every deletion** of a design or simulation.

A declined action is reported to the model as declined; it is told not to ask again and to propose
a cheaper alternative or stop. Reading the project, creating designs (always as new designs: a
design you made is never modified in place), estimating costs, figures and logs never ask.

### What it can do (its tools)

| Tool | What it does |
|---|---|
| `project_overview`, `describe_design` | read the project: designs, simulations, references, wing planform, template parameters |
| `create_naca_airfoil`, `create_template`, `create_wing`, `derive_design` | add designs, or a variant of one (template parameters such as the Ahmed slant angle; wing span, taper, sweep, twist; scale or rotation of a 3D mesh) |
| `describe_simulation` | one simulation's settings, its placed physics (heat transfer, walls, fans, jets, heaters, emitters, domain), status and whether it has a result |
| `estimate_cost` | the planner's estimate (time, grid, simulated and real Reynolds number) without running |
| `run_polar`, `run_wing`, `run_cfd`, `run_sweep` | the airfoil polar, the finite-wing analysis, one CFD run, or one analysis repeated over α, speed or yaw; `run_cfd` takes a `physics` object (heat transfer and walls, heaters, fans, jets and vents, smoke emitters, a closed domain) |
| `create_case`, `run_simulation` | a published reference case ready to run (the de Vahl Davis heated cavity, the Lange et al. heated cylinder, a fan in a room) with its reference value; run a simulation that is already set up (by you on the Setup page, or by `create_case`). The fan room is a consistency check only: thrust mode imposes the thrust, and momentum theory does not apply in a closed room |
| `compare_reference` | each simulation against the published reference its set-up reproduces: measured value with its standard error, the reference, the difference in % and the source; a run whose settings left the published conditions (buoyancy on for the forced-convection cylinder) says why instead. Every CFD result read with `get_results` carries the same comparison (`reference_check`) and the models it ran with (`physics_models`: heat transfer, buoyancy, turbulence, domain) |
| `run_grid_study` | re-run a CFD simulation on three refined grids: observed order, Richardson-extrapolated value and the GCI band (Celik et al. 2008), with the averaging noise taken into account; asks first like any long run, the card listing the three grids |
| `suggest_settings` | propose setting changes as a card you apply or dismiss (it changes nothing itself) |
| `get_results`, `compare_results` | stored results with their statistics; comparison against a baseline with significance |
| `make_plot` | a PNG figure (polar, convergence history, Cp, 2D field, 3D surface / streamlines / smoke / vortices / slice, span loading, sweep, comparison) saved with the experiment and shown in the panel |
| `write_report` | the project's PDF or HTML report of chosen simulations |
| `run_python` | a short script against the scripting API (always asks first) |
| `delete_item` | delete a design or simulation (always asks first) |
| `record_plan`, `record_conclusion` | write the plan and the findings into the experiment log; the plan's expected time is the planner's estimate of the runs the experiment set up (`estimate_cost`, `create_case`), never the model's guess (a live plan said "5 min 00 s" for an 11 s run), and with no estimate yet the plan shows none |

Local models get a focused subset (no report, Python, deletion, grid-study or template-gallery
tools, and `run_cfd` without its large physics object): an 8-billion-parameter model chooses
better among fewer tools, and its context must also hold the results. `create_case` (with a
five-field schema) and `compare_reference` are in the subset: without `create_case`, qwen3:8b
answered that "the tools cannot run a simulation for a differentially heated square cavity",
which was true of its tool list while its prompt told it to use `create_case`.

A run that asks for physics this build's solver does not implement is refused before anything is
recorded, naming what is missing, rather than run without it (see `al.supported_physics` in
[scripting.md](scripting.md)).

## How it stays honest

The assistant is instructed to work like a careful experimenter: state the objective, variables,
controls and expected cost before running; prefer the cheapest analysis that answers the question
(a polar answers an airfoil question in seconds at the real Reynolds number; CFD takes minutes);
and report results with their uncertainty and limits. Aerolume enforces what it can rather than
trusting the instructions:

* **Every final answer is checked, statement by statement** (`aerolume.ai.claims`). The answer is
  cut into sentences, list items and table rows; each one with a number or a claimed action gets a
  verdict: **Verified** (the action matches a successful tool call of that kind, on the design it
  names; each number matches a tool result, a tool input or your goal at the precision it is
  written with, and, when the sentence names a run by its speed or design, a value of *that* run),
  **Derived** (not in any result, but a difference, a change in % or a ratio of two result values:
  the formula is shown), **Not supported** (matches nothing the experiment produced) or
  **Contradicted** (an action no tool call performed, or a number that belongs to another run, with
  the value it should have been). Unsupported and contradicted statements are sent back to the
  model once, listed with those values; what remains afterwards is shown flagged, and the
  experiment ends **not verified** (warning glyph) instead of "Done". The panel's *Checked
  statements* and the log's *Verification* table list every verdict with its evidence. The checker
  came from two live failures of qwen3:8b: "The simulation … has been deleted" with no tool call at
  all, and "the stored 25 m/s polar's best L/D of 26.47" when that run's value was 29.52 (26.47 was
  the other run's, so a plain "does this number appear somewhere" check passed it).
  Live runs with qwen3:8b found more, and each is now checked: a number is matched at the
  precision it is written with (not within 0.5 %, which let chance values pass); "best L/D of
  26.47" is a claim even without bold markers; a sentence about a run the experiment never made
  ("the 25 m/s run reached 31.2") is not supported; "L/D is validated within 2-2.5 %" is
  contradicted when the result lists L/D under `not_validated`; a number the model wrote into its
  own plan counts only in a sentence about the plan; and an uncertainty ("± 1.5 %") must come from
  a result (a standard error, a GCI, half the angle step), never from arithmetic on two results
  and never from another number with the same digits (a live "1.18% ± 0.05%" had matched a run's
  speed of 0.05 m/s; it is now unsupported). A number written right next to a speed belongs to the
  run at that speed even when the sentence names two runs: "the best L/D rises from 29.52 at
  15 m/s to 26.47 at 25 m/s" (the two values swapped) is contradicted, with both correct values,
  while a range ("between 15 and 25 m/s it rises from 26.47 to 29.52") ties no number to either end.
* **Numbers come from tool results.** Every result carries its statistics: CFD means with their
  standard error (and a flag when that error is only a rough estimate), convergence, the simulated
  and the real Reynolds number, the solver's warnings, and validity sentences from the validation
  record - for example, an airfoil simulated in 2D CFD at a capped Reynolds number describes
  low-Re flow; absolute 3D vehicle drag is not validated; Cl,max is only a lower bound when lift was
  still rising. Polars and wing analyses are deterministic and have no statistical error bar; their
  resolution is the angle step.
* **Confidence facts are data, not prose.** A polar or wing result also carries
  `optimum_resolution_deg` (half the angle step: a best-L/D angle from a 1° sweep is known to
  ±0.5°), `validated_quantities` and `not_validated`. For a wing, the lift slope, span efficiency
  and induced drag are validated; profile drag, and with it CD, L/D and CL,max, comes from 2D
  section polars and is not. A small model does not derive this from the validity prose: asked how
  confident a best-L/D answer was, qwen3:8b called the result "validated within a few percent"
  until the result said what the validation covers. A wing result whose strips went beyond their
  section cl_max carries those angles too.
* **"How confident is it?" gets an uncertainty figure.** When the goal asks how far to trust the
  answer and the answer gives no ±, standard error or angle resolution, the model is asked once to
  add it.
* **A standard error is not the accuracy.** CFD values reach the model rounded to their standard
  error by the Results tables' rule (the Particle Data Group's: an error whose three leading
  digits are 100-354 keeps two significant figures, 355-949 one, 950-999 rounds up to two;
  the value goes to the same decimal place), so the model is sent Cd 0.514 ± 0.005, not
  0.51377 ± 0.004553, and Nu 3.3408 ± 0.0010, not 3.34079 ± 0.00095, with
  `stderr_meaning`: the averaging uncertainty of that run only, modelling error not included. An
  answer that says "confident within ±" a standard error about a quantity the result marks not
  validated is sent back once.
* **A steady value has no ±.** When a standard error is below 10⁻⁵ of its value (a settled run:
  a converged cavity's Nu had 4.4 × 10⁻¹⁶, float round-off), the model gets the value to four
  significant figures with `steady: true` and no error, the run card reads "Nu 2.251 (steady)" as
  the Results page does, and the result says that the value's error is the discretisation error,
  which `run_grid_study` (the GCI) gives. An answer that writes a ± for a steady quantity, or a
  ± of zero, is contradicted by the claims check and sent back; "steady" together with the grid
  study counts as the answer's uncertainty statement. (Live qwen3:8b had answered "2.25 ± 0.00
  (4.4e-16)".)
* **The answer's ± is rounded the same way.** Whatever the model writes, every "value ± error"
  in the answer is quoted by that rule as it streams and in the log (`plaintext.round_pm`):
  "3.3407 ± 0.00095" becomes "3.3407 ± 0.0010", the figure the Results table beside it shows. An
  error is only ever shortened, never padded ("3.34 ± 0.01" stays), and numbers in scientific
  notation are left as written.
* **"Validated" needs a reference check inside its band.** A statement that calls a result
  validated or verified passes the claims check only when the experiment compared it with a
  published reference (`compare_reference`, or the `reference_check` every CFD result carries),
  the check is a validation and not a consistency check, and the difference lies inside the
  tolerance band docs/validation.md checks that reference with: ± 2 % for the de Vahl Davis
  cavity at Ra 10³ and 10⁴ (± 3 % at 10⁵, ± 5 % at 10⁶), ± 3 % for the cubic cavity at Ra 10⁴,
  ± 4 % for the heated cylinder at Re 20-40 (where it was checked). Without a reference check the statement is **not supported**;
  outside the band, or with only the fan room's imposed thrust to compare, it is
  **contradicted**; either way it goes back to the model once. A quantity the result lists under
  `validated_quantities` from the documented validation (a polar's cl, a wing's lift slope) and a
  to-do ("must be verified by a grid study") are not such claims. A heading in the form the tools'
  own confidence note uses ("Validated: cl, cd, cm"; "Not validated: Cl_max") is judged on the list
  after its colon, so "Validated: Cl_max" is contradicted. (Live qwen3:8b had written "The
  result is validated against the reference setup" and the check had reported every statement
  verified, because it looked only at numbers and actions.)
* **A validated set-up says so, with its deviation.** A run that reproduces a validated reference
  set-up (the de Vahl Davis cavity, the Lange et al. heated cylinder, under the published
  conditions) lists its Nusselt number in `validated_quantities` with this run's measured
  difference and band ("this run 2.2509 against 2.238, +0.57 % ...; inside its ± 2 % band"), not
  under `not_validated`; the heat rate
  stays not validated, since no benchmark checks it directly. An answer that calls such a
  quantity "not validated" is contradicted. A run whose settings left the benchmark (another
  wall temperature, buoyancy switched on for the cylinder) keeps Nu under `not_validated`.
* **Claims must match what was done.** An answer that says something was deleted, run or created
  ("The simulation ... has been deleted from the project") when no tool call did that is sent back
  once. A local model cannot delete at all (it is told to point you to Project tree > Delete). If
  the corrected answer repeats the claim, the panel replaces it with a warning, folds the original
  away under *Withheld answer*, and the experiment ends **unverified** (warning glyph, in the panel
  and the Project tree) instead of "Done". The prompt rule and this check are independent: with the
  rule taken out of the local prompt, qwen3:8b again answered "… has been deleted from the project
  overview" (measured live, 28.5 s); the check sent it back and the corrected answer said it cannot
  delete. `record_conclusion` applies the same check to the conclusion it records: a summary or
  limitation that claims a deletion, run or creation no tool call performed is refused, so the
  log's *Conclusions* can never state it either (another live run tried to record "The simulation …
  has been deleted and is no longer available", then replied empty, and once asked to answer wrote
  "… has been deleted and cannot be accessed"; that claim was sent back too). The check reads the
  ways chat answers state an action, not only "I deleted": a line that starts with the verb
  ("Deleted “Polar · Re 3.0×10⁵” as requested.", "✅ Ran a polar at Re 10⁶"), "I went ahead and
  removed it", a headline ("Simulation “…” deleted.") and "the speed has been changed to 30 m/s".
  A clause that describes a configuration ("drag falls when the spoiler is removed"), a rule
  ("results are deleted when their design is deleted") or an idiom ("ran out of context") is not
  an action.
* **An empty reply is not an answer.** A model that ends with no text and no tool call is asked
  once to answer; a second empty reply ends the request **failed** with a plain message, never
  "Done" with nothing to read.
* **"Cannot" is checked against the tools.** When an answer says the task cannot be done and the
  goal's keywords route to a reference case (`aerolume.ai.router`: cavity, Rayleigh or natural
  convection to the de Vahl Davis cavity; a heated cylinder with Nusselt or Lange to the forced-
  convection cylinder; a fan to the fan room), and the model never tried that route, it is sent
  back once with the concrete calls, the parameters read from the goal ("call create_case with
  case="cavity", rayleigh=100000, then run_simulation ..., then compare_reference"). A refusal
  after the route was tried and failed (a build whose solver lacks heat transfer) stands. Found
  live: qwen3:8b refused a new cavity at Ra 10⁵ after 7.5 s, and the refusal ended "Done".
* **A comparison the goal asks for is in the answer.** When the goal names a reference (Lange,
  de Vahl Davis, Tric, momentum theory, "the published value", a benchmark or a correlation) and
  the answer quotes neither the reference value nor the difference in %, the loop calls
  `compare_reference` itself on the experiment's runs (a row in the panel, an action in the log)
  and sends the numbers back once; when no run has a reference, it sends the router's calls
  instead. An answer that still leaves the comparison out ends **not verified**, with a
  *Completeness check* in the log. Found live: asked for the cylinder's Nusselt number against the
  Lange correlation, qwen3:8b answered "Nu 3.3402 ± 0.001" (Lange 3.28, +1.8 %), and every number
  in it traced, so the claims check passed it.
* **Statements about the physical models must match the settings.** A statement that names
  buoyancy (Boussinesq, natural or free convection) or a turbulence model (LES, Smagorinsky,
  subgrid) is compared with the models the runs it is about used: naming a model that was off in
  all of them, or calling a model off that all of them used, is **contradicted**. Hypothetical
  clauses ("buoyancy would raise Nu") are not claims. Found live: the Lange answer above cited
  "the assumptions made in the Boussinesq energy equation" for a run with buoyancy off; the
  thermal validity sentence it copied named the Boussinesq model for every thermal run, and now
  says "buoyancy off (forced convection)" when it was.
* **The goal is covered.** When the goal names a design together with a speed, angle or Reynolds
  number ("Tapered wing ... between 15 and 25 m/s") and no run or read of the experiment matches
  one of them, the model is asked once to run it ("call run_wing on Tapered wing at 15 m/s").
* **Each number belongs to its run.** `record_conclusion` checks every finding against the
  simulation it cites and refuses a value that is not that run's, listing the run's actual values
  ("ld_max 29.52 at α 3°"); it also refuses a ± on a polar or wing value (they have no error bar;
  ±0.5° is the angle resolution of an optimum angle) and a degree uncertainty on a quantity that is
  not an angle. The final answer is checked the same way clause by clause: a number given for the
  25 m/s run that is the 15 m/s run's figure is sent back once and recorded as an *Attribution
  check* in the log.
* **Plans say what is varied.** A plan whose controls ("held fixed") include a quantity it varies,
  or the speed when the goal compares speeds, is refused; a plan without variables lists its
  *Settings*.
* **Stored results are reused, and said to be.** A polar or wing run whose angle range (at the same
  or a finer step, same speed and settings) is already stored is answered from the stored
  simulation; nothing runs and nothing waits for approval. A sweep that differs only in its range is
  named after it ("Wing polar · 25 m/s · α −5…15°") so the Project tree tells the two apart. A
  reused result's card ends "(stored result reused)", the model is told to say so, and when its
  answer does not, a line under it does ("Not run again: … is the stored result of an identical
  run computed earlier"). A goal that asks for a **new** run ("Run a new draft simulation of …",
  "re-run", "run it again") runs each simulation again once instead of reusing it; the approval
  card then shows the real cost.
* **Run cards lead with what was measured.** A CFD card shows Nu, the heat rate and fan thrust first,
  then Cd and Cl when the run has forces: a heated cavity reads "Nu 4.562 ± 1.9e-05, Q 0.114 W/m",
  not "Cd n/a, Cl n/a".
* **Plain text, no LaTeX.** Models write numbers as LaTeX out of habit ("$1 \times 10^5$", "$4.56
  \pm 0.01$", "$1.18\%$"). The prompts ask for plain text, and whatever LaTeX math still arrives is
  turned into Unicode as it streams -- "1 × 10⁵", "4.56 ± 0.01", "1.18%" -- so the transcript, the
  log and the claims check read the same text (`aerolume.ai.plaintext`). Prose with currency signs
  ("$5 and $10") is left alone.
* **Examples follow the project.** Every example goal on an empty transcript is drawn from the
  open project: its physics first -- the cavity's Nusselt number against de Vahl Davis, a heated
  body's Nu and heat rate, a fan's delivered thrust, the grid error of Nu -- then questions about
  its designs (an airfoil's best L/D and stall angle, a wing's induced drag, a 3D body's drag and
  how far to trust it). A slot the project has nothing for stays empty; the generic aerodynamic
  examples appear only in a project without designs. (An enclosure with heaters and fans had been
  offered NACA 4412, Ahmed bodies and a wing.)
* **Summaries keep their numbers.** A shortened line -- the plan card's "Plan recorded: ...", a
  log title -- is cut between words after its maths became text, and a number right after the cut
  is kept with its label ("... at Ra 1e4…", not "... at Ra , about 28 s").
* **Local models are held to the method.** With a local model the loop reads the project itself
  and hands the overview over with the goal (Ollama has no forced tool call, and qwen3:8b skipped
  the overview and recomputed a polar the project already had); the first run waits until the
  model has recorded its plan; and an answer after results without a recorded conclusion is sent
  back once to record it. When the model then repeats its answer word for word, the panel shows it
  once. A local model's context is kept for the work: stored results come back compact
  (metrics and confidence facts, validity as short codes; the per-angle table only on request), and
  a read that repeats an earlier one exactly is answered with a pointer to the earlier result.
  Metric names say what they are: `span_efficiency_at_best_LD` and `oswald_e_parabolic_fit`
  (qwen3:8b called the first "the Oswald factor").
* **Number check.** The log remembers every number that appeared in a tool result, a tool input or
  your goal, with where it came from (the claims check shows that origin as its evidence). The
  claims check also accepts a number from the solvers' validity record in the system prompt, but
  only in a sentence about what a solver is validated for ("validated against NACA 0012/2412/4412
  data"), never as a result. Numbers in the conclusions that match none of the remembered numbers
  (at the precision they are written
  with) are listed under *Number check* in the log and in the panel. A derived figure (a percentage
  the model computed) shows up there too, which is the point: you can see which numbers came
  straight from a solver.
* **A model that answers with numbers without running anything** is reminded once that numbers must
  come from tool results.
* **A mistyped id is refused, with the closest ids named** ("Did you mean the id 485fa5563c6c
  (“Tapered wing”)?"): qwen3:8b dropped one digit of a 12-digit design id in three of four live
  runs. The id is suggested, never used.
* **Every tool input is validated** against the tool's schema before anything runs (Claude streams
  tool input as it is generated, and a local model has no schema enforcement at all); an invalid
  call goes back to the model as an error it can correct. Lossless slips of small models (`"20"`
  for 20, one value where a list is expected, `null` for a parameter left out) are repaired,
  nothing else.
* Differences between CFD runs are called significant only when `compare_results` says so.

## Experiment logs

Every experiment is a folder in the project:

```
<project>/experiments/20261011-150821-find-the-angle-of-attack-with-the-best-l/
    experiment.md      the log as a readable note
    experiment.json    the same, machine-readable
    plots/01-polar.png figures the assistant made
```

The log holds the goal, the model, the plan (objective, method, variables, controls, quality,
expected runs and time, validity limits), every action with its outcome and duration, the key
results (quantity, value, unit, uncertainty, source simulation), the conclusions and findings, the
number check (with the design, attribution and action checks), the *Verification* table of the
claims check and the conversation. The status is Done, Not verified (the claims check or the action
check left a statement unsupported), Stopped or Failed. Values are written to 4 significant figures and every table cell
holds something (an em dash when empty), so the columns line up in any Markdown viewer. Deterministic
results say what limits them in the *Uncertainty* column: "±0.5° (half the α step)" for an optimum
angle, "lower bound: still rising at the last α; strips beyond section cl_max at 13–15°" for a
CL,max that did not peak, "profile drag not validated" for a wing's drag figures. Both files are rewritten after every step, so a log is complete
up to the moment a request was stopped. The Project tree lists experiments under **Experiments**
(double-click to open; *Delete log…* removes the folder, not the simulations). The token usage of
each experiment is in `experiment.json`.

## Scripts and the command line

The assistant and scripts are two front ends of one API. When a study is easier to express as code,
use the **Script** panel (Alt+4, [scripting.md](scripting.md)); the assistant itself can propose a
script through `run_python`, shown to you before it runs.

Experiments also run headless, with the same log:

```
python -m aerolume.ai --project "Study.aerolume" "find the angle of attack with the best L/D of NACA 2412 at 20 m/s using the polar analysis"
python -m aerolume.ai --project P --provider ollama --model qwen3:8b --approve runs "..."
```

`--approve ask` (default) asks in the terminal, `runs` approves long runs (code and deletions still
ask), `none` declines everything that would ask; without a terminal anything that needs approval is
declined unless `--approve` allows it. From Python:

```python
import aerolume as al
from aerolume.ai import run_experiment

out = run_experiment(al.open_project("Study.aerolume"), "compare the drag of ...")
print(out.status, out.path)
```

## Privacy and safety

* **Consent first:** nothing is sent to a model off this computer until you agree on the card
  described in [Before anything is sent](#before-anything-is-sent).
* **Claude:** your goals, the names, ids and notes of the designs and simulations the tools read,
  and the tools' results (numbers, warnings) are sent to Anthropic to produce the answers. Geometry
  files, meshes and result fields are not uploaded; figures are saved locally and only their file
  name is reported to the model.
* **Local model:** nothing leaves the computer; Aerolume talks only to the Ollama address you set.
* The tools read and write only inside the open project (figures and logs under `experiments/`,
  reports in the project's reports folder) and never use the network; the only network traffic is
  the conversation with the model provider.
* Text that comes back from tools - design names, notes, file contents, results - is passed to the
  model as data, and its instructions say that nothing in it is an instruction; if such text asks
  for something, the model is told to point it out to you instead.
* Code runs only after you approve the exact script, with the same Stop as the Script panel.
* Settings change only when you click **Apply** on a suggestion card, and **Undo** reverts it.

## Troubleshooting

| Message | What to do |
|---|---|
| No Claude credentials | set `ANTHROPIC_API_KEY`, run `ant auth login`, or enter a key in Preferences ▸ Assistant |
| Claude needs the optional anthropic package | `pip install aerolume[ai]` |
| Claude rejected the credentials (401) | the key is wrong or revoked; replace it in Preferences |
| Claude's rate limit was reached (429) / Claude is overloaded (529) | wait the time shown and run again |
| Ollama is not running at … | start Ollama, or check the address in Preferences |
| Model … is not installed | `ollama pull <model>`, or pick an installed model in Preferences |
| The local model's context is nearly full | start a new experiment, or raise the context in Preferences |
| Ollama could not run … cudaMalloc failed: out of memory | another program (often a CFD run) holds the GPU; wait for it to finish |
| Ollama could not run … std::bad_alloc | the computer is short of memory (check the commit charge in Task Manager); close programs and run again |
| Stopped after N tool rounds | the request needed more steps than the limit in Preferences; ask a narrower question or raise the limit |

On Windows, `localhost` is tried over IPv6 first; Ollama listens on IPv4, so Aerolume talks to
`127.0.0.1` when Ollama answers there (each request otherwise waited about 2 s for the IPv6 attempt
to fail).

A local model answers slowly when the GPU or the computer's memory is busy: while a CFD run, a game
or another program holds GPU memory, Ollama runs part of the model on the CPU. With the GPU fully
busy and about 2.3 GB of its memory taken by other runs, qwen3:8b (6.2 GB with the 8,192-token
context it then used) stalled while loading until Ollama gave up after 5 minutes, and once loaded took more than
the assistant's 300 s to read a 4,500-token prompt (measured 2026-10-18); keeping 16-24 of its 37
layers on the GPU loaded it in 14-22 s and the speed-comparison experiment then took 159-268 s.
Wait for the other GPU work to finish, or use Claude. Measured on an RTX
4060 Ti (8 GB) with `qwen3:8b`: one model turn with the experiment prompt and tools (about 3,600
prompt tokens) takes 7.5 s; the NACA 2412 best-L/D experiment above took 56 s end to end (three
tool calls, the polar itself 32 s), and 7 min 38 s while other GPU jobs ran at the same time.

## For developers

`src/aerolume/ai/` has no Qt: `config` (preferences, the key), `providers` (Claude through the
`anthropic` SDK, Ollama through `urllib`), `tools` (the JSON-schema tools over `aerolume.api`),
`schema` (input validation), `agent` (the loop: model turn, validated tool calls with approvals,
all results of a turn back together), `experiment` (logs and the number check), `prompts`,
`workspace` (the API project plus the window's hooks), `claims` (the claims checker), `router`
(goal keywords to the reference-case calls, and the comparison and "cannot" checks) and `cli`.
`aerolume.mcp_server` serves the same tools over MCP ([mcp.md](mcp.md)). The panel is
`src/aerolume/app/widgets/assistant_panel.py`; in the window, project edits go through
`aerolume.app.scripting.AppHost` and runs through the JobManager.

A new tool is a `Tool(name, description, schema, handler, kind)` in `tools.TOOLS`: the description
says when to call it, the schema must pass `schema.check_schema` (every object
`additionalProperties: false`, no numeric limits - the handler checks ranges and says why), and
`kind` decides the approval rule. Tests: `tests/test_ai_*.py`; `pytest -m slow tests/test_ai_live.py`
runs a real experiment with a local model.
