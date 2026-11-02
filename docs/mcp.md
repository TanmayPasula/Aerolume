# Driving Aerolume from Claude (MCP)

`aerolume-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server. It lets Claude
Desktop, Claude Code or any other MCP client plan, run and analyse experiments on an Aerolume
project through the same tools the in-app assistant uses ([ai_assistant.md](ai_assistant.md)).

No API key goes into Aerolume: the client's own model does the thinking, Aerolume does the physics
on this computer. The server speaks MCP over stdio (one JSON-RPC 2.0 message per line), uses only
the Python standard library, and never opens a network connection.

```
Claude Desktop / Claude Code ──stdio (MCP)──▶ aerolume-mcp ──▶ aerolume.api ──▶ polar, wing, CFD solvers
                                                   │
                                                   └──▶ <project>/experiments/<session>/  (the log)
```

## Starting it

```
aerolume-mcp [--project P [--create]] [--confirm-above SECONDS] [--read-only]
python -m aerolume.mcp_server ...        # the same, without the console script
```

| option | meaning |
|---|---|
| `--project P` | the project folder (`Name.aerolume`) to open at start; without it the client calls `open_project` first |
| `--create` | create the project when it does not exist |
| `--confirm-above S` | runs estimated longer than S seconds need `confirm: true` (default: the assistant's approval threshold, 120 s; `0` asks for every run) |
| `--read-only` | offer only the tools that change nothing (reading, estimating, comparing, checking claims); `open_project` opens existing projects only, and nothing is written into the project (the session log, which the claims check reads, lives in a temporary folder) |

The server writes nothing but protocol messages to stdout. Everything else (its start-up line,
solver output, warnings) goes to stderr, which Claude Desktop keeps in its MCP log
(`%APPDATA%\Claude\logs\mcp-server-aerolume.log` on Windows).

## Claude Desktop

Add the server to `claude_desktop_config.json` (**Settings ▸ Developer ▸ Edit Config**; on Windows
`%APPDATA%\Claude\claude_desktop_config.json`) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "aerolume": {
      "command": "C:/Users/you/venvs/aerolume/Scripts/python.exe",
      "args": ["-m", "aerolume.mcp_server", "--project", "C:/Studies/Wing study.aerolume"]
    }
  }
}
```

`command` is the Python of the environment Aerolume is installed in (a bare `python` only works when
that is the one on `PATH`). Once `aerolume-mcp` is installed as a console script you can use it
directly: `"command": "aerolume-mcp", "args": ["--project", "..."]`. Leave `--project` out to let
Claude choose a project with `list_projects` and `open_project`.

## Claude Code

```
claude mcp add aerolume -- "C:/Users/you/venvs/aerolume/Scripts/python.exe" -m aerolume.mcp_server --project "C:/Studies/Wing study.aerolume"
```

or, to share it with a repository, a `.mcp.json` in its root:

```json
{
  "mcpServers": {
    "aerolume": {
      "command": "C:/Users/you/venvs/aerolume/Scripts/python.exe",
      "args": ["-m", "aerolume.mcp_server", "--create", "--project", "studies/Wing study.aerolume"]
    }
  }
}
```

`claude mcp list` shows whether it connected; in a session, `/mcp` lists its tools.

## Tools

The assistant's tools with the same input schemas, the same validation and the same honesty data
in every result (Reynolds numbers simulated and real, convergence, standard errors with what they
mean, `validated_quantities` / `not_validated`), plus three for MCP clients:

| tool | changes the project | needs `confirm` |
|---|---|---|
| `open_project`, `list_projects` | opens (or creates) a project | — |
| `project_overview`, `describe_design`, `describe_simulation` | no | — |
| `create_naca_airfoil`, `create_template`, `create_wing`, `derive_design`, `create_case` | adds designs (and a case's simulation) | — |
| `estimate_cost` | no (the planner's estimate) | — |
| `run_polar`, `run_wing`, `run_cfd`, `run_sweep`, `run_simulation`, `run_grid_study` | adds simulations and results | above the threshold, or when the estimate fails |
| `get_results`, `compare_results`, `compare_reference` | no (`compare_reference` compares results with the published reference their set-up reproduces; the fan room is a consistency check only) | — |
| `make_plot` | a PNG in the session's folder (returned as an image too) | — |
| `write_report` | a PDF or HTML report in the project | — |
| `delete_item` | deletes a design or simulation | always |
| `record_plan`, `record_conclusion` | the session log | — |
| `check_claims` | no | — |

Not offered: `run_python` (an MCP client runs code itself) and `suggest_settings` (its card needs the
app's window).

`run_cfd` takes a `physics` object for heat transfer, walls, heaters, fans, jets, smoke emitters and
closed domains. A run that asks for physics this build's solver does not implement is refused,
naming what is missing, before anything is recorded.

### Confirmation

A call that needs confirmation and does not carry `confirm: true` runs nothing. It returns an error
result that lists exactly what would run, so the client can ask its user. `confirm: true` counts only
for a call (the same tool with the same input) that the server has answered this way earlier in the
session, and only once: a client that sends `confirm: true` on its first call gets the list back, so
the user always sees what would happen before it does.

```json
{
 "error": "Not done: this call needs confirm=true after the user agrees.",
 "needs_confirmation": true,
 "title": "Wing analysis of Wing · span 2 m, taper 0.6: about 25 s",
 "reason": "The estimated time (25 s; rough for polars and wings) is above the approval threshold (0.0 s, aerolume-mcp --confirm-above).",
 "would_do": [
  {"item": "Design", "value": "Wing · span 2 m, taper 0.6 · wing (3D planform)"},
  {"item": "Analysis", "value": "Finite-wing analysis"},
  {"item": "Speed", "value": "15 m/s"},
  {"item": "Angles", "value": "α −4° to 14°, step 1° (19 angles)"},
  {"item": "Method", "value": "vortex lattice, viscous strips"},
  {"item": "Estimated time", "value": "about 25 s (rough)"}
 ],
 "estimated_seconds": 24.95
}
```

(This reply was produced by the server started with `--confirm-above 0`.) The rules fail closed: a
run whose estimate fails is listed as "cost unknown" and also needs `confirm`; deleting always does.
A call whose own input is wrong is answered with the error and runs nothing. The tools'
`destructiveHint` and `readOnlyHint` annotations say the same to clients that show them.

### Checking an answer: `check_claims`

`check_claims(text)` runs the assistant's claims checker on a draft answer against this session's
tool calls and results. Each statement with a number or a claimed action comes back **verified**
(with where the number came from), **derived** (the formula: a difference, a % change or a ratio of
two results), **unsupported** or **contradicted** (an action no call performed, or a number that
belongs to another simulation, with the value it should have been), plus `ok` and, when something
is wrong, `correct_before_answering`. A statement that calls a result validated or verified needs a
reference check of the session inside its stated band, exactly as in the app (see the assistant
guide, "Validated needs a reference check inside its band"): the server keeps the session's tool
results (the newest 400) and the checker reads the validity lists, the steady quantities and the
reference checks from them. The server's instructions ask the client's model to call it before it
answers.

`record_conclusion` applies the same rules to what it writes into the session log: each finding
must be a value of the simulation it cites (the error lists that run's actual values), and a summary
or limitation that claims a deletion, run or creation no call of the session performed is refused.

## Progress, cancellation, records

* Calls that send a `progressToken` get `notifications/progress` while a solver runs (at most two a
  second, with the solver's message). The value only ever rises, as the spec requires: a fraction
  that does not exceed the last one sent (a solver whose estimate of the remaining work grew) is
  held back until the run passes it.
* `notifications/cancelled` for a running call sets the solver's cancel event: the run stops, keeps
  its partial result in the project marked "cancelled", and no response is sent for that request.
* Calls run one at a time on a worker thread, in arrival order (one GPU queue, one project store);
  pings are answered while a solver runs.
* Every call of a session is recorded in the project like an assistant experiment
  (`experiments/<date>-mcp-session-from-<client>/`, provider "mcp"), with the input, outcome and
  duration of each call, so what an outside agent did to a project can be read afterwards. The
  Project tree lists these logs under **Experiments**.

## Protocol details

* Protocol versions 2025-11-25, 2025-06-18, 2025-03-26 and 2024-11-05: the client's when it asks for
  one of these, otherwise the newest. From 2025-06-18 on, results also carry `structuredContent`
  (the same JSON as the text block).
* Methods: `initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`,
  `notifications/cancelled`. Anything else is answered `-32601` (method not found); an unknown tool
  `-32602`; a line that is not JSON `-32700`. A tool that fails returns a result with
  `isError: true`, so the model can read the reason and correct its call.
* On Windows the server never leaves a read waiting on its stdin pipe: it polls the pipe and reads
  only the bytes already there, and moves file descriptor 0 to NUL. A pending synchronous read
  makes other threads' queries of that pipe wait (measured: `GetFileInformationByHandle`,
  `PeekNamedPipe` and the old C runtime's `fstat` do), and scipy's mingw-built BLAS makes such a
  query while it loads, which froze the first wing run of every session before this was fixed.

## Troubleshooting

| symptom | cause and fix |
|---|---|
| the client says the server disconnected at start | the `command` path is wrong, or `--project` points to a folder that is not a project (add `--create`, or fix the path); the reason is the last line of the server's stderr log |
| every run comes back "needs confirmation" | that is `--confirm-above` (or the assistant's threshold) doing its job: confirm, or raise it |
| "This version's 2D solver does not implement ..." | the physics asked for is not in this build (`al.supported_physics`); nothing was run |
| a run takes much longer than its estimate | the GPU is shared with another program; the estimate assumes it is free |
