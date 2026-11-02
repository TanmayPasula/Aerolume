# Aerolume UI guidelines (v4)

Aerolume should look like engineering software that people have used for years —
the lineage of ParaView, Qt Creator, JetBrains IDEs, Ansys Discovery, XFLR5 — not
like a generated dashboard. It must still be easy: the workflow stays guided
(Geometry → Setup → Simulate → Results → Compare → Report), but guidance comes
from layout, defaults and precise labels, not from decoration and chatty copy.

v3 adopts semantic colour tokens whose contrast is tested in both themes, Atkinson
Hyperlegible type, and dockable panels with saved workspaces. Everything below is binding;
the code contract is `src/aerolume/app/theme.py` (tokens, hooks) and
`src/aerolume/app/widgets/docks.py` (panels, workspaces).

v4 (0.1.0) brings the app closer to the website: it carries over the site's language, not
its layout. The chrome takes the site's monochrome palette, ink
on paper in light and near-black in dark (`theme.SITE_LIGHT` / `SITE_DARK` record the
`website/assets/site.css` value each token came from, and `tests/test_tokens.py` checks them
against the file), the site's 6 px controls and 8 px frames, its two buttons (filled ink,
outline) and its ink focus ring with an offset. Every page, panel, workflow, shortcut and data
colour stays as it was: plots, colour maps, status colours and the blue marks in the data views
keep their meaning.

## 1. What to avoid (the "generated app" look)

Each of these is a defect, not a style choice:

* Navy backgrounds with neon teal/purple accents, glows, gradients, glass effects. (The dark
  theme's neutral near-black is the website's, with no glow or colour on it.)
* Hero banners, animated art, marketing taglines or feature cards inside the app.
* Cards everywhere; corner radii above 8 px (controls 6 px, framed blocks 8 px, as on the
  website); drop shadows.
* Decorative pill badges. More than one status chip in view is a smell; status is text.
* Dashboard KPI tiles with oversized digits. Results are tables with units and uncertainty.
* Two-line navigation items with subtitles; numbered circles; coloured icon tiles; emoji.
* Rainbow / jet colormaps.
* Vague or cute helper text ("Watch the flow live", "Rank your variants", "What can I do?").
  Explanations go in tooltips or the docs; labels are nouns, buttons are verbs.
* Centered layouts and large empty margins in a desktop tool; inconsistent spacing.
* A colour that is not a token. Pages set style hooks or read `theme.tokens()`.

The Start page's study tiles are the one place with boxes side by side: they are a
choice list with a schematic of what each study sets up (section 4), not feature cards. The
Start page's flow motif beside the product name is the one drawing that carries no
information: static, faint (`line_strong`) and a computed flow (section 4), the website's own
motif, never animated and on no other page.

## 2. Colour: semantic tokens with tested contrast

**Themes.** Light is the default: the website's ink on paper, neutral greys with no colour
cast (the norm for CAE tools, and the site's look). The dark theme is the website's
near-black, and *System* (Preferences ▸ Theme, View ▸ Theme) follows the operating system's
light / dark setting and switches when it does. No navy, no saturated backgrounds.

**Switching must be quick.** A theme switch is one application style sheet, applied by
`theme.apply()` in two linear passes (clear, then set): Qt's own "replace the sheet"
path re-polishes every widget once per ancestor, which measured 2.1 s on a window with
every page built (1,933 widgets) and 3.5 s with six results open (5,305 widgets); the
two-pass swap measured 0.27 s and 0.60 s. `tests/test_app_shell_perf.py` keeps it under
0.8 s offscreen. Never walk widgets to re-polish them after a theme change; a widget that
paints by hand re-reads its tokens in a slot on `theme.changed`. Every token is an
opaque `#rrggbb`; `theme.semantic_tokens(name)` lists them, `theme.tokens()` adds the
v1/v2 names as aliases (`text` = `fg`, `panel` = `surface_2`, `border` = `line`,
`accent_fill` = `primary`, `good` / `bad` = `success` / `danger`, ...).

| token | light | dark | use | website (light / dark) |
|---|---|---|---|---|
| bg | `#efefef` | `#0a0a0a` | window chrome: toolbars, status bar, dock rails | a step darker than `--bg` / `--bg` lifted a notch |
| bg_subtle | `#e9e9e9` | `#151515` | panel headers, title rows | `--code-bg` / — |
| surface | `#ffffff` | `#1a1a1a` | inputs, tables, plot and viewport wells | `--surface` / `--code-bg` lifted a notch |
| surface_2 | `#f6f6f6` | `#111111` | side panels, inspectors (alias `panel`) | `--bg` / — |
| fg | `#0e0e0e` | `#f1f1f1` | primary text (alias `text`) | `--fg` |
| fg2 | `#474747` | `#b1b1b1` | secondary text: property labels, captions (alias `text2`) | `--muted` |
| muted | `#5c5c5c` | `#a3a3a3` | hints and units — still text, so ≥ 4.5:1 | `--faint` / — |
| faint | `#8e8e8e` | `#6b6b6b` | disabled text and icons only | |
| line | `#d6d6d6` | `#2b2b2b` | decorative 1 px separators (alias `border`) | about `--line` |
| line_strong | `#b8b8b8` | `#3d3d3d` | table grid, stronger separators, the rule under table headers | |
| control | `#858585` | `#7a7a7a` | outline of inputs, buttons, check boxes, sliders, the off switch | `--line-strong` / — |
| primary | `#0e0e0e` | `#f1f1f1` | primary button and checked-box fill: ink | `--btn-bg` |
| primary_fg | `#ffffff` | `#0a0a0a` | text on primary (dark text on the dark theme's light fill) | `--btn-fg` |
| primary_soft | `#e2e2e2` | `#2e2e2e` | selected rows and items, checked segments | |
| accent | `#2463c4` | `#7fb1f7` | blue marks in the data views: the selected item of a canvas, the seed rake, the running state | (none: the site has no blue) |
| highlight | `#b87508` | `#e2a640` | amber, non-text marks only: heat, the live state, emitters | |
| success / warn / danger | `#17703c` `#8f5a0b` `#b02d27` | `#6dcb93` `#e6b765` `#f5928a` | status text and small glyphs | |
| *_soft | `#e3f1e8` `#fbf0db` `#fae4e2` | `#13271b` `#2b2213` `#331b1a` | the background an inline notice sits on | |
| focus | `#0e0e0e` | `#f1f1f1` | keyboard focus border / ring | `--focus` |

Plot series (unchanged in v4: they are data): light `#2463c4 #d9730d #2f855a #b8322b #6b46c1
#5f6b7a` (also the report figures); dark one step lighter `#5e9df4 #f0943f #4fb47b #f0716a
#a68af2 #a3acb9`, because the light-theme blue was 2.6:1 on the v3 graphite surface (on v4's
near-black the lightest steps measure 6.0:1 and up). Field-view chrome: `field_bg`,
`field_grid`, `field_text`, `field_solid` (neutral mid-grey solids), `field_solid_line`.

**Contrast rule** (WCAG 2.x, enforced by `tests/test_tokens.py` in both themes: 73 rules over
68 colour pairings, listed in `theme.CONTRAST_RULES`; a pairing may serve two rules, as text and as a
checked border):

* text (`fg`, `fg2`, `muted`) on every background it is drawn on, and blue text, resting
  (`accent`: "Running", data labels) and the step further (`accent2`): **≥ 4.5:1** (links
  are ink since v4 and hover to `fg2`, both text tokens);
* status text on the surfaces and on its own `*_soft` background, and `*_fg` on a solid
  status fill: **≥ 4.5:1**;
* control outlines (`control`, also the scroll-bar handle), the focus indicator and the
  primary fill against the surfaces they sit on, the `highlight` marks, the field view's
  solids and every plot series: **≥ 3:1** (WCAG 1.4.11);
* `faint` is for disabled controls only (exempt under WCAG 1.4.3) and stays ≥ 2.2:1;
* `line` is decorative (1.4:1 light, 1.2:1 dark) and never the only edge of a control.

Measured on `surface` (v4): light `fg` 19.3, `fg2` 9.3, `muted` 6.7 (5.8 on `bg`, 5.5 on
`bg_subtle`), `accent` 5.7, `control` 3.7 (3.2 on `bg`, 3.04 on `bg_subtle`, the closest
pairing in the table, as the site's own `--line-strong` is); dark `fg` 15.4, `fg2` 8.1,
`muted` 6.9, `accent` 7.9, `control` 4.1. Plot series at least 3.3:1 (light) and 6.0:1 (dark).
v3 measured light `fg` 16.5, `muted` 6.2, `control` 3.9; dark `fg` 10.7, `muted` 6.2.

**Rules of use.** The chrome's interactive states are ink, as on the website: the primary
button, checked boxes and radios, the current tab and step (a 2 px ink underline), links
(ink, underlined), checked icons (`theme.ACTIVE_ICON_TOKEN` maps a call site's `"accent"` to
`fg`), the focus ring and slider fills. The blue `accent` is kept for marks in the data views,
where the selected or live item has to stand apart from ink outlines (the selected body in the
geometry canvas, the seed rake, the drawn obstacle, "Running" and its progress bar). Status
colours are for text, small glyphs and text on their soft background (`QFrame[notice=...]`),
never a solid fill behind body text. `highlight` is for marks that mean "heat" or "live" — the
radiator in a schematic, a running indicator — and never carries text.

## 3. Type

**Atkinson Hyperlegible Next** for the interface and **Atkinson Hyperlegible Mono** for
numbers in tables, code and the log (SIL OFL, bundled as TrueType in
`app/resources/fonts`; provenance and checksums in its README). They were drawn for
low-vision readers: 0 / O, 1 / l / I and 5 / S cannot be confused, which matters in a
tool whose output is numbers. Qt cannot load the published WOFF2 files, so they are
converted once (`build_fonts.py`). The Latin subsets have no Greek, superscripts above
³ or arrows: every font stack ends with Segoe UI, which Qt uses glyph by glyph (α, ν, ω,
10⁵, →). Preferences ▸ Interface font ▸ System uses Segoe UI throughout.

| role | size | weight | hook |
|---|---|---|---|
| body, controls | 9 pt (12 px) | Regular | default |
| footnotes, units in tight places | 8 pt | Regular | `role="small"` |
| group and section titles | 9 pt | SemiBold | `role="h3"`, PropertyGrid headers |
| page and dialog titles, Start tile titles | 10 / 11 pt | Medium | `role="h2"` / `"h1"` |
| the product name (Start, splash) | 14 pt | Medium | `role="title"` |
| numbers in tables and logs | 9 pt Mono | Regular | `role="mono"`, `theme.mono_font()` |

Nothing larger than 14 pt anywhere; Bold is not used at 9 pt (its hinting fills
counters). Titles from 10 pt up are Medium, as the website sets its headings (v4; SemiBold
before); 9 pt group titles stay SemiBold, where Medium is too close to Regular to rank. Atkinson SemiBold is about 7 % wider than Segoe UI Semibold on Windows and the
Mono 14 % wider than Consolas: size columns from `QFontMetrics`, not constants. Numbers
right-aligned in tables; proper symbols (α, C<sub>d</sub>, Re = 8.2×10⁵, m/s, N/m).

**Shape and spacing.** 4 px grid. Controls 24 px high; table rows 22–24 px; radius 6 px
on inputs and buttons (`theme.RADIUS`, the website's controls), 8 px on framed blocks — Start
tiles, cards, the tour's step box (`RADIUS_LG`, the site's frames) — 3 px on marks inside a
control (check-box indicators, scroll-bar handles), 0 on panels (panels are separated by 1 px
lines). Buttons are the site's two: secondary is an outline in `control` that turns ink on
hover, primary is filled ink. Table headers have no fill of their own and a `line_strong`
rule under them. No shadows, no gradients. Focus: a 1 px `focus` (ink) border on inputs and
buttons; the primary button and a checked box, whose fill is the focus ink, get the website's
offset ring instead — a 1 px `fg` ring with one pixel of the panel showing inside it (the fill
is clipped to the content box; the geometry does not change, so nothing moves when focus
arrives); custom-painted widgets use `theme.paint_focus_ring`, and the Start page's study
tiles paint it only for keyboard focus (Tab, Shift+Tab, F6 or a key press: the site's
`:focus-visible`), not for the window's initial focus or a click. **Checked state**: a
checked tool button or segment is a `primary_soft` fill inside a 1 px `fg` border — the
fill alone is 1.1–1.5:1 against the panels, the border ≥ 3:1 (WCAG 1.4.11, the
"checked indicator" rules). A focused checked tool button shows the panel in its 2 px
padding between ring and fill; a focused segment gets a 2 px ring (border and padding
always add up, so nothing moves). Rich-text links are re-laid out on a theme switch
(`theme.apply`), since Qt bakes the link colour in when the text is set. Widgets with no border of
their own get a stronger mark: a group header (text on the panel) paints a 2 px `focus`
ring around its arrow and title, a slider's handle a 2 px `focus` ring, and borderless
item views (`flat`, `pane`: the Project tree, the Start page lists) keep a transparent
1 px border that focus colours, so the content never moves. Every focusable widget of
the Setup and Start pages changes pixels on focus in both themes
(`tests/test_app_shell_release.py`); v3.3 found the "Heat transfer" header, the angle
slider, the Project tree and both Start lists unchanged.

An independent audit of the pairs the style sheet draws beyond the rule list (v3.1) found
three that the list had missed, all fixed and now in it: hovered links in `primary_hover`
(4.37:1 on the dark surface; then `accent2`, `fg2` since links turned ink in v4), scroll-bar
handles in `line_strong` (1.7–1.9:1; now `control`), and the primary button's 1 px focus ring
(above; since v4 the offset ring).

**Icons.** 16 px monochrome line icons with a 1.25 px stroke in `fg2`; ink (`fg`) for the
active state (v4; it was the blue `accent`). One coherent set, drawn as SVG (`resources/icons`, tests enforce it).
Every icon engine applies the active-state mapping: the theme's bundled set, the hand-drawn
glyphs of the parameter forms (Points, Dimensions), the live-edit tools and the 3D camera
presets (`tests/test_tokens.py` checks all four in both themes).

**Motion.** None except progress indication. (The Start page's flow motif is a still
picture, drawn once per size and theme.)

**Accessibility** (WCAG 2.1 AA, desktop reading; checked by an automated pass over the
shell in both themes - every visible widget of the Start page, each workspace on four
pages, the tour and Preferences):

* Every input has a name a screen reader can read. A label merely placed beside a field
  does not give it one in Qt (measured: `QAccessible...text(Name)` is empty), so form
  rows go through `PropertyGroup.add_row`, `common.tie_label(label, field)` or a
  `QFormLayout` row with a *string* label. A composite editor's buddy is its first
  focusable part; its other parts (a unit selector) are named "<label>: <tooltip>".
  A page the window hosts but does not build (Preferences ▸ Assistant) is passed through
  `common.tie_form_labels(page)`, which does the same for every `addRow(QLabel, field)`.
* Everything a pointer can do has a keyboard path: a toolbar button through its menu
  action, a panel through View ▸ Panels (Alt+1 … Alt+7), the account menu through its
  shortcut, the tour through Esc / arrows / Enter.
* **Focus order follows the screen, page first** (WCAG 2.4.3). Tab goes step bar → page →
  the panels (left rail, right rail, bottom rail, each in rail order) and wraps round;
  `MainWindow.focusNextPrevChild` sorts Qt's chain by region and keeps Qt's order inside
  each. Qt's chain is creation order, and the panels are built before most of a page, so
  v3.4 counted 29 stops of panel chrome, Messages filters and Diagnostics between the step
  bar and the Setup form's first field. The panel headers' collapse, ⋯ and × buttons are not
  Tab stops at all: **F6 / Shift+F6** cycle the regions as in Office and VS Code (step bar,
  page, then each panel on screen), land on a panel's *header* (2 px focus ring), where
  Enter or Space collapses or expands it, Shift+F10 or the Menu key opens its ⋯ menu, Delete
  hides it and Tab goes on into its content. F6 back to the page returns to the field that
  last had focus there. The header takes focus only from F6, never from a click, so
  dragging a panel never takes focus from the editor beside it.
* Text contrast comes from the tokens (above); the automated pass measures the pixels of
  every fully visible label and button text, cropped from a grab of the whole window
  (a transparent label grabbed alone renders on black).
* Scroll areas are not Tab stops of their own when their content takes focus, or when it
  fits: Tab reaches the fields and the area scrolls each into view. `common.
  unfocus_scroll_areas` runs after every page switch; a read-only area that overflows
  keeps its stop for keyboard scrolling and is named after its first heading. v3.3
  counted four nameless stops on Setup (the rail holder and three scroll areas).
* Target size (WCAG 2.2 SC 2.5.8): every button is at least 24 × 24 px (`theme.
  MIN_TARGET`; `IconButton` never goes below it, whatever size is asked: the panel
  headers asked for 22, the Messages bar for 20), or meets the spacing exception: a
  24 px circle on its centre meets no other target's. The test measures it on Setup and
  Geometry. Since v3.4 the style sheet makes the common controls 24 px tall outright rather
  than leaning on the exception: a check box's whole row (`min-height: 24px`; it was 16 px,
  the indicator's height), the segments of `SegmentedControl` (borderless, so the 18 px
  content height every bordered button uses left them at 22 px) and the main toolbar's
  buttons (they were 22 px; the window chrome grew by 2 px). The account button in the menu
  bar is still 35 × 22 px — the menu bar sizes its corner widget to its own row — and passes
  by the spacing exception, with no other target within 24 px.

v3.1 pass (both themes, 1600 × 1000; Start, the four workspaces on Geometry / Results /
Simulate / Setup, the tour, the three Preferences tabs): no text under its WCAG ratio
except behind the tour's scrim, which dims the window on purpose while the step box (at
full contrast) is read. Unnamed controls found and fixed in the shell: the nine fields of
Preferences ▸ Assistant. Still unnamed, in pages the shell does not own: the Geometry
page's parts table and one combo box, the Assistant's goal box, and the Script panel's
editor, output, script list and "More" button.

## 3b. Panels, rails and workspaces

```
┌ Menu: File  Edit  View  Simulation  Tools  Help                              [account] ┐
│ Toolbar: New Open Save │ Import ▾ │ ▶ Run ■ Stop │ Report │ Assistant Script   Workspace ▾ │
│ Steps:  Geometry   Setup   Simulate   Results   Compare   Report        3D │ design › sim │
├──────────────┬──────────────────────────────────────────────────────┬───────────────────┤
│ ˅ Project ⋯ ×│                                                      │ ˅ Assistant   ⋯ × │
│  ▾ NACA 2412 │            page content / viewport                   │   (right rail)    │
│     ● α 4°   │            (each page keeps its own inspector)       │ ˅ Diagnostics ●⋯ ×│
├──────────────┴──────────────────────────────────────────────────────┴───────────────────┤
│ ˃ Messages  0 errors  2 warnings  …newest line…                                     ⋯ × │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ Status: Ready │ Running NACA 2412 α 4° 43 % · 1.25 M cells · 61 steps/s │ CUDA · RTX 4060 Ti │ SI │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

Every side panel — Project, Messages, Assistant, Script, Properties, Flow visualisation,
Diagnostics — is a dock with the same header: collapse chevron, title (9 pt SemiBold on
`bg_subtle`), optional extras (the Messages counters), an optional on/off **switch**, the
⋯ menu and ×. What a user can do with any panel:

* **collapse** it to its header (chevron, double-click on the header, or ⋯ ▸ Collapse);
* **reorder** it in its rail (⋯ ▸ Move up / Move down, or drag the header);
* **move** it to the left, right or bottom rail (⋯ menu);
* **hide** it (× or ⋯ ▸ Hide panel) and bring it back from **View ▸ Panels**
  (Alt+1 … Alt+7);
* run a rail in **solo mode** (⋯ ▸ Solo mode): opening one panel collapses the others there;
* resize a rail by its edge; a double-click on the edge restores the workspace's widths.

**Header switches** toggle a panel's *effect* while keeping every value (an A/B
switch): Diagnostics ▸ record on/off; flow-visualisation layers. A setting that is part of
a form stays a check box. The switch is a 28 × 16 track: off = `control` outline, on =
`primary` fill with the knob at the right (the state never relies on colour alone). A
switch belongs to the content it was enabled for (`DockPanel.enable_switch(...,
content=)`, default: the content on screen), so a page's switch shows only while that
page fills the panel and never stands in for the window's own.

**Workspaces** (toolbar right end, View ▸ Workspace): *Default* (Project left, the log
folded at the bottom), *Analysis* (Assistant and the open log beside the results),
*Live* (Flow visualisation on the right of a live run, Diagnostics under it), *Scripting*
(the Script panel takes 40 % of the window height, the Assistant on the right). Each keeps
its own arrangement across restarts; Reset puts the preset back. Panels never float. A preset that changes gets a
higher `rev`, and an arrangement saved from an older revision is replaced by the new
preset once (Live is rev 2: Diagnostics' list-beside-plots content needs 592 px, which in
the right rail left the 3D Simulate page 798 of 1600 px and, capped at 1280 px, hid its
plots behind a sideways scroll bar). Scripting's bottom rail is a *share* of the window
height, not pixels (rev 2): stored in pixels and re-learnt from the rail's geometry, it
shrank to whatever a small window had left (173 px after a 1280 × 800 session) and the
editor showed three lines at 1600 × 1000. A share-sized rail is learnt only from a drag of
its edge (accepted between 15 and 60 %), and the Script console keeps the same share across
window resizes. Rev 3 lowered the share from 54 to 40 % (`"shares": {"bottom": 0.40}`, at
least 300 px): at 54 % the Setup page above kept about 300 px, a 180 px 3D preview and
truncated Estimate and Planner lists (v3.4). Measured over a Heated room's Setup page: the
Script panel 400 of 1000 px with 20.6 editor lines and the page 448 px at 1600 × 1000;
314 of 800 px with 15.2 lines and the page 334 px at 1280 × 800.

**A new study never opens in a tool workspace.** Scripting is for a task, not a study's
starting layout: a study created there opened with an empty "Untitled" script over half
the window. `DockManager.study_workspace()` names the workspace a new study opens in — the
current one, or when that is Scripting, the last other workspace the user had (kept in the
settings file as `workspaces/last_study`; Default when there was none). The Messages log
says so, and View ▸ Workspace ▸ Scripting brings the Script panel back with its text.

**The page area reports the current page's minimum.** A `QStackedWidget` takes its
minimum from every page it holds, so after a visit to Results (432 px minimum height) the
bottom rail could never have the height Setup (334) or Simulate (274) give up.
`main_window._PageStack` reports only the page on screen; a switch to a taller page hands
the height back from the rails. The Geometry page still needs 431 px, so at 1280 × 800
the Script panel over it shows about 9 lines.

**Narrow windows.** Below 1180 px the right rail folds into the left one as tabs and comes
back at 1220 px, where it was. Below 1400 px each side rail is capped at 28 % of the
window width (a panel's own minimum, the Assistant's 320 px, still wins): at 1280 px a
content minimum used to widen the Live workspace's right rail to 400 px and leave the 3D
view about 350 px wide. At every width a side rail is also held to what leaves the page
its own minimum width beside the other rail (`DockManager.page_guard_caps`): without it a
592 px panel moved to the right rail at 1400–1520 px squeezed the 3D Simulate page to
598–640 px against its 711 px minimum and cut it off. Pages must still work at 1280 × 800
with one rail open.

**Sharing the bottom rail.** Opening Script or Diagnostics folds Messages to its title row
(its counters and newest line stay visible); two open panels plus the log left the page
160-250 px of height at 800 px.

**On the Start page** the project panels step aside, but the Assistant and Script panels
(which need no project) stay when wanted: a lit toolbar toggle always means a panel on
screen.

**Panels with nothing to show** (Properties, Flow visualisation and Diagnostics on a page
that provides none) are hidden and their View entry is disabled with the reason as its
tooltip — never an empty panel. Pages fill them through `dock_panels()` (docks.py).

* **Project tree** (left): designs with their simulations nested; status glyph per
  simulation (not run, queued, running with %, done, failed, stopped); context menu (new
  simulation, run, duplicate, rename, delete). Names elide at the end; the tooltip has the
  whole name. Selection drives `AppState`.
* **Step bar**: flat tabs with a 2 px ink underline on the active step and a small check
  on completed ones. No numbers in circles, no subtitles. The breadcrumb at its right end
  names the design and the simulation, once when both carry the same name (the physics
  templates name a case's simulation after its design; "Heated cylinder · d 10 mm · ΔT
  10 K › Heated cylinder · d 10 mm · ΔT 10 K" took 400 px).
* **Properties** (in a page or the Properties panel): a property grid — label column
  (`fg2`), editor right, groups with a thin header line and a collapse arrow; units inside
  or right of the editor.
* **Messages**: what the planner decided and why, warnings, errors, the run log. Folded to
  its header by default (counts and the newest line); it opens by itself for errors only.
* **Status bar** carries device, run progress and units — no device chip in the toolbar.

## 4. Pages

* **Start** (no project open): the product line (mark, 14 pt name, one sentence) with the
  website's flow motif in the room to its right (`pages.start.FlowMotif`: short streaks along
  the streamlines of inviscid flow past a cambered Joukowski airfoil at 4°, the same flow the
  website draws, in `line_strong`, fading in from the left; one cached pixmap per size and
  theme, no timer, no focus, no accessible name; the mark inverts on the dark theme); **New
  study** as one row of tiles — Airfoil (2D), Vehicle (2D), 3D body from a mesh, Wing (3D),
  Heated room (3D, heat transfer), Electronics enclosure (2D, heat transfer: the room takes
  tens of minutes on a Draft GPU run, the enclosure 204 s, so a first heat study can finish
  while the user watches). When the width cannot hold every tile at its title's width (beside
  the Assistant at 1280 px) the row wraps into balanced rows (three and three), never wider
  than the window. Every tile has one height, the tallest description's at the tiles'
current width; each tile reports its own width change and the page re-fits after the grid
has settled (fitting on the page's resize used the old, wider widths, and on the first show
at 1600 × 1000 the descriptions lost their third line). Measuring and painting share one
text box (`StudyTile._text_width`): they had differed by the 1 px border inset, enough to
wrap a line one word earlier when painted than when measured. Each tile has a line schematic of what the study sets up
  (free stream or closed box, the body, the heated part in `highlight`), its dimension and
  one sentence of what it starts with; the tiles are buttons (Tab, Enter, Space) with the
  sentence as their accessible description. Below: Recent projects (name, type, folder,
  modified) and Examples (3D first) beside an **Open** column (Open project…, Open example,
  More study types…) and **Learn** (Welcome tour, Scripting guide, Experiment assistant,
  Driving Aerolume from Claude, Keyboard shortcuts). Footer: Documentation, Validation,
  Licence and the version. Each help link once. At 1280 × 800 the tiles, Recent projects
  and the first examples (the heat-transfer ones among them) are above the fold; the rest
  scrolls (131 px with five recent projects and seven examples). Seven examples ship with finished results: the 3D sedan, the tapered
  wing, the NACA 2412 section and the sedan-fastback pair, plus three heat-transfer ones
  (the de Vahl Davis cavity at Ra 10⁵, the heated cylinder at Re 40 and a 2D room with a
  fan, a heater, a supply jet and a vent), listed after the 3D ones with a thermometer, so
  every Results tab (Heat transfer, Fans) has a finished study to show.
* **Geometry**: CAD-like viewport — `field_bg`, light grid, outline in `fg` with a light grey
  fill, dimension lines in `fg2`, hatched road. Inspector = property grid.
* **Setup**: property grid groups (Analysis, Flow, Attitude, Ground, Mesh & time, Physics:
  emitters, heat, fans, ports; Solver (collapsed)). Right column: the Estimate table and the
  planner's messages as plain lines with a small glyph.
* **Simulate**: the viewport takes the space. Small viewport toolbar (field, colormap,
  overlays, fit). Vertical colour bar at the right edge of the viewport. Right panel "Run":
  thin progress bar, phase, t, steps/s, ETA; "Monitors" plot; "Explore" (live) collapsed.
* **Results**: one header line of facts plus a single warning line when needed, then tabs.
* **Compare**: table first (metrics × variants, deltas and %), plots below.
* **Report**: form + list of generated files.

## 4b. Live editing (Explore, AeroJAX-style)

* Drawing obstacles, moving emitters, fans and heaters happens in the viewport with the
  tools of a small tool strip (brush, polygon, eraser, move; brush size; undo / redo /
  clear). A tool is a checkable icon button with a tooltip naming it and its key.
* What is drawn shows as the solids do (neutral grey, 1 px outline); a placed item shows
  as its glyph with handles while selected. Heat sources and hot bodies use `highlight`.
* Changes apply to the running flow without a restart; the Run panel says so in one line
  ("Obstacle added at t = 12.4") rather than with a toast.
* Diagnostics (divergence, change per step, CFL, dt, residual) live in the Diagnostics
  panel, logarithmic where they span decades, exportable as CSV; the solver trace opens
  as its own tab of that panel.

## 5. Plots and fields

* pyqtgraph styled like publication plots: surface background, 1 px axes in `fg2`, grid
  alpha ≈ 0.15, small boxed legend, axis labels with units. Averaging windows are a light
  grey band, not a saturated fill.
* Colormaps: speed `viridis`; Cp `coolwarm` (centred on 0); vorticity `RdBu_r` (centred);
  smoke greyscale on a dark-grey field; ν_t `magma`; temperature θ `inferno` from the
  ambient (heat is the one field where warm colours mean warm). Never jet/rainbow.
* Solids are drawn in `field_solid` with a 1 px `field_solid_line` outline.
* **3D colour maps** (`viz3d.colors.FIELD_INFO_3D`; the same lookup tables as 2D):

  | field | map | range |
  |---|---|---|
  | speed \|V\|, velocity u | `viridis` | speed from 0 |
  | velocity v, w; pressure p; Cp | `coolwarm` | centred on 0 |
  | skin friction Cf (and the wall-shear magnitude τ) | `viridis` | from 0 |
  | vorticity \|ω\| (a magnitude in 3D) | `magma` | from 0 |
  | Q-criterion; sensitivity dJ/dn | `RdBu_r` | centred on 0 |
  | eddy viscosity ν_t | `magma` | from 0 |
  | solid fraction | greyscale | 0–1 |

  Ranges are robust percentiles; diverging fields are symmetric about 0.

## 5b. 3D viewer

The 3D view (`app/widgets/view3d.py`) follows the same restraint as the 2D canvases; it
should read like ParaView or a CAD viewport, not a game.

* **Background** is `field_bg`, flat — no gradient, no skybox, no vignette. A ground plane
  with a light grid only where there is a road.
* **Bodies** without a field are the neutral `field_solid` grey, smooth-shaded under a white
  light kit with low specular, crease edges at 45°; triangle edges only on request.
* **Fields on surfaces and slices** use the maps above. Slices are unlit, so a colour matches
  its colour bar exactly. Streamlines are thin tubes coloured by speed; smoke is greyscale.
  Emitters show as small `highlight` glyphs where they release.
* **Colour bar** vertical at the right edge, title with units, end triangles when values
  exceed the range. It never covers the model.
* **Camera**: perspective 25° (parallel on request); presets Iso, Front, Side, Top, Back,
  Bottom; turntable orbit; left drag orbits, right / middle or Shift+left pans, the wheel
  zooms toward the cursor, double-click or F fits.
* **Symmetry-plane results are always shown whole** (mirrored).
* **Empty and failed states** are plain text in the viewport ("3D view unavailable" with
  the reason). Never a blank grey rectangle.

## 5c. The assistant

* A suggestion arrives as a card listing every change, old value → new value; nothing is
  applied without a click; every value is re-validated against the editor's limits; one
  suggestion is one undo step.
* Project data reaches a model only with consent, and the panel says which provider runs
  (Claude, or a local Ollama model). No API key is ever shown back.
* Claude Desktop and Claude Code can drive Aerolume without a key in the app through the
  `aerolume-mcp` stdio server (Help ▸ Driving Aerolume from Claude).

## 5d. Start-up, tour, errors

* **Splash** (while PyTorch and VTK load): a plain 520 × 300 panel — mark, name, one
  sentence, the step being loaded ("Loading PyTorch…") and a thin bar. No art.
* **Welcome tour** (first start; Help ▸ Welcome tour): steps over the real window, each
  dimming everything but the widget it is about; Back, Next, Skip tour; Esc skips. A tour
  explains only what is on screen, so it has two parts: on the Start page three steps
  (New study, Examples, Learn); with a project open five (step bar, Project panel, Run and
  Stop, workspaces, Script and Assistant). Finishing the Start part runs the project part
  the first time a project opens; skipping either part ends the tour for good. A step
  whose widget is hidden (a panel the user closed) says how to show it instead of
  pointing at nothing. A step about two controls outlines both (Run and Stop; Assistant
  and Script), and a box for a toolbar control sits below the whole toolbar, never over
  the neighbouring buttons.
* **Unexpected errors** open one dialog per burst: the one-line error, a warn notice that
  the app keeps running and where the report was saved, the redacted report (home folder
  as `~`) with Copy report and Open log folder.
* **One instance**: opening a project while Aerolume runs opens it in the running window.
* **The desktop**: creating or opening a project writes its `project.aerolume` marker (the
  file the installer's association opens) and, in an installed build, adds it to the
  taskbar jump list; folders inside the program (the bundled examples) are never written.
* **Check for updates** (Help) is off until a release feed is set (Preferences ▸ Release
  feed; there is no published feed yet). It is one GET of that address on a worker thread
  and sends nothing; the answer says the version found, or why the feed could not be read.

## 6. Copy

Sentence case. Labels are nouns ("Angle of attack", "Ground clearance"); buttons are
verbs ("Run", "Import…", "Export CSV…", "Copy report"). Status is factual ("Not run",
"Running 43 %", "Done in 3 min 24 s", "Failed — see Messages"). An action keeps its name
through the flow (the Run button, F5 and "Run again" all say Run). Errors say what
happened and what to do, never apologise. No exclamation marks, no emoji, no
second-person cheerleading. Explanations live in tooltips and the user guide; a number the
program filled in is shown, not hidden.
