# Accounts (optional)

Aerolume needs no account. Projects, simulations, results and reports live in
folders on your computer, and every feature works offline. An account adds three
things on top:

* **Cloud projects**: a copy of a project folder in the cloud, so you can pick
  it up on another computer, with a clear status when the copies differ.
* **Share links**: a read-only, expiring link to a result's report.
* **A profile**: display name, affiliation, picture and default units/theme.

## The honest constraints, up front

* **There is no Aerolume server.** Accounts run on a [Supabase](https://supabase.com)
  project that *you* create and own (free tier is enough to start).
  `supabase/README.md` walks through it. Until one is connected, the Account
  button says "Not connected" and nothing is ever sent anywhere.
* **Email links finish sign-in only on the computer running Aerolume**, while the
  app is waiting (the link redirects to `http://127.0.0.1:<port>/callback/<one-time nonce>`). On a
  phone or another computer, type the 6-digit code from the same email instead;
  the project's email templates must include that code (see `supabase/README.md`).
* **Supabase's built-in email service is for testing**: a few emails per hour,
  and on new projects only to members of your Supabase organisation. Real use
  needs custom SMTP.
* **Upload size**: the free plan accepts *files* up to 50 MB, and a normal
  project with a few 3D results packs to more than that (one 3D run takes about
  15 MB). Aerolume therefore uploads a larger project in parts of up to 45 MB and
  joins them again on download, so the 50 MB limit never blocks a project. What
  does bound a project is the plan's total storage (1 GB on the free plan when
  this was written; an upload briefly holds the old and the new version) and
  `max_project_mb` below. See *Upload size* under Cloud projects.
* **Share links show what Supabase serves.** Whether a browser renders the
  report HTML or shows its source depends on the content type Supabase uses for
  HTML on your project's domain. The share dialog includes the PDF version of
  the report when one exists, which always opens.

## What leaves your computer

| When | What is sent to your Supabase project |
|---|---|
| Not connected, or signed out | Nothing (except **Test connection**, which reads public settings) |
| Signing in | Your email and password, or the code from an email, or a browser sign-in via GitHub / Google |
| Signed in | Your profile, when you edit it |
| You press **Upload** | That project folder, zipped (the Cloud projects window shows its size first) |
| You create a share link | The report files you chose |

Never sent: projects you did not upload, the Assistant's API key, your
preferences file, logs.

## Connect a project

**Account menu > Account settings** (the Account button at the right end of the menu bar):
paste the *Project URL* and the *anon* (or *publishable*) key from the Supabase
dashboard (**Project Settings > API**), then **Test connection**. The test checks
that the project answers, that it accepts the key, that the migrations are
applied, and which browser sign-ins are enabled.

Rules the app enforces:

* The URL must be `https://`. Plain `http://` is accepted only for a server on
  this computer (`127.0.0.1`, `localhost`), such as the local Supabase stack or
  the mock below, because sign-in tokens travel with every request.
* Never paste the `service_role` / secret key: it bypasses every access rule.
  The anon key is designed to be public; access rules on the server, not the
  key, protect your data.

Alternatively set both environment variables (they take precedence, and
Account settings then shows the fields read-only):

```
AEROLUME_SUPABASE_URL=https://<project-ref>.supabase.co
AEROLUME_SUPABASE_ANON_KEY=<anon or publishable key>
```

Optional settings in `account.json` (in the config folder below):

| Key | Default | Meaning |
|---|---|---|
| `max_upload_mb` | 50 | The bucket's per-file limit. Parts are 90 % of it (45 MB); set it to the bucket's `file_size_limit` if you change that |
| `max_project_mb` | 1024 | Largest project archive the app uploads (the free plan's total storage; raise it on a paid plan) |
| `redirect_ports` | `[47821, 47822, 47823, 47824]` | Loopback ports for browser sign-in; must match the project's redirect URLs |

## Signing in

Open the Account button at the right end of the menu bar (**Ctrl+Shift+U**).
It shows a person icon while no project is connected, **Sign in** when one is,
and your picture (or initials) with your display name once you are signed in.
The menu and the sign-in window name the server: *Supabase project* `<ref>`
for a hosted project, or *Server: 127.0.0.1 (local test server)* for the local
stack or the mock below.

* **Email and password.** *Create account* sends a confirmation email (unless
  the project turns confirmations off). Passwords need at least 8 characters.
* **Email me a sign-in link** signs in without a password: open the link on this
  computer, or type the code.
* **Forgot password?** sends a reset email; after the link or code, Aerolume asks
  for the new password.
* **Continue with GitHub / Google** opens your web browser (only for providers
  the project enabled). Aerolume uses PKCE and a one-time listener on
  `127.0.0.1`; your provider password never passes through Aerolume.

A session lasts until you sign out; the short-lived access token is renewed in
the background. **Profile > Sign out of all devices** ends every session.

## Cloud projects

**Account menu > Cloud projects** lists every project in your cloud and every
local folder linked to one:

| Status | Meaning | What to do |
|---|---|---|
| Up to date | Both copies have the same content | Nothing |
| Local changes | This folder changed since the last upload | **Upload** |
| Newer in cloud | Another computer uploaded; this folder is unchanged | **Update local folder** |
| Conflict | Both changed since they were last synced | Choose: replace the cloud copy with yours, or download the cloud version into a new folder |
| Cloud only | Not on this computer | **Download** |
| Not uploaded | Local only | **Upload** |
| Deleted from cloud | The linked cloud copy was deleted | **Upload** to create it again |

How it decides, and why it is safe:

* **Content, not clocks.** Each folder has a content hash: SHA-256 over every
  file's path and bytes (ignoring OS litter such as `Thumbs.db`, crash
  leftovers, and the save timestamp in `project.json`). The same content gives
  the same hash on every computer.
* **Three-way comparison.** Aerolume remembers the hash each folder had at its
  last upload or download and compares it with the folder now and the cloud now.
* **Nothing is overwritten silently.** An upload replaces the cloud copy only if
  the cloud copy is still the version this folder last saw; if another computer
  uploaded in between, you get the Conflict choice instead. Updating a local
  folder from the cloud keeps the old folder as `<name> (local copy <date>)` when
  it had changes that were never uploaded.
* **You always work in the local folder.** The cloud copy is a copy; opening a
  project never downloads anything by itself.

Upload saves the open project first, so the cloud copy is what you see.

### Upload size

Select a row to see, under the list, what an upload would send *before* you
press Upload: the packed size (exact: the archive is measured, not estimated,
and can be well below the size on disk because exported VTK files compress),
how many parts it goes up in, and the largest runs and folders, largest first.
For example, a project with ten Draft runs and one smoke export:

```
Uploading sends 103.7 MB in 3 parts of up to 45.0 MB (10 runs; 116.6 MB on disk).
Largest: Exported files 39.0 MB · Car smoke (3D run) 15.6 MB · Heated box (3D run) 14.7 MB · ...
```

The status line counts the parts while they go up. Every part is uploaded
before the cloud record changes, and the previous version is removed only after
the record points at the new one, so an interrupted upload leaves the old cloud
copy as it was (the parts already sent are removed again).

A project over `max_project_mb` is refused before anything is sent, with the
three largest items named. Exported files (`exports/`) are often the largest and
can be written again from their results at any time.

## Share links

**Account menu > Share result** (with a project open) shares one of the
project's reports (create an HTML report on the Report step first). Choose a
title and an expiry (7 days to 1 year); **Create link** uploads the report (and
its PDF, if present) and gives you the link.

* Anyone with the link can open the files until it expires; the link cannot be
  guessed and the share list cannot be browsed.
* **Revoke** deletes the files from the cloud; the link stops working at once.
* Your earlier links are listed in the same dialog to copy again or revoke.

## Security notes

* **Tokens on disk**: `account_session` in the config folder. On Windows it is
  encrypted with DPAPI for your Windows account (another user, or a copy of the
  file on another machine, cannot read it); elsewhere it is created readable
  only by you (mode 0600). It is bound to the project URL, so connecting a
  different project never reuses it.
* **Never logged**: tokens, passwords, codes and the anon key do not appear in
  logs, error messages or `repr()` output (tested).
* **Server-side rules** (`supabase/migrations`): every table has row-level
  security; you can only read and change your own rows and files; anonymous
  requests have no table access.

## Try it without a cloud project

```sh
python -m aerolume.accounts.mock --user ada@example.org:"a long password"
```

starts a local stand-in server, prints a URL and key to paste into Account
settings, and prints every "email" (code and link) in its terminal. It keeps
everything in memory and implements the same endpoints and access rules
Aerolume uses; the test suite runs against it. It is not a real backend.

## From scripts

```python
from aerolume.accounts import get_service

svc = get_service()                 # configured from Account settings / the environment
if svc.configured and not svc.signed_in:
    svc.sign_in("ada@example.org", "a long password")
folder = r"C:\Users\ada\Documents\Aerolume\Wing study.aerolume"
preview = svc.upload_preview(folder)        # local only: packed size, parts, largest runs
print(preview.size_bytes, preview.parts, [g.label for g in preview.groups[:3]])
status = svc.upload_project(folder, progress=print)   # "Uploading part 2 of 3 ..."
print(status.label)                 # "Up to date"
```

`svc.sync_overview()`, `svc.download_project(id, folder)`,
`svc.create_share([report_html], title=..., expires_days=30)` and the rest are
documented in `src/aerolume/accounts/service.py`. Errors are
`aerolume.accounts.AccountError` subclasses with a plain-language message and a
`code`.

## Files

In the config folder (`%APPDATA%\Aerolume` on Windows, `~/.config/aerolume`
elsewhere, or `AEROLUME_CONFIG_DIR`):

| File | Contents |
|---|---|
| `account.json` | Project URL, anon key, optional settings |
| `account_session` | The encrypted session (delete it to sign out locally) |
| `cloud_sync.json` | Which local folder is linked to which cloud project, and the hash at the last sync |

## When something goes wrong

| Message | Cause |
|---|---|
| "Accounts are not set up" | No project connected (the default) |
| "The project rejected the anon key" | Wrong or rotated key |
| "...missing Aerolume's tables or functions" | Migrations not applied (`supabase db push`) |
| "Cannot find <host>" / "did not answer in time" | Offline, or a mistyped URL |
| "Confirm your email address first" | The sign-up link was not opened yet; use the code |
| "Too many emails were sent" | Supabase's email rate limit; wait, or set up SMTP |
| "No reply from the browser in time" | The browser sign-in was not finished within the wait; start again |
| "...over this server's project limit" | The project is larger than `max_project_mb`; see *Upload size* |
| "The server refused a ... file" | The bucket's file limit is lower than `max_upload_mb`; set `max_upload_mb` to the bucket's limit |
| "The server's database is older than this version" | Apply the newer files in `supabase/migrations` (`supabase db push`) |
