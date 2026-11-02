# Supabase backend for Aerolume accounts

Accounts are optional. Aerolume works fully offline with no account, and no
Supabase project exists until you create one. This folder holds everything a
project needs to serve Aerolume's sign-in, cloud projects and share links:

| File | What it creates |
|---|---|
| `migrations/20261018120000_aerolume_profiles.sql` | `profiles` table, sign-up trigger, `aerolume_schema_version()` |
| `migrations/20261018120100_aerolume_projects_shares.sql` | `projects` and `shares` tables, `get_share(token)` |
| `migrations/20261018120200_aerolume_storage.sql` | the `projects`, `avatars` and `shares` buckets and their policies |
| `migrations/20261030120000_aerolume_project_parts.sql` | `projects.part_count` for archives stored in parts; schema version 2 |
| `config.toml` | settings for a local stack (`supabase start`) |

The migrations are idempotent: running them twice changes nothing.

## 1. Create a project

1. Sign in at <https://supabase.com/dashboard> and choose **New project**.
2. Pick a name, a region near your users and a database password (keep it; the
   CLI asks for it when linking).
3. Wait until the project is ready. Note the **project ref** (the
   `abcdefghijklmnop` in `https://abcdefghijklmnop.supabase.co`).

## 2. Apply the migrations

With the Supabase CLI (<https://supabase.com/docs/guides/cli>), from the
repository root:

```sh
supabase login
supabase link --project-ref <project-ref>
supabase db push
```

`db push` applies every file in `supabase/migrations` that the project has not
seen yet, in file-name order.

Without the CLI: open **SQL Editor** in the dashboard and run the four files in
the order of the table above. A project set up for an earlier Aerolume needs
only the files it has not run yet (Test connection names the schema version it
found).

Check: **Table Editor** lists `profiles`, `projects` and `shares`, each marked
"RLS enabled"; **Storage** lists three private buckets.

## 3. Configure sign-in

**Authentication > URL Configuration**

* *Site URL*: `http://127.0.0.1:47821/callback`
* *Redirect URLs*: add all four loopback addresses the app may use (it takes the
  first free port). Each sign-in waits on its own one-time path,
  `/callback/<nonce>` (16 random bytes, letters, digits, `-` and `_`), so that
  another web page cannot end a pending sign-in by loading the predictable port;
  the entries are therefore globs:
  * `http://127.0.0.1:47821/callback/*`
  * `http://127.0.0.1:47822/callback/*`
  * `http://127.0.0.1:47823/callback/*`
  * `http://127.0.0.1:47824/callback/*`

  The ports can be changed with `redirect_ports` in `account.json` (see
  `docs/accounts.md`); the list here must match. In Supabase's glob syntax `*`
  matches any characters except `.` and `/`, which covers the nonce. (Written
  from Supabase's documented glob rules; no live project was available to try it.)

**Authentication > Emails** (templates). The default emails contain only a link.
The link signs the user in automatically when it is opened **on the computer
running Aerolume** while the app is waiting. To make sign-in work from a phone or
another computer, add the 6-digit code to the *Confirm signup*, *Magic link* and
*Reset password* templates, for example:

```html
<p>Your Aerolume code: <strong>{{ .Token }}</strong></p>
```

The app has a field for that code on every "Check your email" step.

**Email delivery.** Supabase's built-in email service is meant for testing: it
sends only a few messages per hour and, on new projects, only to members of your
organisation. Before inviting other users, set up custom SMTP under
**Project Settings > Authentication > SMTP Settings**.

**OAuth providers (optional).** In both cases the callback URL to give the
provider is `https://<project-ref>.supabase.co/auth/v1/callback`.

* GitHub: <https://github.com/settings/developers> > *New OAuth App*. Paste the
  client ID and a new client secret into **Authentication > Providers > GitHub**.
* Google: Google Cloud console > *APIs & Services > Credentials > Create OAuth
  client ID* (type *Web application*), with the callback above as an authorised
  redirect URI. Paste the ID and secret into **Authentication > Providers > Google**.

Aerolume shows a "Continue with ..." button only for providers the project has
enabled (it reads the project's public auth settings).

## 4. Connect Aerolume

**Project Settings > API** shows the *Project URL* and the *anon* key (or, on
newer projects, the *publishable* key; either works). Both are meant to be used
by client apps: row-level security, not the key, protects the data. Never use the
`service_role` / secret key in Aerolume.

In Aerolume: **Account button (top right) > Account settings**, paste both, then **Test
connection**. It checks that the project answers, that the key is accepted, that
the migrations are applied (schema version 2) and which providers are enabled.

Alternatively set `AEROLUME_SUPABASE_URL` and `AEROLUME_SUPABASE_ANON_KEY` in the
environment (they take precedence over Preferences).

## Limits and honest caveats

* **Upload size.** The free plan accepts files up to 50 MB, and the buckets are
  created with that limit. That caps each *file*, not a project: a project
  archive larger than 45 MiB (90 % of the limit) is stored as numbered parts,
  `<hash>.zip.part001`, `.part002`, ... in the same folder, and reassembled and
  checksum-verified on download. A typical project with several 3D results
  (measured: 54 MB packed) uploads as two parts. What does bound a project is
  the plan's total storage (1 GB on the free plan when this was written) and
  `max_project_mb` in `account.json` (default 1024). During an upload the old and
  the new version briefly exist side by side. On a paid plan you can raise the
  global limit (**Storage > Settings**) and the buckets' `file_size_limit`; set
  `max_upload_mb` in `account.json` to the same value so parts are as large as
  allowed. If the bucket limit is ever *lower* than `max_upload_mb`, the upload
  stops with a message saying so and nothing is left behind.
* **Share links** are Storage signed URLs to files in the private `shares`
  bucket; Storage stops honouring them when they expire (1-365 days, chosen
  when sharing), and revoking a share deletes its files, which ends the links
  at once. Whether a browser renders the report HTML or shows its source depends
  on the `Content-Type` Supabase serves for HTML on your project's domain; if
  it shows source, share the PDF version of the report as well (the share
  dialog includes it when one exists).
* **Email links** complete automatically only on the computer running Aerolume
  (see section 3); use the code elsewhere.

## Access rules (what the migrations enforce)

| Data | Who can read | Who can write |
|---|---|---|
| `profiles` row | its user | its user (created by a trigger at sign-up) |
| `projects` rows | their owner | their owner |
| `shares` rows | their owner; anyone holding a token via `get_share(token)` (title, file names, dates only) | their owner |
| `projects/<user id>/...` objects | their owner | their owner |
| `avatars/<user id>/...` objects | their owner | their owner |
| `shares/<token>/...` objects | the share's owner; anyone with a signed link until it expires | the share's owner |

Anonymous requests have no table rights at all (`revoke all ... from anon`).

## Local development

* **Supabase CLI stack** (needs Docker): `supabase start` in the repository
  root runs Postgres, Auth and Storage locally with `config.toml`; it prints
  `API URL: http://127.0.0.1:54321` and an anon key. `supabase db reset`
  re-applies the migrations. Emails go to the local Inbucket/Mailpit page it
  prints.
* **In-process mock** (no Docker, no SQL): `python -m aerolume.accounts.mock`
  prints a URL and key to paste into Account settings; emails (with their
  codes and links) are printed in that terminal. It implements the endpoints and
  access rules Aerolume uses, keeps everything in memory, and is what the test
  suite runs against (`tests/test_accounts*.py`).

## Removing

Delete the project in the dashboard, then clear the URL and key in Preferences >
Account. Aerolume goes back to working offline; local projects are never touched.
