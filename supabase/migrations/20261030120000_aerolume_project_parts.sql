-- Aerolume accounts, part 4 (schema version 2): project archives in parts.
--
-- Supabase's free plan refuses any single file over 50 MB, and one 3D result can
-- take 15 MB, so a normal project with a handful of 3D runs packs to more than
-- that (measured: seven Draft runs, 54 MB). Aerolume therefore stores an
-- archive larger than one part (45 MiB, 90 % of the bucket limit) as several
-- objects next to each other:
--   projects/<owner>/<id>/<content_hash>.zip.part001 ... .partNNN
-- Each part is a plain byte range of the zip, uploaded as application/octet-stream
-- because one part alone is not a zip. part_count says how many there are; an
-- archive in one piece keeps the single <content_hash>.zip name and part_count 1,
-- so rows written by schema version 1 read correctly.
--
-- The storage policies of part 3 already cover the part names (they check only
-- the first folder, the owner's id), so only the column, the bucket's MIME list
-- and the version number change here.

alter table public.projects
  add column if not exists part_count integer not null default 1
  check (part_count between 1 and 1000);

comment on column public.projects.part_count is
  'Number of Storage objects holding the archive: 1 = object_path itself, N = object_path || ''.part001'' .. ''.partNNN''.';

-- Same row as part 3, with application/octet-stream added for the parts.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('projects', 'projects', false, 52428800, array['application/zip', 'application/octet-stream'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- The desktop app checks this number in Account settings > Test connection.
create or replace function public.aerolume_schema_version()
returns integer
language sql
immutable
set search_path = ''
as $$
  select 2;
$$;

revoke all on function public.aerolume_schema_version() from public;
grant execute on function public.aerolume_schema_version() to anon, authenticated;
