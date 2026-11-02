-- Aerolume accounts, part 2: cloud projects and share links.
--
-- projects: one row per uploaded project folder. The zip lives in the "projects"
--   bucket at <owner>/<id>/<content_hash>.zip; content_hash (SHA-256 of the files,
--   see src/aerolume/accounts/cloud.py) is what sync status and conflict detection
--   compare. The app updates a row only "where content_hash = <the hash it saw>",
--   so two computers uploading at once cannot overwrite each other silently.
-- shares: one row per read-only link. The token (32 random bytes, base64url) names
--   the share's folder in the private "shares" bucket; the links themselves are
--   Storage signed URLs that expire. The table is owner-only, so tokens cannot be
--   listed; public.get_share(token) lets whoever holds a token read its public facts.

create table if not exists public.projects (
  id           uuid primary key default gen_random_uuid(),
  owner        uuid not null default auth.uid() references auth.users (id) on delete cascade,
  name         text not null check (char_length(name) between 1 and 200),
  content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
  size_bytes   bigint not null default 0 check (size_bytes >= 0),
  file_count   integer not null default 0 check (file_count >= 0),
  object_path  text not null check (object_path like (owner::text || '/' || id::text || '/%')),
  app_version  text not null default '',
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

comment on table public.projects is 'Aerolume project folders uploaded to Storage (bucket "projects").';

create index if not exists projects_owner_updated_idx on public.projects (owner, updated_at desc);

alter table public.projects enable row level security;

drop policy if exists "projects: owner reads" on public.projects;
create policy "projects: owner reads" on public.projects
  for select to authenticated
  using ((select auth.uid()) = owner);

drop policy if exists "projects: owner inserts" on public.projects;
create policy "projects: owner inserts" on public.projects
  for insert to authenticated
  with check ((select auth.uid()) = owner);

drop policy if exists "projects: owner updates" on public.projects;
create policy "projects: owner updates" on public.projects
  for update to authenticated
  using ((select auth.uid()) = owner)
  with check ((select auth.uid()) = owner);

drop policy if exists "projects: owner deletes" on public.projects;
create policy "projects: owner deletes" on public.projects
  for delete to authenticated
  using ((select auth.uid()) = owner);

revoke all on table public.projects from anon;
grant select, insert, update, delete on table public.projects to authenticated;

drop trigger if exists projects_set_updated_at on public.projects;
create trigger projects_set_updated_at
  before update on public.projects
  for each row execute function public.aerolume_set_updated_at();


create table if not exists public.shares (
  token      text primary key check (char_length(token) >= 32 and token ~ '^[A-Za-z0-9_-]+$'),
  owner      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  project_id uuid references public.projects (id) on delete set null,
  sim_id     text,
  title      text not null default '' check (char_length(title) <= 200),
  files      jsonb not null default '[]'::jsonb check (jsonb_typeof(files) = 'array'),
  links      jsonb not null default '{}'::jsonb check (jsonb_typeof(links) = 'object'),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz
);

comment on table public.shares is 'Read-only share links for Aerolume results (files in bucket "shares").';

create index if not exists shares_owner_created_idx on public.shares (owner, created_at desc);

alter table public.shares enable row level security;

drop policy if exists "shares: owner reads" on public.shares;
create policy "shares: owner reads" on public.shares
  for select to authenticated
  using ((select auth.uid()) = owner);

-- A share may point at a cloud project only if that project is the caller's own.
drop policy if exists "shares: owner inserts" on public.shares;
create policy "shares: owner inserts" on public.shares
  for insert to authenticated
  with check (
    (select auth.uid()) = owner
    and (project_id is null
         or exists (select 1 from public.projects p
                    where p.id = project_id and p.owner = (select auth.uid())))
  );

drop policy if exists "shares: owner updates" on public.shares;
create policy "shares: owner updates" on public.shares
  for update to authenticated
  using ((select auth.uid()) = owner)
  with check ((select auth.uid()) = owner);

drop policy if exists "shares: owner deletes" on public.shares;
create policy "shares: owner deletes" on public.shares
  for delete to authenticated
  using ((select auth.uid()) = owner);

revoke all on table public.shares from anon;
grant select, insert, update, delete on table public.shares to authenticated;

drop trigger if exists shares_set_updated_at on public.shares;
create trigger shares_set_updated_at
  before update on public.shares
  for each row execute function public.aerolume_set_updated_at();

-- "Readable by token": the public facts of one live share, for whoever holds the
-- token. security definer so it can read past the owner-only policy; it returns
-- at most the one row whose token matches exactly, never a list, and never the
-- owner id or the signed links.
create or replace function public.get_share(share_token text)
returns table (token text, title text, files jsonb, created_at timestamptz, expires_at timestamptz)
language sql
stable
security definer
set search_path = ''
as $$
  select s.token, s.title, s.files, s.created_at, s.expires_at
  from public.shares as s
  where s.token = share_token
    and (s.expires_at is null or s.expires_at > now());
$$;

revoke all on function public.get_share(text) from public;
grant execute on function public.get_share(text) to anon, authenticated;
