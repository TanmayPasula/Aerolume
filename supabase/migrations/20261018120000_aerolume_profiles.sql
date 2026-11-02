-- Aerolume accounts, part 1: user profiles.
--
-- One row per auth user, created by a trigger at sign-up. Owner-only: a user can
-- read and change their own row and nobody else's; anonymous requests get nothing.
-- The migration is idempotent (safe to run twice).

create table if not exists public.profiles (
  id           uuid primary key references auth.users (id) on delete cascade,
  display_name text not null default '' check (char_length(display_name) <= 80),
  affiliation  text not null default '' check (char_length(affiliation) <= 120),
  -- object path in the "avatars" bucket; always inside the owner's own folder
  avatar_path  text check (avatar_path is null or avatar_path like (id::text || '/%')),
  preferences  jsonb not null default '{}'::jsonb check (jsonb_typeof(preferences) = 'object'),
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

comment on table public.profiles is 'Aerolume user profile (display name, affiliation, avatar, preferences).';

alter table public.profiles enable row level security;

drop policy if exists "profiles: owner reads" on public.profiles;
create policy "profiles: owner reads" on public.profiles
  for select to authenticated
  using ((select auth.uid()) = id);

drop policy if exists "profiles: owner inserts" on public.profiles;
create policy "profiles: owner inserts" on public.profiles
  for insert to authenticated
  with check ((select auth.uid()) = id);

drop policy if exists "profiles: owner updates" on public.profiles;
create policy "profiles: owner updates" on public.profiles
  for update to authenticated
  using ((select auth.uid()) = id)
  with check ((select auth.uid()) = id);

-- No delete policy: the row goes with the auth user (on delete cascade).

revoke all on table public.profiles from anon;
grant select, insert, update on table public.profiles to authenticated;

-- updated_at maintenance, shared by every Aerolume table
create or replace function public.aerolume_set_updated_at()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  new.updated_at := now();
  return new;
end;
$$;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
  before update on public.profiles
  for each row execute function public.aerolume_set_updated_at();

-- A profile row for every new user. security definer because the auth schema's
-- insert runs as supabase_auth_admin, which has no rights on public.profiles;
-- search_path is pinned so the function cannot be hijacked by a same-named object.
create or replace function public.aerolume_handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (id, display_name)
  values (
    new.id,
    left(coalesce(new.raw_user_meta_data ->> 'display_name',
                  new.raw_user_meta_data ->> 'full_name',
                  new.raw_user_meta_data ->> 'name',
                  ''), 80)
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists aerolume_on_auth_user_created on auth.users;
create trigger aerolume_on_auth_user_created
  after insert on auth.users
  for each row execute function public.aerolume_handle_new_user();

-- The schema version the desktop app checks in Preferences > Account > Test connection.
create or replace function public.aerolume_schema_version()
returns integer
language sql
immutable
set search_path = ''
as $$
  select 1;
$$;

revoke all on function public.aerolume_schema_version() from public;
grant execute on function public.aerolume_schema_version() to anon, authenticated;
