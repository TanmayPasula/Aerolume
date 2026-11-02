-- Aerolume accounts, part 3: Storage buckets and their access policies.
--
-- All three buckets are private. Object paths start with a folder that decides
-- who may touch them:
--   projects/<user id>/<project id>/<content hash>.zip   owner only
--   avatars/<user id>/avatar.png|jpg                     owner only
--   shares/<share token>/<file>                          the share's owner writes;
--                                                        viewers read through
--                                                        expiring signed URLs
-- Uploads with "x-upsert: true" need the select, insert and update policies.
-- The size limits match the app's defaults (account.json "max_upload_mb"); raise
-- both together on a paid plan.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('projects', 'projects', false, 52428800, array['application/zip']),
  ('avatars',  'avatars',  false, 1048576,  array['image/png', 'image/jpeg']),
  ('shares',   'shares',   false, 52428800, array['text/html', 'application/pdf', 'image/png', 'image/jpeg'])
on conflict (id) do update
  set public = excluded.public,
      file_size_limit = excluded.file_size_limit,
      allowed_mime_types = excluded.allowed_mime_types;

-- projects + avatars: the first folder of the path is the owner's user id ------

drop policy if exists "aerolume projects: owner reads" on storage.objects;
create policy "aerolume projects: owner reads" on storage.objects
  for select to authenticated
  using (bucket_id = 'projects' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume projects: owner inserts" on storage.objects;
create policy "aerolume projects: owner inserts" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'projects' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume projects: owner updates" on storage.objects;
create policy "aerolume projects: owner updates" on storage.objects
  for update to authenticated
  using (bucket_id = 'projects' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'projects' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume projects: owner deletes" on storage.objects;
create policy "aerolume projects: owner deletes" on storage.objects
  for delete to authenticated
  using (bucket_id = 'projects' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume avatars: owner reads" on storage.objects;
create policy "aerolume avatars: owner reads" on storage.objects
  for select to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume avatars: owner inserts" on storage.objects;
create policy "aerolume avatars: owner inserts" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume avatars: owner updates" on storage.objects;
create policy "aerolume avatars: owner updates" on storage.objects
  for update to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text)
  with check (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);

drop policy if exists "aerolume avatars: owner deletes" on storage.objects;
create policy "aerolume avatars: owner deletes" on storage.objects
  for delete to authenticated
  using (bucket_id = 'avatars' and (storage.foldername(name))[1] = (select auth.uid())::text);

-- shares: the first folder is a share token the caller owns -----------------

drop policy if exists "aerolume shares: owner reads" on storage.objects;
create policy "aerolume shares: owner reads" on storage.objects
  for select to authenticated
  using (
    bucket_id = 'shares'
    and exists (select 1 from public.shares s
                where s.token = (storage.foldername(name))[1] and s.owner = (select auth.uid()))
  );

drop policy if exists "aerolume shares: owner inserts" on storage.objects;
create policy "aerolume shares: owner inserts" on storage.objects
  for insert to authenticated
  with check (
    bucket_id = 'shares'
    and exists (select 1 from public.shares s
                where s.token = (storage.foldername(name))[1] and s.owner = (select auth.uid()))
  );

drop policy if exists "aerolume shares: owner updates" on storage.objects;
create policy "aerolume shares: owner updates" on storage.objects
  for update to authenticated
  using (
    bucket_id = 'shares'
    and exists (select 1 from public.shares s
                where s.token = (storage.foldername(name))[1] and s.owner = (select auth.uid()))
  )
  with check (
    bucket_id = 'shares'
    and exists (select 1 from public.shares s
                where s.token = (storage.foldername(name))[1] and s.owner = (select auth.uid()))
  );

drop policy if exists "aerolume shares: owner deletes" on storage.objects;
create policy "aerolume shares: owner deletes" on storage.objects
  for delete to authenticated
  using (
    bucket_id = 'shares'
    and exists (select 1 from public.shares s
                where s.token = (storage.foldername(name))[1] and s.owner = (select auth.uid()))
  );
