"""The Supabase migrations: syntax shape and the access rules they must state.

No PostgreSQL server or SQL parser is available offline, so this is a small
lexer that understands what the migrations use - ``--`` and ``/* */`` comments,
'string' literals (with ''), "quoted identifiers" and $$ / $tag$ dollar-quoted
bodies - and splits the files into statements. On top of that the tests check:

* every statement is complete (balanced parentheses, no stray quote) and starts
  with a statement keyword the migrations are allowed to use;
* every table in ``public`` has row-level security enabled, is revoked from
  ``anon``, and has a policy for each operation the app performs, each one tied
  to ``auth.uid()``; no policy is ``using (true)``;
* every ``security definer`` function pins ``search_path``;
* every storage bucket is private and has owner policies for select, insert,
  update and delete (upsert needs all of them);
* the tables, columns, CHECK constraint names, bucket ids and MIME types match
  what the client sends and what the mock server enforces, so the mock cannot
  drift from the schema without a test failing.

Applying the migrations to a real Postgres is documented in supabase/README.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from aerolume.accounts import service
from aerolume.accounts.mock import MockSupabase

ROOT = Path(__file__).resolve().parents[1] / "supabase"
MIGRATIONS = sorted((ROOT / "migrations").glob("*.sql"))

STATEMENT_KEYWORDS = {"create", "alter", "drop", "insert", "revoke", "grant", "comment"}


def split_sql(text: str) -> list[str]:
    """Statements of ``text`` with comments removed; raises on unterminated quotes / comments."""
    out, buf, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise SyntaxError("unterminated /* comment")
            i = j + 2
            continue
        if c in ("'", '"'):
            j = i + 1
            while True:
                j = text.find(c, j)
                if j < 0:
                    raise SyntaxError(f"unterminated {c} quote near {text[i:i + 40]!r}")
                if text.startswith(c * 2, j):
                    j += 2
                    continue
                break
            buf.append(text[i:j + 1])
            i = j + 1
            continue
        m = re.match(r"\$([A-Za-z_]*)\$", text[i:])
        if m:
            tag = m.group(0)
            j = text.find(tag, i + len(tag))
            if j < 0:
                raise SyntaxError(f"unterminated dollar quote {tag}")
            buf.append(text[i:j + len(tag)])
            i = j + len(tag)
            continue
        if c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if "".join(buf).strip():
        raise SyntaxError(f"statement without a terminating semicolon: {''.join(buf).strip()[:60]!r}")
    return out


def _outside_quotes(stmt: str) -> str:
    """The statement with string literals and dollar bodies blanked (for structural checks)."""
    s = re.sub(r"\$([A-Za-z_]*)\$.*?\$\1\$", "$$", stmt, flags=re.DOTALL)
    return re.sub(r"'(?:[^']|'')*'", "''", s)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", _outside_quotes(s)).strip().lower()


@pytest.fixture(scope="module")
def statements() -> list[str]:
    assert MIGRATIONS, "supabase/migrations is empty"
    stmts = []
    for path in MIGRATIONS:
        assert re.match(r"^\d{14}_[a-z0-9_]+\.sql$", path.name), f"{path.name}: use <timestamp>_<name>.sql"
        stmts += split_sql(path.read_text(encoding="utf-8"))
    return stmts


def test_lexer_handles_the_constructs_used():
    sql = ("create function f() returns int language sql as $$ select ';' ; $$; -- c ; \n"
           "comment on table t is 'it''s; fine'; /* ; */ select \"a;b\";")
    assert len(split_sql(sql)) == 3
    for bad in ("select 'open;", "create function f() as $$ body;", "select 1", "/* open"):
        with pytest.raises(SyntaxError):
            split_sql(bad)


def test_every_statement_is_well_formed(statements):
    for stmt in statements:
        body = _outside_quotes(stmt)
        assert body.count("(") == body.count(")"), f"unbalanced parentheses: {stmt[:80]}"
        first = body.split(None, 1)[0].lower()
        assert first in STATEMENT_KEYWORDS, f"unexpected statement: {stmt[:60]}"
        assert "service_role" not in body.lower(), "migrations must not grant anything to service_role"


def _tables(statements) -> dict[str, str]:
    """Column definitions per table: the CREATE TABLE list plus later ADD COLUMNs, comma-joined."""
    out = {}
    for s in statements:
        n = _norm(s)
        m = re.match(r"create table if not exists public\.(\w+) \((.*)\)$", n, re.DOTALL)
        if m:
            out[m.group(1)] = m.group(2)
        m = re.match(r"alter table public\.(\w+) add column if not exists (.*)$", n, re.DOTALL)
        if m:
            out[m.group(1)] = out[m.group(1)] + ", " + m.group(2)
    return out


def _policies(statements) -> list[dict]:
    pol = []
    for s in statements:
        n = re.sub(r"\s+", " ", s).strip()
        m = re.match(r'create policy "([^"]+)" on (\w+)\.(\w+) for (select|insert|update|delete|all) '
                     r'to (\w+)(.*)$', n, re.IGNORECASE | re.DOTALL)
        if m:
            pol.append({"name": m.group(1), "schema": m.group(2), "table": m.group(3), "cmd": m.group(4).lower(),
                        "role": m.group(5).lower(), "rest": m.group(6).lower()})
    return pol


def test_public_tables_are_owner_only(statements):
    tables = _tables(statements)
    assert set(tables) == {"profiles", "projects", "shares"}
    norm = [_norm(s) for s in statements]
    pols = _policies(statements)
    for t in tables:
        assert f"alter table public.{t} enable row level security" in norm, f"{t}: RLS not enabled"
        assert f"revoke all on table public.{t} from anon" in norm, f"{t}: anon not revoked"
        mine = [p for p in pols if p["schema"] == "public" and p["table"] == t]
        cmds = {p["cmd"] for p in mine}
        need = {"select", "insert", "update"} | ({"delete"} if t != "profiles" else set())
        assert need <= cmds, f"{t}: missing policies for {need - cmds}"
        for p in mine:
            assert p["role"] == "authenticated", f"{p['name']}: only signed-in users"
            assert "auth.uid()" in p["rest"], f"{p['name']}: not tied to the caller"
            assert "using (true)" not in p["rest"] and "check (true)" not in p["rest"]
            if p["cmd"] in ("insert", "update"):
                assert "with check" in p["rest"], f"{p['name']}: writes need WITH CHECK"
            if p["cmd"] in ("select", "update", "delete"):
                assert p["rest"].lstrip().startswith("using"), f"{p['name']}: needs USING"
    # drop-if-exists before every create policy keeps the migrations re-runnable
    drops = {re.match(r'drop policy if exists "([^"]+)"', s, re.IGNORECASE).group(1) for s in statements
             if s.lower().startswith("drop policy")}
    assert {p["name"] for p in pols} <= drops


def test_definer_functions_pin_search_path_and_grants_are_explicit(statements):
    fns = [s for s in statements if re.match(r"create or replace function", s, re.IGNORECASE)]
    assert len(fns) >= 4
    for f in fns:
        n = _norm(f)
        if "security definer" in n:
            assert "set search_path = ''" in n, f"definer function without pinned search_path: {f[:60]}"
    norm = [_norm(s) for s in statements]
    for name, sig in (("get_share", "text"), ("aerolume_schema_version", "")):
        assert f"revoke all on function public.{name}({sig}) from public" in norm
        assert f"grant execute on function public.{name}({sig}) to anon, authenticated" in norm
    raw = next(s for s in fns if "public.get_share" in s)
    body = re.sub(r"\s+", " ", raw.split("$$")[1]).lower()        # the function body itself
    assert "where s.token = share_token" in body and "s.expires_at > now()" in body
    returns = _norm(raw).split("returns table", 1)[1].split("language", 1)[0]
    assert "owner" not in returns and "links" not in returns       # never the owner id or the links


def _buckets(statements) -> dict[str, dict]:
    """Bucket settings after every migration ran (later upserts win, as in Postgres)."""
    buckets: dict[str, dict] = {}
    for ins in (s for s in statements if "storage.buckets" in s.lower() and s.lower().startswith("insert")):
        assert "on conflict (id) do update" in _norm(ins), "bucket inserts must be re-runnable upserts"
        rows = re.findall(r"\('(\w+)',\s*'(\w+)',\s*(true|false),\s*(\d+),\s*array\[([^\]]*)\]\)", ins)
        assert rows, ins[:80]
        buckets.update({r[0]: {"public": r[2] == "true", "limit": int(r[3]),
                               "mime": tuple(re.findall(r"'([^']+)'", r[4]))} for r in rows})
    return buckets


def test_storage_buckets_private_with_owner_policies(statements):
    buckets = _buckets(statements)
    assert set(buckets) == {service.PROJECTS_BUCKET, service.AVATARS_BUCKET, service.SHARES_BUCKET}
    assert not any(b["public"] for b in buckets.values()), "every bucket is private"
    assert buckets["avatars"]["limit"] == service.MAX_AVATAR_BYTES
    assert buckets["projects"]["limit"] == 50 * 1024 * 1024 == buckets["shares"]["limit"]
    # what the client uploads is what the buckets allow, and what the mock checks
    assert set(service.SHARE_TYPES.values()) == set(buckets["shares"]["mime"])
    for b, mimes in MockSupabase.BUCKETS.items():
        assert set(mimes) == set(buckets[b]["mime"]), b
    pols = [p for p in _policies(statements) if p["schema"] == "storage"]
    for b in buckets:
        mine = [p for p in pols if f"bucket_id = '{b}'" in p["rest"]]
        assert {p["cmd"] for p in mine} == {"select", "insert", "update", "delete"}, b
        for p in mine:
            assert p["table"] == "objects" and p["role"] == "authenticated" and "auth.uid()" in p["rest"]
            if b == "shares":
                assert "from public.shares" in p["rest"] and "s.owner = (select auth.uid())" in p["rest"]
            else:
                assert "(storage.foldername(name))[1] = (select auth.uid())::text" in p["rest"]


def test_schema_matches_client_and_mock(statements):
    tables = _tables(statements)

    def coldefs(t: str) -> dict[str, str]:
        parts = [x.strip() for x in re.split(r",(?![^(]*\))", tables[t]) if x.strip()]
        return {x.split()[0]: x for x in parts}
    assert set(coldefs("profiles")) >= {"id", "display_name", "affiliation", "avatar_path", "preferences",
                                        "updated_at"}
    assert set(coldefs("projects")) >= {"id", "owner", "name", "content_hash", "size_bytes", "file_count",
                                        "object_path", "app_version", "created_at", "updated_at"}
    assert set(coldefs("shares")) >= {"token", "owner", "project_id", "sim_id", "title", "files", "links",
                                      "created_at", "expires_at"}
    for t, pk in MockSupabase.TABLES.items():
        assert "primary key" in coldefs(t)[pk], f"{t}: primary key {pk}"
        assert MockSupabase.OWNER[t] in coldefs(t)
    # Postgres names a column CHECK "<table>_<column>_check"; the mock must report real names
    mock_src = (Path(__file__).resolve().parents[1] / "src/aerolume/accounts/mock.py").read_text(encoding="utf-8")
    names = re.findall(r'bad\("(\w+)_check"\)', mock_src)
    assert len(names) >= 7
    for name in names:
        table = next(t for t in tables if name.startswith(t + "_"))
        col = name[len(table) + 1:]
        assert col in coldefs(table) and "check" in coldefs(table)[col], name
    # the version the app requires is the one the migration returns
    # (the last definition wins: each schema version re-creates the function)
    raw = [s for s in statements if "aerolume_schema_version()" in s and "returns" in s.lower()][-1]
    assert re.search(rf"select\s+{service.SCHEMA_VERSION}\s*;", raw)


def test_multi_part_archives_schema(statements):
    """Schema version 2: the part_count column, the part MIME type, and the client's names for them."""
    assert (ROOT / "migrations" / service.PARTS_MIGRATION).is_file()
    cols = {x.strip().split()[0]: x.strip() for x in re.split(r",(?![^(]*\))", _tables(statements)["projects"])}
    pc = cols["part_count"]
    assert "integer" in pc and "not null" in pc and "default 1" in pc
    assert "check (part_count between 1 and 1000)" in pc
    assert service.PART_TYPE in _buckets(statements)["projects"]["mime"]
    assert service.ZIP_TYPE in _buckets(statements)["projects"]["mime"]
    # the mock enforces the same bound and reports Postgres' constraint name
    mock_src = (Path(__file__).resolve().parents[1] / "src/aerolume/accounts/mock.py").read_text(encoding="utf-8")
    assert 'bad("projects_part_count_check")' in mock_src and "1 <= parts <= 1000" in mock_src


def test_readme_lists_the_redirect_urls_the_app_uses():
    from aerolume.accounts.config import DEFAULT_REDIRECT_PORTS
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for port in DEFAULT_REDIRECT_PORTS:
        assert f"http://127.0.0.1:{port}/callback" in readme
    for migration in MIGRATIONS:
        assert migration.name in readme
