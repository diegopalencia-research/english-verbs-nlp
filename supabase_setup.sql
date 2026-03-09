-- ─────────────────────────────────────────────────────────────────────────────
-- english-verb-patterns — Supabase PostgreSQL setup
-- Paste this entire file into Supabase SQL Editor → Run
-- ─────────────────────────────────────────────────────────────────────────────


-- ── 1. SEARCH LOGS ────────────────────────────────────────────────────────────
-- Every lookup is recorded here — powers the analytics dashboard

create table if not exists search_logs (
  id          bigserial    primary key,
  searched_at timestamptz  not null default now(),
  verb        text         not null,
  found       boolean      not null,
  verb_type   text,        -- 'Regular' | 'Irregular' | 'Participial' | 'cache:Regular' | 'api:Irregular' ...
  matched_as  text         -- 'base form' | 'simple past' | 'past participle' | null
);

-- Index for fast analytics queries
create index if not exists idx_search_logs_verb        on search_logs (verb);
create index if not exists idx_search_logs_searched_at on search_logs (searched_at desc);
create index if not exists idx_search_logs_found       on search_logs (found);


-- ── 2. PENDING VERBS ──────────────────────────────────────────────────────────
-- Verbs not in the local Excel — discovered through user searches
-- This is the "self-expanding knowledge base"

create table if not exists pending_verbs (
  id                bigserial    primary key,
  added_at          timestamptz  not null default now(),
  verb              text         not null unique,
  search_count      integer      not null default 1,
  ml_label          text,        -- 'Regular' | 'Irregular'
  ml_conf           numeric(5,2),-- 0.00 – 100.00
  predicted_ending  text         -- '/t/' | '/d/' | '/ɪd/' | null
);

-- Index for Community Trends (top searched)
create index if not exists idx_pending_verbs_search_count on pending_verbs (search_count desc);
create index if not exists idx_pending_verbs_verb         on pending_verbs (verb);


-- ── 3. ROW LEVEL SECURITY ─────────────────────────────────────────────────────
-- Enable RLS on both tables — then grant anon key INSERT/SELECT only
-- Your app uses the anon key, which is safe to expose in Streamlit

alter table search_logs  enable row level security;
alter table pending_verbs enable row level security;

-- Allow anon to INSERT (log searches) and SELECT (read analytics)
create policy "anon_insert_search_logs"
  on search_logs for insert
  to anon
  with check (true);

create policy "anon_select_search_logs"
  on search_logs for select
  to anon
  using (true);

create policy "anon_insert_pending_verbs"
  on pending_verbs for insert
  to anon
  with check (true);

create policy "anon_select_pending_verbs"
  on pending_verbs for select
  to anon
  using (true);

-- Allow anon to UPDATE search_count only
create policy "anon_update_pending_verbs"
  on pending_verbs for update
  to anon
  using (true)
  with check (true);


-- ── 4. USEFUL VIEWS ───────────────────────────────────────────────────────────
-- Pre-built queries for the Community Trends tab

-- Top 20 most searched unknown verbs
create or replace view top_searched_verbs as
  select verb, search_count, ml_label, ml_conf, predicted_ending, added_at
  from pending_verbs
  order by search_count desc
  limit 20;

-- Daily search volume (last 30 days)
create or replace view daily_search_volume as
  select
    date_trunc('day', searched_at) as day,
    count(*)                        as total_searches,
    sum(case when found then 1 else 0 end) as found_count,
    sum(case when not found then 1 else 0 end) as not_found_count
  from search_logs
  where searched_at > now() - interval '30 days'
  group by 1
  order by 1 desc;

-- Search breakdown by verb type
create or replace view search_type_breakdown as
  select
    verb_type,
    count(*) as total,
    round(count(*) * 100.0 / sum(count(*)) over (), 1) as pct
  from search_logs
  where verb_type is not null
  group by verb_type
  order by total desc;


-- ── DONE ──────────────────────────────────────────────────────────────────────
-- After running this, go to:
--   Project Settings → API → copy "URL" and "anon public" key
--   Paste both into .streamlit/secrets.toml (see secrets.toml.example)
