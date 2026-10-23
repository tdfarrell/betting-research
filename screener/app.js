/* Ballwatcher Screener: phone layout (Oct 23 redesign, "A+B").
   Five tabs: Today (the strongest spots), Games, Props, My card (your bets, graded live) and Record.
   Research opens in tap-up sheets: one pick card per market, details behind tap-to-open sections.
   Bets are logged to ballwatcher.com (same database as Today's Card and The Record) through admin/app/* with a
   sign-in token kept in this app; see app/app_api.php on the site.
   Loaded before the main script: everything here runs later, from BWAPP.start(), and uses the main script's helpers
   (SPORTS, PROPCFG, gatherToday, propPick, statRows, renderPX, …). */
(function () {
  "use strict";
  const A = { on: false, tab: "today", tsp: "all", tShow: 4, today: null, gsp: null, gkey: {}, psp: null, pmk: {}, pq: "", pleans: true, pShow: 40,
    slates: {}, props: {}, card: null, cardAt: 0, rec: {}, recWho: "mine", recPer: "30", items: [], slip: [], slipMode: "straight", par: { odds: "", units: "1" },
    bld: null, timer: null, sheet: null, me: null, cfg: { model_record: true } };
  const LS = {
    get(k, d = null) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
    set(k, v) { try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) {} },
  };
  const q = s => document.querySelector(s);
  const H = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const mn = s => String(s ?? "").replace(/-(?=\d)/g, "−");                 // display minus
  const num = (x, d = 1) => x == null || isNaN(x) ? "—" : (+x).toFixed(d).replace(/\.0+$/, "");
  const sg = x => x == null || isNaN(x) ? "—" : (x > 0 ? "+" : "") + num(x);
  const pc = x => x == null ? "—" : Math.round(x * 100) + "%";
  const UP = s => String(s || "").toUpperCase();
  const ICON = {
    chev: '<svg class="ap-chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg>',
    x: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>',
    eye: '<svg viewBox="0 0 24 24" fill="none" stroke="#FFB020" stroke-width="2" aria-hidden="true"><ellipse cx="12" cy="12" rx="10" ry="6.5"/><circle cx="12" cy="12" r="2.6"/></svg>',
    today: '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
    games: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M12 5v14M3 12h18"/>',
    props: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/>',
    card: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h3"/>',
    record: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  };
  const SP_ORDER = ["nba", "nfl", "cfb", "cbb"];
  const spLabel = sp => (SPORTS[sp] || {}).label || UP(sp);

  /* ---------------------------------------------------------------- sign-in + API (ballwatcher.com/admin/app/*) */
  const API = LS.get("bw_api") || (/(^|\.)ballwatcher\.com$/.test(location.hostname) ? "/admin/app/" : "https://ballwatcher.com/admin/app/");
  let TOKEN = LS.get("bw_app_token");
  const signedIn = () => !!TOKEN;
  function setToken(t, name) { TOKEN = t || null; LS.set("bw_app_token", TOKEN); LS.set("bw_app_name", t ? (name || "") : null); if (!t) { A.me = null; A.card = null; } drawNav(); }
  async function api(action, body) {
    let r;
    try {
      r = await fetch(API + action, { method: "POST", headers: { "Content-Type": "application/json", ...(TOKEN ? { "X-BW-Token": TOKEN } : {}) }, body: JSON.stringify(body || {}), cache: "no-store" });
    } catch (e) { throw new Error("Couldn't reach ballwatcher.com. Check your connection and try again."); }
    let j = null; try { j = await r.json(); } catch (e) {}
    if (r.status === 401 && action !== "login") setToken(null);
    if (!r.ok) { const err = new Error((j && j.error) || `ballwatcher.com answered ${r.status}`); err.status = r.status; throw err; }
    return j || {};
  }

  /* ---------------------------------------------------------------- time + names */
  const etDay = iso => new Date(new Date(iso) - 4 * 3600e3).toISOString().slice(0, 10);
  const clock = iso => new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const whenTxt = iso => { if (!iso) return ""; const d = etDay(iso); return d === todayET() ? clock(iso)
    : new Date(iso).toLocaleDateString([], { weekday: "short" }) + " " + clock(iso); };
  const TWO = ["Trail Blazers", "Red Sox", "White Sox", "Blue Jays"];
  function nick(sp, full, fb) {
    if (!full) return fb || "";
    if (sp === "nba" || sp === "nfl") { for (const t of TWO) if (full.endsWith(t)) return t; return full.split(" ").pop(); }
    return fb || full;
  }
  const city = (sp, full) => sp === "nba" || sp === "nfl" ? (full || "").replace(new RegExp("\\s*" + nick(sp, full) + "$"), "") || full : full;
  const awayN = (sp, g) => nick(sp, g.away_name, g.away), homeN = (sp, g) => nick(sp, g.home_name, g.home);
  const lastTok = s => String(s || "").trim().split(/\s+/).pop();
  function pickTxt(sp, g, k) {   // "Bucks −1.5" / "Under 240.5"
    const m = g[k] || {}; if (!m.pick) return "—";
    if (k === "total") return mn(m.pick);
    const t = m.pick_side === "home" ? homeN(sp, g) : m.pick_side === "away" ? awayN(sp, g) : m.pick.replace(/\s+\S+$/, "");
    return `${t} ${mn(lastTok(m.pick))}`;
  }
  const pillTxt = (sp, g, k) => { const m = g[k] || {}; if (!m.pick) return "—"; if (k === "total") return mn(m.pick.replace(/^Over/, "O").replace(/^Under/, "U")); return mn(m.pick); };
  const gameName = g => `${g.away_name || g.away} ${g.neutral ? "vs" : "@"} ${g.home_name || g.home}`;
  const initials = n => String(n || "").split(" ").map(x => x[0]).slice(0, 2).join("");
  const MK_SHORT = { pts: "pts", reb: "reb", ast: "ast", tpm: "3PM", pra: "PRA", pr: "pts+reb", pa: "pts+ast", ra: "reb+ast", stl: "stl", blk: "blk",
    rec_yds: "rec yds", rec: "rec", rush_yds: "rush yds", rr_yds: "rush+rec", pass_yds: "pass yds", pass_td: "pass TDs", atd: "TD" };
  // how the site's auto-grader reads a stat in a pick ("Name over 16.5 points")
  const STAT_WORD = { pts: "points", reb: "rebounds", ast: "assists", tpm: "threes", pra: "pts+reb+ast", pr: "pts + reb", pa: "pts + ast", ra: "reb + ast",
    stl: "steals", blk: "blocks", pass_yds: "passing yards", pass_td: "passing tds", rush_yds: "rushing yards", rec_yds: "receiving yards", rec: "receptions",
    rr_yds: "rush + rec yards", atd: "anytime td", k: "strikeouts", h: "hits", hr: "home runs", rbi: "rbis", r: "runs" };

  /* ---------------------------------------------------------------- data */
  async function loadSlates(sp) {
    if (A.slates[sp]) return A.slates[sp];
    const S_ = SPORTS[sp], idx = await getJSON(S_.index), slates = idx.slates || [], t = todayET();
    const past = S_.done ? slates.filter(x => S_.done(x, t, slates)) : [];
    let cur = slates.filter(x => !past.includes(x)); if (!cur.length && past.length) cur = [past[past.length - 1]];
    return A.slates[sp] = { idx, cur, files: {}, def: cur.length ? S_.defaultSlate(cur) : null };
  }
  async function loadSlate(sp, key) {
    const L = await loadSlates(sp); if (!key) return null;
    if (!L.files[key]) L.files[key] = normalizeSlate(await getJSON(SPORTS[sp].slateFile(key)));
    return L.files[key];
  }
  async function loadPropsDoc(sp) {
    if (A.props[sp]) return A.props[sp];
    const L = await loadSlates(sp), key = L.def;
    if (!key || !PROPCFG[sp]) return A.props[sp] = { doc: null, L: null, key };
    let doc = null, dk = null;
    try { doc = prepModel(await getJSON(PROPCFG[sp].file(key)), sp, { nfl: await NFLCAL, nba: await NBACAL }); } catch (e) { doc = null; }
    try { dk = await getJSON(PROPCFG[sp].dk); } catch (e) { dk = null; }
    return A.props[sp] = { doc, L: dk, key };
  }
  function useProps(sp, doc, L) { state.sport = sp; state.props = doc; state.dkLines = L; }
  const scoreOf = (sp, g) => { const M = state.gradeMap || {}; const p = M[`${sp}:${g.id}:side`] || M[`${sp}:${g.id}:total`]; return p && p.score ? p.score : ""; };

  /* ---------------------------------------------------------------- shell */
  function want() {
    const v = LS.get("bw_layout");
    if (v === "app") return true; if (v === "desk") return false;
    return matchMedia("(max-width: 760px)").matches || matchMedia("(display-mode: standalone)").matches || !!navigator.standalone;
  }
  function start() {
    A.on = true; state.view = "app"; state.sport = null;
    document.body.classList.add("app");
    if (!matchMedia("(max-width: 760px)").matches) document.body.classList.add("desk-preview");
    document.title = "Ballwatcher Screener";
    try { A.slip = JSON.parse(LS.get("bw_app_slip_v1", "[]")) || []; } catch (e) { A.slip = []; }
    const app = q("#app"); app.hidden = false;
    app.innerHTML = `<div id="apView"></div><button type="button" id="apSlipBar" hidden></button><nav id="apNav" aria-label="Main"></nav>`;
    app.addEventListener("click", onClick); q("#apSheet").addEventListener("click", onClick);
    q("#apSheet").addEventListener("input", onInput); q("#apSheet").addEventListener("change", onChange);
    app.addEventListener("input", onInput);
    q("#apScrim").onclick = closeSheet;
    document.addEventListener("keydown", e => { if (e.key === "Escape" && A.sheet) closeSheet(); });
    bindGrab();
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && A.tab === "card") renderCard(true); });
    const h = location.hash.replace(/^#/, "");
    const tab = ["games", "props", "card", "record"].includes(h) ? h : "today";
    try { if (location.hash) history.replaceState(null, "", location.pathname + location.search); } catch (e) {}
    loadGrades().then(ok => { if (ok && (A.tab === "today" || A.tab === "games")) render(); });
    go(tab);
    loadConfig();
    if (signedIn()) api("me").then(m => { A.me = m; LS.set("bw_app_name", m.name); }).catch(() => {});   // also drops an expired sign-in quietly
  }
  // what the site owner has chosen to show (Settings): the model's record can be hidden from everyone but admins
  async function loadConfig() {
    try { A.cfg = await api("config"); } catch (e) { return A.cfg; }
    if (A.on && A.tab === "record") render();
    document.body.classList.toggle("no-model-record", A.cfg.model_record === false);
    return A.cfg;
  }
  function deskLink() {
    loadConfig();   // the desktop page gets a way into the phone layout (My card, bet slip, Record)
    const r = document.getElementById("installRow"); if (!r) return;
    const b = document.createElement("button"); b.type = "button"; b.className = "linkbtn"; b.style.marginLeft = "14px";
    b.textContent = "Open the app layout (My card, bet slip, Record)";
    b.onclick = () => { LS.set("bw_layout", "app"); location.reload(); };
    r.appendChild(b);
  }
  function go(tab) {
    A.tab = tab; clearInterval(A.timer); A.timer = null;
    drawNav(); window.scrollTo(0, 0); render();
  }
  function drawNav() {
    const nav = q("#apNav"); if (!nav) return;
    const tabs = [["today", "Today"], ["games", "Games"], ["props", "Props"], ["card", "My card"], ["record", "Record"]];
    const live = A.card ? (A.card.live || []).length : 0;
    nav.innerHTML = tabs.map(([k, l]) => `<button type="button" data-ap="tab" data-v="${k}" ${A.tab === k ? 'aria-current="page"' : ""}><span class="bd"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">${ICON[k]}</svg>${k === "card" && live ? `<i>${live}</i>` : ""}</span>${l}</button>`).join("");
    drawSlipBar();
  }
  function drawSlipBar() {
    const b = q("#apSlipBar"); if (!b) return;
    const n = A.slip.length;
    b.hidden = !n || !signedIn() || !!A.sheet;
    b.innerHTML = `<span>Bet slip · ${n} pick${n === 1 ? "" : "s"}</span><span>Review ›</span>`;
    b.dataset.ap = "slip";
  }
  function head(title, sub) {
    return `<header class="ap-hd"><span class="ap-brand">${ICON.eye}Ballwatcher</span><button type="button" class="ap-round" data-ap="help" aria-label="How to use the app">?</button></header>
      <section class="ap-title">${sub ? `<p>${H(sub)}</p>` : ""}<h1>${H(title)}</h1></section>`;
  }
  const todayLong = () => new Date(todayET() + "T12:00:00").toLocaleDateString([], { weekday: "long", month: "short", day: "numeric" });
  function render() {
    const v = q("#apView"); if (!v) return;
    A.items = [];
    if (A.tab === "today") return renderToday(v);
    if (A.tab === "games") return renderGames(v);
    if (A.tab === "props") return renderProps(v);
    if (A.tab === "card") return renderCard();
    if (A.tab === "record") return renderRecord();
  }
  const reg = x => (A.items.push(x), A.items.length - 1);
  function toast(msg) { const t = q("#apToast"); t.textContent = msg; t.classList.add("on"); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove("on"), 2200); }

  /* ---------------------------------------------------------------- Today */
  function segHtml(act, opts, cur) {
    return `<div class="ap-seg" role="group">${opts.map(([v, l]) => `<button type="button" data-ap="${act}" data-v="${v}" aria-pressed="${v === cur}">${H(l)}</button>`).join("")}</div>`;
  }
  async function renderToday(v) {
    v.innerHTML = head("Tonight's best spots", todayLong()) + `<div class="ap-empty">Loading today's board…</div>`;
    if (!A.today) {
      try { A.today = await gatherToday(); } catch (e) { A.today = { games: [], props: [], day: todayET() }; }
      if (A.tab !== "today") return;
    }
    const d = A.today, f = x => A.tsp === "all" || x.sp === A.tsp;
    const has = new Set([...d.games.map(x => x.sp), ...d.props.map(x => x.sp)]);
    const seg = segHtml("tsp", [["all", "All"], ...SP_ORDER.filter(sp => has.has(sp) || sp === "nba" || sp === "nfl").map(sp => [sp, spLabel(sp)])], A.tsp);
    const open = x => !x.c.started;
    const leans = d.props.filter(x => f(x) && x.cday === d.day && x.tag === "LEAN" && open(x)).sort((a, b) => b.pk.chance - a.pk.chance);
    const next = d.props.filter(x => f(x) && x.cday > d.day && x.tag === "LEAN").sort((a, b) => b.pk.chance - a.pk.chance);
    const byGame = new Map();
    for (const x of d.games.filter(f)) { const k = x.sp + x.g.id; if (!byGame.has(k)) byGame.set(k, { sp: x.sp, g: x.g, sl: x.sl }); }
    const games = [...byGame.values()].sort((a, b) => (a.g.kickoff_utc || "").localeCompare(b.g.kickoff_utc || ""));
    const list = leans.length ? leans : next;
    let html = head("Tonight's best spots", todayLong()) + seg;
    if (list.length) {
      const x = list[0];
      html += `${leans.length ? "" : `<h2 class="ap-h2">COMING UP · PROP LEANS</h2>`}${heroProp(x, 1)}`;
      if (list.length > 1) {
        html += `<h2 class="ap-h2">MORE TOP LEANS · PROPS</h2><div class="ap-list">${list.slice(1, A.tShow).map((y, i) => propRow(y, i + 2, !leans.length)).join("")}</div>`;
        if (list.length > A.tShow) html += `<button type="button" class="ap-more" data-ap="tmore">See all ${list.length} leans</button>`;
      }
    } else if (!games.length) html += `<div class="ap-empty">Nothing on the board ${A.tsp === "all" ? "today" : "for " + spLabel(A.tsp) + " today"} yet. Lines and props show up as DraftKings posts them.</div>`;
    if (games.length) html += `<h2 class="ap-h2">GAMES · OUR PICKS</h2><div class="ap-list">${games.map(gameRow).join("")}</div>`;
    v.innerHTML = html;
  }
  function heroProp(x, rank) {
    const c = x.c, pk = x.pk, m = c.markets[x.mk] || {};
    const i = reg({ t: "p", sp: x.sp, doc: x.doc, L: x.L, c, mk: x.mk });
    return `<button type="button" class="ap-card lean" data-ap="item" data-i="${i}">
      <div class="ap-top"><span class="ap-k acc">#${rank} LEAN</span><span class="ap-when">${spLabel(x.sp)} · ${H(c.team)} ${c.home ? "vs" : "@"} ${H(c.opp)} · ${H(whenTxt(c.kickoff_utc))}</span></div>
      <div class="ap-big">${H(c.name)} ${pk.side} ${num(pk.line)}</div>
      <div class="ap-sub">${H(x.label || (x.doc && x.doc.labels[x.mk]) || "")}</div>
      <div class="ap-bar win"><div><i style="width:${Math.round(pk.chance * 100)}%"></i></div><b>${pc(pk.chance)}</b></div>
      <div class="ap-small">Our model ${m.proj == null ? "—" : num(m.proj)} · DraftKings ${num(pk.line)}${pk.odds ? ` (${H(mn(pk.odds))})` : ""}</div></button>`;
  }
  function propRow(x, rank, showDay) {
    const c = x.c, pk = x.pk;
    const i = reg({ t: "p", sp: x.sp, doc: x.doc, L: x.L, c, mk: x.mk });
    return `<button type="button" class="ap-row${rank ? "" : " norank"}" data-ap="item" data-i="${i}">${rank ? `<span class="ap-rank">${rank}</span>` : ""}
      <span style="min-width:0"><span class="ap-rt">${H(c.name)} ${pk.side === "over" ? "O" : "U"} ${num(pk.line)}${!rank && x.tag === "LEAN" ? '<span class="ap-tag">LEAN</span>' : ""}</span>
      <span class="ap-rs">${H(x.label || "")} · ${H(c.team)} ${c.home ? "vs" : "@"} ${H(c.opp)} · ${H(showDay ? whenTxt(c.kickoff_utc) : clock(c.kickoff_utc))}</span></span>
      <span class="ap-pct${x.tag === "LEAN" ? "" : " acc"}">${pc(pk.chance)}</span>${ICON.chev}</button>`;
  }
  function gameRow(x) {
    const { sp, g } = x, i = reg({ t: "g", sp, g, sl: x.sl });
    const sc = scoreOf(sp, g), st = sc ? `<span class="ap-tag fin">FINAL ${H(sc)}</span>` : g.started ? '<span class="ap-tag live">STARTED</span>' : "";
    const pill = k => (g[k] || {}).pick ? `<span class="ap-pill">${H(pillTxt(sp, g, k))} <b>${pc(g[k].chance)}</b></span>` : "";
    return `<button type="button" class="ap-row norank" data-ap="item" data-i="${i}">
      <span style="min-width:0"><span class="ap-rt">${H(awayN(sp, g))} ${g.neutral ? "vs" : "@"} ${H(homeN(sp, g))}${st}</span><span class="ap-rs">${H(g.kickoff_utc ? whenTxt(g.kickoff_utc) : g.kick_et)} · ${spLabel(sp)}</span></span>
      <span class="ap-pills">${pill("side")}${pill("total")}</span>${ICON.chev}</button>`;
  }

  /* ---------------------------------------------------------------- Games */
  async function renderGames(v) {
    v.innerHTML = `${head("Games")}<div class="ap-empty">Loading…</div>`;
    const avail = [];
    await Promise.all(SP_ORDER.map(async sp => { try { const L = await loadSlates(sp); if (L.cur.length) avail.push(sp); } catch (e) {} }));
    if (A.tab !== "games") return;
    const sps = SP_ORDER.filter(sp => avail.includes(sp));
    if (!sps.length) { v.innerHTML = `${head("Games")}<div class="ap-empty">No games on the DraftKings board right now.</div>`; return; }
    if (!A.gsp || !sps.includes(A.gsp)) {
      const t = todayET(); A.gsp = sps.find(sp => A.slates[sp].cur.some(s => s.date === t || (s.first_day && s.first_day <= t && addDays(s.first_day, 6) >= t))) || sps[0];
    }
    const sp = A.gsp, L = A.slates[sp], S_ = SPORTS[sp], key = A.gkey[sp] || L.def;
    let html = `${head("Games")}${segHtml("gsp", sps.map(s => [s, spLabel(s)]), sp)}`;
    if (L.cur.length > 1) html += `<div class="ap-chips">${L.cur.map(s => { const k = S_.slateKey(s);
      return `<button type="button" data-ap="gkey" data-v="${H(k)}" aria-pressed="${k === key}">${H(sp === "nfl" ? s.label : dayLabel(s.date))}</button>`; }).join("")}</div>`;
    v.innerHTML = html + `<div class="ap-empty">Loading games…</div>`;
    let sl; try { sl = await loadSlate(sp, key); } catch (e) { sl = null; }
    if (A.tab !== "games" || A.gsp !== sp) return;
    if (!sl) { v.innerHTML = html + `<div class="ap-empty">Couldn't load these games. Pull down to refresh in a minute.</div>`; return; }
    const games = [...sl.games].sort((a, b) => (a.kickoff_utc || "").localeCompare(b.kickoff_utc || ""));
    v.innerHTML = html + `<h2 class="ap-h2">${games.length} GAME${games.length === 1 ? "" : "S"} · OUR PICKS <span>tap for research</span></h2><div class="ap-list">${games.map(g => gameRow({ sp, g, sl })).join("") || ""}</div>`;
  }

  /* ---------------------------------------------------------------- Props */
  async function renderProps(v) {
    v.innerHTML = `${head("Player props")}<div class="ap-empty">Loading…</div>`;
    const sps = SP_ORDER.filter(sp => PROPCFG[sp]);
    if (!A.psp) A.psp = (A.today && A.today.props.find(x => x.cday === todayET()) || {}).sp || "nba";
    const sp = A.psp;
    let P; try { P = await loadPropsDoc(sp); } catch (e) { P = { doc: null }; }
    if (A.tab !== "props" || A.psp !== sp) return;
    let html = `${head("Player props")}${segHtml("psp", sps.map(s => [s, spLabel(s)]), sp)}`;
    if (!P.doc) { v.innerHTML = html + `<div class="ap-empty">No ${spLabel(sp)} prop cards right now. They're built with each data pull once DraftKings posts lines.</div>`; return; }
    useProps(sp, P.doc, P.L);
    const cards = P.doc.cards || [];
    const mks = PROPCFG[sp].mk.filter(([mk]) => mk !== "atd" && cards.some(c => c.markets[mk] && dkLine(c, mk)));
    if (!mks.length) { v.innerHTML = html + `<div class="ap-empty">DraftKings hasn't posted ${spLabel(sp)} prop lines yet.</div>`; return; }
    const mk = mks.some(([k]) => k === A.pmk[sp]) ? A.pmk[sp] : mks[0][0];
    A.pmk[sp] = mk;
    const lab = P.doc.labels[mk];
    let rows = [];
    for (const c of cards) { if (!c.markets[mk] || c.started) continue; const pk = propPick(c, mk); if (pk) rows.push({ sp, doc: P.doc, L: P.L, c, mk, pk, tag: pk.tag, label: lab }); }
    const nLean = rows.filter(x => x.tag === "LEAN").length;
    if (A.pleans && nLean) rows = rows.filter(x => x.tag === "LEAN");
    const qq = A.pq.trim().toLowerCase();
    if (qq) rows = rows.filter(x => (x.c.name + " " + x.c.team + " " + x.c.opp).toLowerCase().includes(qq));
    rows.sort((a, b) => b.pk.chance - a.pk.chance);
    html += `<div class="ap-mk" role="group" aria-label="Market">${mks.map(([k, l]) => `<button type="button" data-ap="pmk" data-v="${k}" aria-pressed="${k === mk}">${H(l)}</button>`).join("")}</div>
      <input class="ap-search" type="search" id="apPq" placeholder="Search player or team" value="${H(A.pq)}" aria-label="Search player or team">
      ${nLean ? segHtml("pleans", [["1", `Leans (${nLean})`], ["0", "Every pick"]], A.pleans ? "1" : "0") : ""}
      <h2 class="ap-h2">${H(lab).toUpperCase()} · ${A.pleans && nLean ? "LEANS" : "PICKS"} <span>strongest first</span></h2>
      <div class="ap-list">${rows.slice(0, A.pShow).map(x => propRow(x, 0, true)).join("") || `<div class="ap-empty">No players match.</div>`}</div>
      ${rows.length > A.pShow ? `<button type="button" class="ap-more" data-ap="pmore">Show more (${rows.length - A.pShow})</button>` : ""}`;
    const had = document.activeElement && document.activeElement.id === "apPq";
    v.innerHTML = html;
    if (had) { const s = q("#apPq"); s.focus(); s.setSelectionRange(s.value.length, s.value.length); }
  }

  /* ---------------------------------------------------------------- sheets */
  let lockY = 0;
  function openSheet(html, kind) {
    const s = q("#apSheet");
    q("#apSheetBody").innerHTML = html;
    s.classList.add("on"); s.setAttribute("aria-hidden", "false"); q("#apScrim").classList.add("on");
    if (!A.sheet) { lockY = window.scrollY; document.body.style.top = `-${lockY}px`; document.body.classList.add("ap-lock"); s.scrollTop = 0; }
    A.sheet = kind || "sheet"; drawSlipBar();
  }
  function closeSheet() {
    const s = q("#apSheet"); if (!A.sheet) return;
    s.classList.remove("on"); s.style.transform = ""; s.setAttribute("aria-hidden", "true"); q("#apScrim").classList.remove("on");
    document.body.classList.remove("ap-lock"); document.body.style.top = ""; window.scrollTo(0, lockY);
    A.sheet = null; drawSlipBar();
    if (A.tab === "card" && A.dirty) { A.dirty = false; renderCard(true); }
  }
  function bindGrab() {
    const g = q("#apGrab"), s = q("#apSheet"); let y0 = null, dy = 0;
    g.addEventListener("touchstart", e => { y0 = e.touches[0].clientY; dy = 0; s.style.transition = "none"; }, { passive: true });
    g.addEventListener("touchmove", e => { if (y0 == null) return; dy = Math.max(0, e.touches[0].clientY - y0); s.style.transform = `translateY(${dy}px)`; }, { passive: true });
    g.addEventListener("touchend", () => { s.style.transition = ""; if (dy > 90) closeSheet(); else s.style.transform = ""; y0 = null; });
  }
  const sheetHead = (title, sub, pre, side) => `<div class="ap-sh"><div style="display:flex;gap:12px;align-items:center;min-width:0">${pre || ""}<div style="min-width:0">${side ? `<p>${side}</p>` : ""}<h2>${title}</h2>${sub ? `<p>${sub}</p>` : ""}</div></div>
    <button type="button" class="ap-x" data-ap="close" aria-label="Close">${ICON.x}</button></div>`;
  const det = (title, body, open) => `<details class="ap-det"${open ? " open" : ""}><summary>${title}</summary><div class="in">${body}</div></details>`;

  /* game sheet */
  function gameItem(sp, g, k) {
    const m = g[k] || {}; if (!m.pick) return null;
    return { key: `${sp}:${g.id}:${k}:${m.pick_side || ""}`, sport: UP(sp), market: k === "side" ? "Spread" : "Total", pick: m.pick,
      label: pickTxt(sp, g, k), sub: `${awayN(sp, g)} ${g.neutral ? "vs" : "@"} ${homeN(sp, g)} · ${k === "side" ? "spread" : "total"}`,
      game: gameName(g), starts_at: g.kickoff_utc, odds: m.odds || "", units: "1" };
  }
  const inSlip = key => A.slip.some(x => x.key === key);
  function addBtn(it, label) {
    if (!it || !signedIn()) return "";
    const on = inSlip(it.key), i = reg({ t: "add", it });
    return `<button type="button" class="ap-btn${on ? " on" : ""}" data-ap="add" data-i="${i}">${on ? "✓ In your slip" : `+ Add ${H(label || it.label)} to slip`}${it.odds ? ` · ${H(mn(it.odds))}` : ""}</button>`;
  }
  function pickCard(sp, g, k) {
    const m = g[k] || {}, v = g.v3 || {}, d = g.dk || {};
    if (!m.pick) return "";
    const side = k === "side";
    const dkTxt = side ? `${H(g.home_abbr)} ${mn(sg(d.home_spread))}` : num(d.total);
    const dkOdds = side ? (m.pick_side === "home" ? d.home_spread_odds : d.away_spread_odds) : (m.pick_side === "over" ? d.over_odds : d.under_odds);
    const mod = side ? (v.home_spread == null ? "—" : `${H(g.home_abbr)} ${mn(sg(v.home_spread))}`) : (v.total == null ? "—" : num(v.total));
    let note = "";
    if (side && v.home_spread != null) { const homeFav = v.home_spread < 0, by = Math.abs(v.home_spread);
      note = by < 0.5 ? "We have this one dead even." : `We have ${H(city(sp, homeFav ? g.home_name : g.away_name) || (homeFav ? g.home : g.away))} by about ${Math.round(by)}.`; }
    if (!side && v.total != null && d.total != null) { const df = v.total - d.total;
      note = Math.abs(df) < 0.5 ? "We're right on DraftKings' number." : `We expect about ${Math.round(Math.abs(df))} ${df < 0 ? "fewer" : "more"} points than the line.`; }
    const sc = (state.gradeMap || {})[`${sp}:${g.id}:${k}`];
    const res = sc && sc.res ? ` <span class="ap-chip ${sc.res === "W" ? "won" : sc.res === "L" ? "lost" : ""}">${sc.res === "W" ? "Won" : sc.res === "L" ? "Lost" : "Push"}</span>` : "";
    return `<section class="ap-pick"><div class="ap-k">${side ? "SPREAD" : "TOTAL"} · OUR PICK${res}</div>
      <div class="ap-line"><span>${H(pickTxt(sp, g, k))}</span><b>${pc(m.chance)}</b></div>
      <div class="ap-thin"><i style="width:${Math.round((m.chance || 0) * 100)}%"></i></div>
      <div class="ap-two"><div><i>DraftKings${dkOdds ? ` · ${H(mn(dkOdds))}` : ""}</i><b>${dkTxt}</b></div><div><i>Our model</i><b>${mod}</b></div></div>
      ${note ? `<p>${note}</p>` : ""}${g.started ? "" : addBtn(gameItem(sp, g, k))}</section>`;
  }
  function openGame(sp, g, sl) {
    state.sport = sp; state.slate = sl;
    const T = sl.teams || {}, Ta = T[g.away] || {}, Th = T[g.home] || {};
    const rec = t => [t.record, t.ats && t.ats !== "—" ? `ATS ${t.ats}` : ""].filter(Boolean).join(" · ");
    const resList = t => (t.results || []).slice().reverse().map(r => `<span class="${r.res}">${dayLabel(r.date).replace(/^\w+, /, "")} ${r.site === "A" ? "@" : "vs"} ${H(r.opp)} ${r.res} ${H(r.score)}${r.ats ? ` · ATS ${r.ats}` : ""}${r.ou ? ` · ${r.ou === "O" ? "over" : r.ou === "U" ? "under" : "push"}` : ""}</span>`).join("") || "<span>No games yet</span>";
    const ou = t => { const r = (t.results || []).filter(x => x.ou); const o = r.filter(x => x.ou === "O").length, u = r.filter(x => x.ou === "U").length; return o + u ? `O/U ${o}-${u}` : ""; };
    const restTxt = x => x == null ? "" : x === 1 ? "back-to-back" : `${x - 1} day${x === 2 ? "" : "s"} off`;
    const injList = t => (t.injuries || []).length ? t.injuries.map(i => `<li><b>${H(i.name)}</b> · ${H(i.status)}</li>`).join("") : `<li>None listed</li>`;
    let inj = "";
    if (sp === "nba") inj = `<p style="margin:0 0 6px"><b>${H(g.away_abbr)}</b>${g.rest && g.rest.away ? ` · ${restTxt(g.rest.away)}` : ""}</p><ul>${injList(Ta)}</ul>
      <p style="margin:12px 0 6px"><b>${H(g.home_abbr)}</b>${g.rest && g.rest.home ? ` · ${restTxt(g.rest.home)}` : ""}</p><ul>${injList(Th)}</ul>`;
    const notes = [...(g.flags || []), ...(g.notes || [])];
    if (notes.length) inj += `<ul style="margin-top:${inj ? 12 : 0}px">${notes.map(n => `<li>${H(n)}</li>`).join("")}</ul>`;
    if (sp === "cbb") inj += `<p style="margin:${inj ? 12 : 0}px 0 0;color:var(--ap-muted)">College injury reports aren't in the data. Check news before tip.</p>`;
    // the full research drawer from the desktop page, minus what the cards above already show
    const t = document.createElement("template");
    try { t.innerHTML = predDrawer(SPORTS[sp].drawer)(g, T); } catch (e) { t.innerHTML = ""; }
    t.content.querySelectorAll(".panel").forEach(p => { const h = (p.querySelector("h4") || {}).textContent || "";
      if (/^Line movement|^Injury report|results$/.test(h)) p.remove(); });
    t.content.querySelectorAll(".addrow,.addb").forEach(x => x.remove());
    const wx = sp === "nfl" ? nflWx(g) : g.weather ? `${g.weather}${g.temp_f != null ? ` ${g.temp_f}°` : ""}` : "";
    const sub = Ta.record || Th.record ? `${H(g.away_abbr)} ${H(Ta.record || "—")} · ${H(g.home_abbr)} ${H(Th.record || "—")}` : "";
    const html = sheetHead(`${H(awayN(sp, g))} <span style="color:var(--ap-muted);font-weight:600">${g.neutral ? "vs" : "at"}</span> ${H(homeN(sp, g))}`, sub, "",
        `${spLabel(sp)} · ${H(g.kick_et || whenTxt(g.kickoff_utc))}${g.venue ? ` · ${H(g.venue)}` : ""}${wx ? ` · ${H(wx)}` : ""}${scoreOf(sp, g) ? ` · Final ${H(scoreOf(sp, g))}` : ""}`)
      + pickCard(sp, g, "side") + pickCard(sp, g, "total")
      + det("How they match up", `<div class="gs"><div class="gs-hd"><span>${H(g.away_abbr)}</span><span>Matchup</span><span>${H(g.home_abbr)}</span></div>${statRows(g, T)}</div>`, true)
      + (inj ? det(sp === "nba" ? "Injuries &amp; rest" : "Notes", inj) : "")
      + det("Recent results &amp; trends", `<p style="margin:0 0 6px"><b>${H(Ta.name || g.away_name || g.away)}</b> · ${H([rec(Ta), ou(Ta)].filter(Boolean).join(" · ") || "no games yet")}</p><div class="gf-row">${formRow(Ta)}</div><div class="res" style="margin-top:8px">${resList(Ta)}</div>
         <p style="margin:14px 0 6px"><b>${H(Th.name || g.home_name || g.home)}</b> · ${H([rec(Th), ou(Th)].filter(Boolean).join(" · ") || "no games yet")}</p><div class="gf-row">${formRow(Th)}</div><div class="res" style="margin-top:8px">${resList(Th)}</div>`)
      + det("Line movement", `<div id="apLm">Loading…</div>`)
      + (t.innerHTML.trim() ? det("Full research", t.innerHTML) : "");
    openSheet(html, "game");
    A.reopen = () => openGame(sp, g, sl);
    loadHistory().then(Hs => { const el = q("#apLm"); if (!el) return; const pts = Hs[g.id] || [];
      el.innerHTML = lineChart(pts, "home_spread", `${g.home_abbr} spread`) + lineChart(pts, "total", "Total"); });
  }

  /* player sheet */
  function propItem(sp, c, mk, pk) {
    const lab = (state.props.labels || {})[mk] || mk;
    return { key: `${sp}:p:${c.id}:${mk}:${pk.side}:${pk.line}`, sport: UP(sp), market: "Player prop", pick: `${c.name} ${pk.side} ${pk.line} ${STAT_WORD[mk] || lab.toLowerCase()}`,
      label: `${c.name} ${pk.side === "over" ? "O" : "U"} ${num(pk.line)} ${MK_SHORT[mk] || lab}`, sub: `${c.team} ${c.home ? "vs" : "@"} ${c.opp} · ${lab}`,
      game: c.home ? `${c.opp} @ ${c.team}` : `${c.team} @ ${c.opp}`, starts_at: c.kickoff_utc, odds: pk.odds || "", units: "1" };
  }
  function openPlayer(sp, doc, L, c, mk) {
    useProps(sp, doc, L);
    const lab = doc.labels || {};
    const mks = Object.keys(c.markets).filter(k => k !== "atd" && dkLine(c, k));
    const all = mks.length ? mks : Object.keys(c.markets);
    if (!c.markets[mk]) mk = all[0];
    state.pf.mk = mk;
    const m = c.markets[mk], pk = propPick(c, mk), d = mk !== "atd" && dkLine(c, mk);
    const hs = c.headshot ? `<img class="ap-hs" src="${H(c.headshot)}" alt="" onerror="this.outerHTML='<div class=&quot;ap-hs&quot;>${H(initials(c.name))}</div>'">` : `<div class="ap-hs">${H(initials(c.name))}</div>`;
    const role = sp === "nba" ? (c.starter ? "starter" : "bench") : c.role;
    let html = sheetHead(H(c.name), `${H(c.team)} ${c.home ? "vs" : "@"} ${H(c.opp)} · ${H(whenTxt(c.kickoff_utc))}${role ? ` · ${H(role)}` : ""}${c.status ? ` · <span style="color:var(--ap-loss)">${H(c.status)}</span>` : ""}`, hs);
    html += `<div class="ap-mk" role="group" aria-label="Market">${all.map(k => `<button type="button" data-ap="pxmk" data-v="${k}" aria-pressed="${k === mk}">${H(lab[k] || k)}</button>`).join("")}</div>`;
    const unit = (MK_SHORT[mk] || "").replace("3PM", "threes");
    if (pk) {
      const lean = pk.tag === "LEAN";
      html += `<section class="ap-pick${lean ? " lean" : ""}"><div class="ap-k${lean ? " acc" : ""}">${lean ? "LEAN" : "OUR PICK"}</div>
        <div class="ap-line"><span>${pk.side === "over" ? "Over" : "Under"} ${num(pk.line)}</span><b>${pc(pk.chance)}</b></div>
        <div class="ap-thin"><i style="width:${Math.round(pk.chance * 100)}%"></i></div>
        <div class="ap-two"><div><i>DraftKings${pk.odds ? ` · ${H(mn(pk.odds))}` : ""}</i><b>${num(pk.line)} ${H(unit)}</b></div><div><i>Our model</i><b>${m.proj == null ? "—" : num(m.proj)} ${H(unit)}</b></div></div>
        ${c.started ? "" : addBtn(propItem(sp, c, mk, pk), `${pk.side === "over" ? "Over" : "Under"} ${num(pk.line)}`)}</section>`;
    } else {
      html += `<section class="ap-pick"><div class="ap-k">${H(lab[mk] || mk)}</div><p style="margin-top:6px">${d ? "No model chance for this line." : "No DraftKings line for this market yet."}${m && m.proj != null ? ` Our model projects ${num(m.proj)}.` : ""}</p></section>`;
    }
    const line = d ? d.line : pxLineFor(c, mk), minHit = Math.floor(line) + 1;
    if (hasLog(c) && mk !== "atd") {
      const rows = glog(c), k = gstat(mk);
      const tile = (l, r) => { const [h, n] = hitAt(r, mk, minHit); return `<div><i>${H(l)}</i><b>${n ? `${h}/${n}` : "—"}</b></div>`; };
      html += `<h2 class="ap-h2">HOW OFTEN ${c.name.split(" ").length ? "HE'S" : ""} GONE OVER ${num(line)}</h2><div class="ap-tiles">${tile("L5", winOf(rows, "l5", c))}${tile("L10", winOf(rows, "l10", c))}${tile("L20", winOf(rows, "l20", c))}${tile(`vs ${c.opp}`, winOf(rows, "h2h", c))}</div>`;
      const last = rows.slice(-10);
      if (last.length) {
        const vals = last.map(x => +x[k] || 0), top = Math.max(line * 1.3, ...vals, 1);
        html += `<h2 class="ap-h2">LAST ${last.length} GAMES</h2><div class="ap-bars"><div class="ap-plot" style="grid-template-columns:repeat(${last.length},minmax(0,1fr))">${last.map((x, i) =>
            `<div class="${vals[i] >= minHit ? "o" : ""}" style="height:${Math.max(2, vals[i] / top * 100).toFixed(1)}%" title="${H(PROPCFG[sp].gtitle(x))} ${x.home ? "vs" : "@"} ${H(x.opp)}: ${vals[i]}"><span>${vals[i]}</span></div>`).join("")}
          <div class="ln" style="bottom:${(line / top * 100).toFixed(1)}%"></div></div>
          <div class="ap-labs" style="grid-template-columns:repeat(${last.length},minmax(0,1fr))">${last.map(x => `<span>${x.home ? "" : "@"}${H(x.opp)}</span>`).join("")}</div>
          <p class="ap-cap">Dashed line = ${d ? "DraftKings " : ""}${num(line)}. Orange bars went over.</p></div>`;
      }
    }
    // what the model sees, in plain words
    const md = c.model || {}, u = c.usage || {}, why = [];
    if (m && m.proj != null) why.push(`Projects ${num(m.proj)} ${H(unit || (lab[mk] || "").toLowerCase())} (middle of its range)${d ? ` against DraftKings' ${num(d.line)}` : ""}.`);
    if (sp === "nba" && u.min_pg != null) why.push(`${c.starter ? "Starting" : "Coming off the bench"}, averaging ${num(u.min_pg)} minutes${c.games ? ` over ${c.games} game${c.games === 1 ? "" : "s"} this season` : ""}.`);
    if (c.rest === 1) why.push("Second night of a back-to-back.");
    if (m && m.def_avg != null) why.push(`${H(c.opp)} allows ${num(m.def_avg)} to ${H(c.role || c.pos)}s${m.dvp ? ` (${m.dvp.rank > (m.dvp.of || 30) * 2 / 3 ? "one of the softest" : m.dvp.rank <= (m.dvp.of || 30) / 3 ? "one of the toughest" : "about average"} matchups)` : ""}.`);
    if (sp === "nba" && c.games < 5) why.push(`Only ${c.games} game${c.games === 1 ? "" : "s"} this season, so last season carries a lot of the projection.`);
    if (c.implied != null) why.push(`${H(c.team)} are projected for ${num(c.implied)} points.`);
    if (why.length) html += det(pk ? `Why the model likes the ${pk.side}` : "What the model sees", `<ul>${why.map(w => `<li>${w}</li>`).join("")}</ul>`, true);
    if (hasLog(c)) html += det("Trends explorer · any line, any split", `<div class="px"><div id="px"></div></div>`);
    // game log
    const S0 = doc.season, rows = hasLog(c) ? glog(c).filter(x => x.season === S0).slice().reverse() : [];
    if (sp === "nba" && rows.length) html += det("Game log · this season", `<div class="rwrap"><table class="rtable"><thead><tr><th>Date</th><th>Opp</th><th class="num">Min</th><th class="num">Pts</th><th class="num">Reb</th><th class="num">Ast</th><th class="num">3PM</th></tr></thead><tbody>${rows.map(x => `<tr><td>${H(PROPCFG.nba.glabel(x))}</td><td>${x.home ? "" : "@"}${H(x.opp)}</td><td class="num">${x.min}</td><td class="num">${x.pts}</td><td class="num">${x.reb}</td><td class="num">${x.ast}</td><td class="num">${x.tpm}</td></tr>`).join("")}</tbody></table></div>`);
    else if (c.log && c.log.length) { const cols = LOG_COLS[c.pos] || LOG_COLS.WR;
      html += det("Game log · this season", `<div class="rwrap"><table class="rtable"><thead><tr><th>Wk</th><th>Opp</th>${cols.map(k => `<th class="num">${k[0]}</th>`).join("")}</tr></thead><tbody>${c.log.map(x => `<tr><td>${x.wk}</td><td>${H(x.opp)}</td>${cols.map(k => `<td class="num">${H(k[1](x))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`); }
    html += det("Every market · hit-rate tiers", Object.keys(c.markets).map(k => `<div style="margin:6px 0 14px"><div style="font-weight:600;margin-bottom:4px">${H(lab[k] || k)}</div>${tierTable(c, k)}</div>`).join(""));
    openSheet(html, "player");
    A.pctx = { sp, doc, L, c };
    const on = q('#apSheet [data-ap="pxmk"][aria-pressed="true"]'); if (on) on.parentElement.scrollLeft = on.offsetLeft - 20;
    if (hasLog(c)) { if (!px || px.pid !== c.id || px.gid !== c.game_id || px.mk !== mk) pxReset(c, mk); renderPX(c); bindPX(c); }
  }

  /* ---------------------------------------------------------------- bet slip */
  function saveSlip() { LS.set("bw_app_slip_v1", JSON.stringify(A.slip)); }
  function toggleSlip(it) {
    const i = A.slip.findIndex(x => x.key === it.key);
    if (i >= 0) { A.slip.splice(i, 1); toast("Removed from your slip"); } else { A.slip.push({ ...it }); toast(`Added · ${A.slip.length} in your slip`); }
    saveSlip(); drawSlipBar();
  }
  const amDec = o => { const n = parseFloat(String(o || "").replace("−", "-").replace("+", "")); if (!isFinite(n) || Math.abs(n) < 100) return null; return n > 0 ? 1 + n / 100 : 1 + 100 / -n; };
  const decAm = dd => !dd || dd <= 1 ? "" : dd >= 2 ? "+" + Math.round((dd - 1) * 100) : String(Math.round(-100 / (dd - 1)));
  const toWin = (u, o) => { const dd = amDec(o), x = parseFloat(u); return dd && x > 0 ? x * (dd - 1) : null; };
  const uTxt = x => x == null ? "—" : (Math.round(x * 100) / 100).toString() + "u";
  function estParlay() { const ds = A.slip.map(x => amDec(x.odds)); return ds.every(Boolean) && ds.length > 1 ? decAm(ds.reduce((a, b) => a * b, 1)) : ""; }
  function slipHtml(msg) {
    const n = A.slip.length;
    let html = sheetHead("Bet slip", n ? "" : "Add picks with the + buttons, or build one from scratch.");
    if (!signedIn()) return html + signInForm("Sign in to log bets to your card.");
    if (n < 2 && A.slipMode === "parlay") A.slipMode = "straight";
    if (n) html += `<div style="margin-top:4px">${segHtml("smode", [["straight", `Straight bets (${n})`], ["parlay", "Parlay"]], A.slipMode)}</div>`;
    html += `<div class="ap-form">`;
    if (A.slipMode === "parlay") {
      if (!A.par.odds) A.par.odds = estParlay();
      html += A.slip.map((x, i) => `<div class="ap-leg2"><span><b>${H(x.label)}</b><small>${H(x.sub || x.game)}</small></span><input class="ap-in" inputmode="text" data-f="odds" data-i="${i}" value="${H(x.odds)}" placeholder="odds" aria-label="Leg odds (optional)"><button type="button" data-ap="srm" data-i="${i}" aria-label="Remove">✕</button></div>`).join("");
      html += `<div class="ap-g2" style="margin-top:6px"><label class="ap-fl"><span>DraftKings parlay odds</span><input class="ap-in" id="apPo" data-f="podds" value="${H(A.par.odds)}" placeholder="+264" inputmode="text"></label>
        <label class="ap-fl"><span>Units</span><input class="ap-in" id="apPu" data-f="punits" value="${H(A.par.units)}" inputmode="decimal"></label></div>
        <div class="ap-tw"><span>To win</span><b id="apTw">${uTxt(toWin(A.par.units, A.par.odds))}</b></div>
        <p class="ap-cap">${estParlay() ? `The legs multiply out to about ${H(mn(estParlay()))}. ` : ""}Type in DraftKings' actual parlay price from your bet slip.</p>`;
    } else if (n) {
      html += A.slip.map((x, i) => `<div class="ap-leg2 u"><span><b>${H(x.label)}</b><small>${H(x.sub || x.game)}</small></span><input class="ap-in" data-f="odds" data-i="${i}" value="${H(x.odds)}" placeholder="odds" aria-label="DraftKings odds"><input class="ap-in" data-f="units" data-i="${i}" value="${H(x.units || "1")}" inputmode="decimal" aria-label="Units"><button type="button" data-ap="srm" data-i="${i}" aria-label="Remove">✕</button></div>`).join("");
      html += `<div class="ap-tw"><span>Odds · units</span><span></span></div><div class="ap-tw"><span>Total to win</span><b id="apTw">${uTxt(A.slip.reduce((a, x) => a + (toWin(x.units || 1, x.odds) || 0), 0))}</b></div>`;
    }
    if (msg) html += msg;
    if (n) html += `<button type="button" class="ap-btn main" data-ap="log" style="margin-top:18px">${A.slipMode === "parlay" ? `Log ${new Set(A.slip.map(x => x.game)).size === 1 ? "same game parlay" : "parlay"} · ${n} legs` : `Log ${n} bet${n === 1 ? "" : "s"}`}</button>`;
    html += `<button type="button" class="ap-btn" data-ap="build">+ Build a bet from scratch</button>${n ? `<button type="button" class="ap-btn ghost" data-ap="sclear">Clear slip</button>` : ""}</div>`;
    return html;
  }
  function openSlip(msg) { openSheet(slipHtml(msg), "slip"); A.reopen = () => openSlip(); }
  async function logSlip() {
    const n = A.slip.length; if (!n) return;
    let body;
    if (A.slipMode === "parlay") {
      const starts = A.slip.map(x => x.starts_at).filter(Boolean).sort()[0];
      body = { kind: new Set(A.slip.map(x => x.game)).size === 1 ? "sgp" : "parlay", dk_odds: A.par.odds, units: A.par.units, starts_at: starts,
        legs: A.slip.map(x => ({ sport: x.sport, pick: x.pick, game: x.game, odds: x.odds, starts_at: x.starts_at })) };
    } else {
      body = { bets: A.slip.map(x => ({ kind: "straight", sport: x.sport, market: x.market, pick: x.pick, game: x.game, starts_at: x.starts_at, odds: x.odds, units: x.units || "1" })) };
    }
    const btn = q('[data-ap="log"]'); if (btn) { btn.disabled = true; btn.textContent = "Logging…"; }
    try {
      const r = await api("bet", body);
      A.slip = []; A.par = { odds: "", units: "1" }; A.slipMode = "straight"; saveSlip();
      closeSheet(); toast(`Logged ${r.ids.length > 1 ? r.ids.length + " bets" : "it"} · on your card`);
      A.card = null; go("card");
    } catch (e) {
      openSlip(`<div class="ap-err">${H(e.message)}</div>`);
    }
  }

  /* build a bet from scratch: any game, any line (alt lines too), any player */
  const STATS = {
    nba: [["pts", "Points"], ["reb", "Rebounds"], ["ast", "Assists"], ["tpm", "Threes"], ["pra", "Pts + Reb + Ast"], ["pr", "Pts + Reb"], ["pa", "Pts + Ast"], ["ra", "Reb + Ast"], ["stl", "Steals"], ["blk", "Blocks"]],
    nfl: [["pass_yds", "Passing yards"], ["pass_td", "Passing TDs"], ["rush_yds", "Rushing yards"], ["rec_yds", "Receiving yards"], ["rec", "Receptions"], ["rr_yds", "Rush + rec yards"], ["atd", "Anytime TD"]],
    mlb: [["k", "Strikeouts"], ["h", "Hits"], ["hr", "Home runs"], ["rbi", "RBIs"], ["r", "Runs"]],
  };
  STATS.cfb = STATS.nfl; STATS.cbb = STATS.nba.slice(0, 5);
  const MKTS = [["spread", "Spread"], ["ml", "Moneyline"], ["total", "Total"], ["tt", "Team total"], ["prop", "Player prop"]];
  function newBld(sp) { return { sp: sp || A.gsp || "nba", gid: "", mkt: "spread", side: "home", ou: "over", line: "", odds: "", player: "", stat: "", custom: "", when: "", team: "", touched: false }; }
  function bldGames(b) {
    const L = A.slates[b.sp]; if (!L) return null;
    const out = [];
    for (const s of L.cur) { const sl = L.files[SPORTS[b.sp].slateKey(s)]; if (sl) for (const g of sl.games) out.push(g); }
    return out.sort((a, c) => (a.kickoff_utc || "").localeCompare(c.kickoff_utc || ""));
  }
  async function ensureBldData(b) {
    if (!SPORTS[b.sp]) return;
    try { const L = await loadSlates(b.sp); await Promise.all(L.cur.slice(0, 4).map(s => loadSlate(b.sp, SPORTS[b.sp].slateKey(s)).catch(() => null))); } catch (e) {}
    if (PROPCFG[b.sp]) { try { await loadPropsDoc(b.sp); } catch (e) {} }
  }
  function bldGame(b) { return b.gid && b.gid !== "other" ? (bldGames(b) || []).find(g => g.id === b.gid) : null; }
  function bldCard(b, g) {
    const P = A.props[b.sp]; if (!P || !P.doc || !b.player) return null;
    const nm = b.player.trim().toLowerCase();
    return (P.doc.cards || []).find(c => c.name.toLowerCase() === nm && (!g || c.game_id === g.id)) || null;
  }
  function bldPrefill(b) {   // DraftKings' line and price for what's chosen, when we have them
    const g = bldGame(b), d = (g && g.dk) || {};
    let line = "", odds = "";
    if (b.mkt === "spread" && g) { line = b.side === "home" ? d.home_spread : d.home_spread == null ? null : -d.home_spread; odds = b.side === "home" ? d.home_spread_odds : d.away_spread_odds; }
    if (b.mkt === "ml" && g) odds = b.side === "home" ? d.home_ml : d.away_ml;
    if (b.mkt === "total" && g) { line = d.total; odds = b.ou === "over" ? d.over_odds : d.under_odds; }
    if (b.mkt === "prop") { const c = bldCard(b, g), P = A.props[b.sp];
      if (c && P && b.stat && b.stat !== "atd") { useProps(b.sp, P.doc, P.L); const x = dkLine(c, b.stat); if (x) { line = x.line; odds = b.ou === "over" ? x.over : x.under; } } }
    b.line = line == null || line === "" ? "" : String(line); b.odds = odds ? String(odds) : "";
    b.dkLine = b.line; b.dkOdds = b.odds;
  }
  function bldTeams(b) {
    const g = bldGame(b);
    if (g) return [["away", awayN(b.sp, g), g.away_abbr], ["home", homeN(b.sp, g), g.home_abbr]];
    return null;
  }
  function bldPick(b) {   // the pick written the way the site's auto-grader reads it, plus a display label
    const g = bldGame(b), teams = bldTeams(b);
    const tm = teams ? (teams.find(t => t[0] === b.side) || teams[1]) : null;
    const team = tm ? tm[2] : b.team.trim(), tlab = tm ? tm[1] : b.team.trim();
    const ln = parseFloat(b.line), lnTxt = isFinite(ln) ? (ln > 0 ? "+" : "") + num(ln, 1) : "";
    const game = g ? gameName(g) : b.custom.trim();
    let pick = "", label = "", market = "";
    if (b.mkt === "spread") { if (!team || !isFinite(ln)) return null; pick = ln === 0 ? `${team} pk` : `${team} ${lnTxt}`; label = `${tlab} ${mn(ln === 0 ? "pk" : lnTxt)}`; market = "Spread"; }
    else if (b.mkt === "ml") { if (!team) return null; pick = `${team} ML`; label = `${tlab} moneyline`; market = "Moneyline"; }
    else if (b.mkt === "total") { if (!isFinite(ln) || !game) return null; pick = `${b.ou === "over" ? "Over" : "Under"} ${num(ln, 1)}`; label = pick; market = "Total"; }
    else if (b.mkt === "tt") { if (!team || !isFinite(ln)) return null; pick = `${team} ${b.ou} ${num(ln, 1)}`; label = `${tlab} team total ${b.ou} ${num(ln, 1)}`; market = "Team total"; }
    else if (b.mkt === "prop") {
      const nm = b.player.trim(), st = b.stat || (STATS[b.sp] || STATS.nba)[0][0]; if (!nm) return null;
      if (st === "atd") { pick = `${nm} anytime td`; label = `${nm} anytime TD`; }
      else { if (!isFinite(ln)) return null;
        pick = b.ou === "atleast" ? `${nm} ${Math.round(ln)}+ ${STAT_WORD[st]}` : `${nm} ${b.ou} ${num(ln, 1)} ${STAT_WORD[st]}`;
        label = b.ou === "atleast" ? `${nm} ${Math.round(ln)}+ ${(STATS[b.sp] || STATS.nba).find(s => s[0] === st)[1].toLowerCase()}` : `${nm} ${b.ou === "over" ? "O" : "U"} ${num(ln, 1)} ${MK_SHORT[st] || STAT_WORD[st]}`; }
      market = "Player prop";
    }
    const starts = g ? g.kickoff_utc : (b.when ? new Date(b.when).toISOString() : "");
    return { key: `custom:${b.sp}:${pick}:${game}`, sport: UP(b.sp), market, pick, label, sub: game ? `${game.replace(/^(.*) @ (.*)$/, (s, a, h) => g ? `${awayN(b.sp, g)} @ ${homeN(b.sp, g)}` : s)} · ${market.toLowerCase()}` : market,
      game, starts_at: starts, odds: b.odds, units: "1" };
  }
  function bldHtml(msg) {
    const b = A.bld, games = bldGames(b), g = bldGame(b), teams = bldTeams(b);
    const sports = [["nba", "NBA"], ["nfl", "NFL"], ["cfb", "CFB"], ["cbb", "CBB"], ["mlb", "MLB"]];
    let html = sheetHead("Build a bet", "Any game, any line: alt lines, team totals, props.");
    html += `<div class="ap-form">
      <label class="ap-fl"><span>Sport</span><select class="ap-sel" data-f="b.sp">${sports.map(([k, l]) => `<option value="${k}" ${b.sp === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>
      <label class="ap-fl"><span>Game</span><select class="ap-sel" data-f="b.gid"><option value="">${games ? "Choose a game" : "Loading games…"}</option>${(games || []).map(x => `<option value="${H(x.id)}" ${b.gid === x.id ? "selected" : ""}>${H(awayN(b.sp, x))} ${x.neutral ? "vs" : "@"} ${H(homeN(b.sp, x))} · ${H(x.kick_et || whenTxt(x.kickoff_utc))}</option>`).join("")}<option value="other" ${b.gid === "other" ? "selected" : ""}>Another game (type it in)</option></select></label>`;
    if (b.gid === "other") html += `<div class="ap-g2"><label class="ap-fl"><span>Game</span><input class="ap-in txt" data-f="b.custom" value="${H(b.custom)}" placeholder="Lakers @ Celtics"></label>
      <label class="ap-fl"><span>Starts</span><input class="ap-in txt" type="datetime-local" data-f="b.when" value="${H(b.when)}"></label></div>`;
    if (b.gid) {
      html += `<div class="ap-mk" style="margin:14px -20px 0" role="group" aria-label="Bet type">${MKTS.map(([k, l]) => `<button type="button" data-ap="bmkt" data-v="${k}" aria-pressed="${b.mkt === k}">${l}</button>`).join("")}</div>`;
      const sideSeg = teams ? `<div style="margin:12px -20px 0">${segHtml("bside", teams.map(t => [t[0], t[1]]), b.side)}</div>`
        : `<label class="ap-fl"><span>Team</span><input class="ap-in txt" data-f="b.team" value="${H(b.team)}" placeholder="Team name or abbreviation"></label>`;
      const ouSeg = (opts) => `<div style="margin:12px -20px 0">${segHtml("bou", opts, b.ou)}</div>`;
      const stepper = (lab, step) => `<label class="ap-fl"><span>${lab}</span><div class="ap-step"><button type="button" data-ap="bstep" data-v="-${step}" aria-label="Lower">−</button><input class="ap-in" data-f="b.line" value="${H(b.line)}" inputmode="decimal" aria-label="${lab}"><button type="button" data-ap="bstep" data-v="${step}" aria-label="Raise">+</button></div></label>`;
      if (b.mkt === "spread") html += sideSeg + stepper("Spread (alt lines: step it up or down)", 0.5);
      if (b.mkt === "ml") html += sideSeg;
      if (b.mkt === "total") html += ouSeg([["over", "Over"], ["under", "Under"]]) + stepper("Total", 0.5);
      if (b.mkt === "tt") html += sideSeg + ouSeg([["over", "Over"], ["under", "Under"]]) + stepper("Team total", 0.5);
      if (b.mkt === "prop") {
        const P = A.props[b.sp], cards = P && P.doc ? (P.doc.cards || []).filter(c => !g || c.game_id === g.id) : [];
        const stats = STATS[b.sp] || STATS.nba; if (!b.stat) b.stat = stats[0][0];
        html += `<label class="ap-fl"><span>Player</span><input class="ap-in txt" list="apPl" data-f="b.player" value="${H(b.player)}" placeholder="Start typing a name" autocomplete="off"><datalist id="apPl">${cards.map(c => `<option value="${H(c.name)}">`).join("")}</datalist></label>
          <label class="ap-fl"><span>Stat</span><select class="ap-sel" data-f="b.stat">${stats.map(([k, l]) => `<option value="${k}" ${b.stat === k ? "selected" : ""}>${l}</option>`).join("")}</select></label>`;
        if (b.stat !== "atd") html += ouSeg([["over", "Over"], ["under", "Under"], ["atleast", "At least (X+)"]]) + stepper(b.ou === "atleast" ? "At least" : "Line", b.ou === "atleast" ? 1 : 0.5);
      }
      const alt = b.dkLine !== undefined && b.dkLine !== "" && b.line !== "" && parseFloat(b.line) !== parseFloat(b.dkLine);
      html += `<label class="ap-fl"><span>DraftKings odds${alt ? " · alt line, enter DraftKings' price for it" : ""}</span><input class="ap-in" data-f="b.odds" value="${H(b.odds)}" placeholder="-110" inputmode="text"></label>`;
      const it = bldPick(b);
      html += `<div class="ap-prev"><i>Your pick</i><b id="apBp">${it ? H(it.label) : "—"}</b><i id="apBs" style="margin-top:4px">${it ? H(it.sub) + (it.odds ? " · " + H(mn(it.odds)) : "") : "Fill in the pick above"}</i></div>`;
      if (msg) html += msg;
      html += `<button type="button" class="ap-btn main" data-ap="badd" style="margin-top:16px">Add to slip</button>`;
    }
    html += A.slip.length ? `<button type="button" class="ap-btn" data-ap="slip">Review slip (${A.slip.length})</button>` : "";
    html += `<p class="ap-cap" style="margin-top:12px">Picks are written the way the auto-grader reads them, so they settle on their own when the game ends. Half and quarter bets need grading by hand in the admin.</p></div>`;
    return html;
  }
  function openBuilder(msg) {
    if (!A.bld) A.bld = newBld();
    openSheet(bldHtml(msg), "build"); A.reopen = () => openBuilder();
    const b = A.bld;
    if (!bldGames(b) || (PROPCFG[b.sp] && !A.props[b.sp])) ensureBldData(b).then(() => { if (A.sheet === "build" && A.bld === b) refreshBuilder(); });
  }
  function refreshBuilder(msg) {
    const s = q("#apSheet"), y = s.scrollTop, f = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.f : null;
    q("#apSheetBody").innerHTML = bldHtml(msg); s.scrollTop = y;
    if (f) { const el = q(`#apSheet [data-f="${f}"]`); if (el && el.tagName === "INPUT") { el.focus(); try { el.setSelectionRange(el.value.length, el.value.length); } catch (e) {} } }
  }
  function bldPreview() {
    const it = bldPick(A.bld), p = q("#apBp"), s = q("#apBs"); if (!p) return;
    p.textContent = it ? it.label : "—"; s.textContent = it ? it.sub + (it.odds ? " · " + mn(it.odds) : "") : "Fill in the pick above";
  }

  /* ---------------------------------------------------------------- sign in, help */
  function signInForm(why) {
    return `<div class="ap-form"><p style="margin:6px 0 0;color:var(--ap-muted);font-size:14px">${H(why)} Use your Ballwatcher account email and password. You stay signed in on this device.</p>
      <label class="ap-fl"><span>Email</span><input class="ap-in txt" type="email" id="apEm" autocomplete="username" value="${H(LS.get("bw_app_email", ""))}"></label>
      <label class="ap-fl"><span>Password</span><input class="ap-in txt" type="password" id="apPw" autocomplete="current-password"></label>
      <div id="apSiMsg"></div>
      <button type="button" class="ap-btn main" data-ap="signin" style="margin-top:16px">Sign in</button></div>`;
  }
  async function doSignIn() {
    const em = (q("#apEm") || {}).value || "", pw = (q("#apPw") || {}).value || "", out = q("#apSiMsg"), b = q('[data-ap="signin"]');
    if (b) { b.disabled = true; b.textContent = "Signing in…"; }
    try {
      const r = await api("login", { email: em, password: pw });
      LS.set("bw_app_email", em); setToken(r.token, r.name); A.me = { name: r.name, role: r.role, visibility: r.visibility }; loadConfig();
      toast(`Signed in${r.name ? " as " + r.name : ""}`);
      if (A.sheet === "slip") openSlip(); else { closeSheet(); go("card"); }
    } catch (e) { if (out) out.innerHTML = `<div class="ap-err">${H(e.message)}</div>`; if (b) { b.disabled = false; b.textContent = "Sign in"; } }
  }
  function openHelp(acMsg) {
    const html = sheetHead("How to use it", "") + `<div class="ap-form ap-help"><ol>
        <li><span>1</span><div><b>Today</b> shows the strongest prop Leans first, then every game with our spread and total picks. The % is how often picks like it won in testing.</div></li>
        <li><span>2</span><div>Tap any pick or game to research it: DraftKings vs our number, matchups, injuries, trends and line movement. Sections open when you tap them.</div></li>
        <li><span>3</span><div>${signedIn() ? "Tap <b>+ Add to slip</b> on a pick, or build any bet (alt lines, team totals, parlays) from <b>My card</b>. Log it and it's graded live." : "Signed in, you can add picks to a bet slip, log straight bets and parlays, and watch them graded live on <b>My card</b>."}</div></li>
        <li><span>4</span><div><b>Record</b> shows ${[...(signedIn() ? ["your bets"] : []), "the leaderboard", ...(A.cfg.model_record !== false ? ["how the model's picks have done"] : [])].join(", ").replace(/, ([^,]*)$/, " and $1")}.</div></li></ol>
      ${signedIn() ? accountHtml(acMsg)
        : `<button type="button" class="ap-btn main" data-ap="signinsheet" style="margin-top:18px">Sign in to My card</button>`}
      <button type="button" class="ap-btn ghost" data-ap="desk">Use the full desktop layout</button>
      <p class="ap-cap">21+ · Gambling problem? Call 1-800-GAMBLER · Not affiliated with DraftKings · Lines: DraftKings at time of pull.</p></div>`;
    openSheet(html, "help");
  }

  function accountHtml(msg) {
    const me = A.me || { name: LS.get("bw_app_name") || "", role: "member", visibility: "private" }, admin = me.role === "admin";
    return `<h2 class="ap-h2" style="margin:24px 0 8px">YOUR ACCOUNT</h2>
      <p style="margin:0;font-size:14.5px">Signed in as <b>${H(me.name)}</b>${admin ? " · admin" : ""}.</p>
      ${admin ? `<p class="ap-cap">Bets you log are Ballwatcher's picks: they go on Today's Card, The Record and the leaderboard.</p>`
        : `<p class="ap-cap" style="margin-top:12px">Who can see your bets</p>
          <div style="margin:6px -20px 0">${segHtml("vis", [["private", "Only me"], ["public", "Public + leaderboard"]], me.visibility === "public" ? "public" : "private")}</div>
          <p class="ap-cap">${me.visibility === "public" ? "Your record and bets show on the public leaderboard on ballwatcher.com and in the app." : "Your bets are private. Nobody else can see them."}</p>`}
      <label class="ap-fl"><span>Display name${admin ? "" : " (shown on the leaderboard)"}</span><div class="ap-g2" style="grid-template-columns:minmax(0,1fr) auto"><input class="ap-in txt" id="apNm" value="${H(me.name)}" maxlength="60"><button type="button" class="ap-btn" data-ap="savename" style="margin:0;width:auto;padding:0 16px;height:48px">Save</button></div></label>
      <details class="ap-det" style="margin:14px 0 0"><summary>Change password</summary><div class="in">
        <label class="ap-fl" style="margin-top:0"><span>Current password</span><input class="ap-in txt" type="password" id="apPw0" autocomplete="current-password"></label>
        <label class="ap-fl"><span>New password (8+ characters)</span><input class="ap-in txt" type="password" id="apPw1" autocomplete="new-password"></label>
        <button type="button" class="ap-btn" data-ap="savepw">Change password</button></div></details>
      <div id="apAcMsg">${msg || ""}</div>
      <button type="button" class="ap-btn" data-ap="signout" style="margin-top:16px">Sign out on this device</button>`;
  }
  async function saveAccount(body, okMsg) {
    try { const r = await api("account", body); A.me = { ...(A.me || {}), name: r.name, role: r.role, visibility: r.visibility }; LS.set("bw_app_name", r.name); openHelp(`<div class="ap-ok">${H(okMsg)}</div>`); }
    catch (e) { const m = q("#apAcMsg"); if (m) m.innerHTML = `<div class="ap-err">${H(e.message)}</div>`; }
  }

  /* ---------------------------------------------------------------- My card */
  async function renderCard(quiet) {
    const v = q("#apView"); if (!v || A.tab !== "card") return;
    const me = A.me || {}, note = !signedIn() ? "" : me.role === "admin" ? "Ballwatcher's picks · on Today's Card" : me.visibility === "public" ? "Public · on the leaderboard" : "Private · only you see these";
    const top = `${head("My card", todayLong())}${note ? `<p class="ap-note" style="margin-top:4px">${H(note)} · <button type="button" class="ap-more" data-ap="help" style="display:inline;min-height:0;padding:0">change</button></p>` : ""}<div style="padding:0 20px"><button type="button" class="ap-btn" data-ap="build">+ Build a bet</button></div>`;
    if (!signedIn()) { v.innerHTML = head("My card") + signInForm("Sign in to log bets, build parlays and watch them graded live."); return; }
    if (!A.card || !quiet) { if (!A.card) v.innerHTML = top + `<div class="ap-empty">Loading your card…</div>`; }
    try { A.card = await api("card"); A.cardAt = Date.now(); }
    catch (e) { if (e.status === 401) { drawNav(); v.innerHTML = head("My card") + signInForm("Your sign-in expired."); return; } if (!A.card) { v.innerHTML = top + `<div class="ap-err" style="margin:16px 20px">${H(e.message)}</div>`; return; } }
    if (A.tab !== "card") return;
    if (A.sheet) { A.dirty = true; drawNav(); return; }   // don't redraw under an open sheet
    const c = A.card, s = c.summary || {};
    const sec = (t, list) => list.length ? `<h2 class="ap-h2">${t}</h2>${list.map(betHtml).join("")}` : "";
    v.innerHTML = top + `<section class="ap-sum"><div><i>Today</i><b class="${s.net > 0 ? "up" : s.net < 0 ? "dn" : ""}">${H(s.net_txt || "0u")}</b></div><div><i>At risk</i><b>${H(s.risk_txt || "0u")}</b></div><div><i>Live</i><b>${c.live.length} of ${s.open || 0}</b></div></section>`
      + (c.live.length || c.upnext.length || c.awaiting.length || c.settled.length ? "" : `<div class="ap-empty">Nothing on your card yet. Add picks from Today, Games or Props, or build one above.</div>`)
      + sec("LIVE", c.live) + sec("UP NEXT", c.upnext) + sec("AWAITING RESULTS", c.awaiting) + sec("SETTLED TODAY", c.settled)
      + `<p class="ap-note">Live scores refresh every minute. Bets settle automatically a few minutes after the final.</p>`;
    drawNav();
    clearInterval(A.timer);
    A.timer = setInterval(() => { if (A.tab === "card" && document.visibilityState === "visible") renderCard(true); }, (c.live.length ? 60 : 180) * 1000);
  }
  function chipOf(b) {
    if (b.status === "won") return ["won", b.badge];
    if (b.status === "lost") return ["lost", b.badge];
    if (b.status === "push" || b.status === "void") return ["", b.badge];
    if (b.live && b.live.state === "post") return ["live", "FINAL · GRADING"];
    if (b.started) return ["live", b.live && b.live.detail ? `LIVE · ${b.live.detail}` : b.tone === "alive" ? b.badge : "LIVE"];
    return ["", whenTxt(b.starts_at)];
  }
  function trackHtml(lv) {
    if (!lv) return "";
    const cls = lv.now === "winning" ? "" : lv.now === "losing" ? "l" : "n";
    let bar = "";
    if (lv.pct != null && lv.line) bar = `<div class="ap-trk"><i class="${cls}" style="width:${Math.min(100, lv.pct / 1.5 * 100).toFixed(1)}%"></i><u style="left:${(100 / 1.5).toFixed(1)}%"></u></div>`;
    return `${bar}<div class="ap-lt"><span>${H(lv.score || "")}</span><b class="${lv.now === "winning" ? "up" : lv.now === "losing" ? "dn" : ""}">${H(lv.text || "")}</b></div>`;
  }
  function betHtml(b, view) {
    const [cls, txt] = chipOf(b);
    const par = b.kind !== "straight";
    const title = par ? `${b.legs.length}-leg ${b.kind === "sgp" ? "SGP" : "parlay"} · ${mn(b.odds_txt)}` : mn(b.label);
    const sub = par ? `${num(b.units, 2)}u to win ${b.to_win == null ? "—" : num(b.to_win, 2) + "u"}` : `${H(b.game || b.sub || "")} · ${num(b.units, 2)}u at ${H(mn(b.odds_txt))}`;
    let legs = "";
    if (par) legs = `<div class="ap-legs">${b.legs.map(l => { const lv = l.live, st = l.status !== "pending" ? l.status : lv && lv.now ? lv.now : lv && lv.state === "in" ? "live" : "";
      const right = l.status !== "pending" ? (l.status === "won" ? "Hit" : l.status === "lost" ? "Missed" : "Push") : lv ? (lv.state === "pre" ? whenTxt(b.starts_at) : lv.text || lv.detail) : "";
      return `<div class="ap-leg"><i class="${st}"></i><span>${H(mn(l.pick))}${lv && lv.state === "in" && lv.score ? `<small>${H(lv.score)} · ${H(lv.detail || "")}</small>` : ""}</span><em class="${lv && l.status === "pending" ? lv.now || "" : ""}">${H(right || "")}</em></div>`; }).join("")}</div>`;
    const note = b.status !== "pending" && b.note ? `<div class="ap-gn">${H(b.note.replace(/^Auto: /, ""))}</div>` : "";
    const rm = !view && !b.started && b.status === "pending" ? `<button type="button" class="ap-rm" data-ap="del" data-v="${b.id}">Delete this bet</button>` : "";
    return `<section class="ap-bet"><div class="ap-top"><span class="ap-chip ${cls}">${H(txt)}</span><span class="ap-when">${par ? (b.kind === "sgp" ? "Same game parlay" : "Parlay") : H(b.sport + (b.market ? " · " + b.market : ""))}</span></div>
      <h3>${H(title)}</h3><div class="s">${sub}</div>${par ? "" : trackHtml(b.live)}${legs}${note}${rm}</section>`;
  }
  async function delBet(btn) {
    if (!btn.classList.contains("sure")) { btn.classList.add("sure"); btn.textContent = "Tap again to delete"; setTimeout(() => { btn.classList.remove("sure"); btn.textContent = "Delete this bet"; }, 4000); return; }
    btn.textContent = "Deleting…";
    try { await api("delete", { id: +btn.dataset.v }); toast("Deleted"); renderCard(true); } catch (e) { toast(e.message); }
  }

  /* ---------------------------------------------------------------- Record */
  async function renderRecord() {
    const v = q("#apView"); if (!v) return;
    const modelOk = A.cfg.model_record !== false;
    const opts = [...(signedIn() ? [["mine", "My bets"]] : []), ["board", "Leaderboard"], ...(modelOk ? [["model", "Model picks"]] : [])];
    let who = opts.some(o => o[0] === A.recWho) ? A.recWho : opts[0][0];
    let html = head("Record") + segHtml("rwho", opts, who)
      + `<div class="ap-chips">${[["7", "7 days"], ["30", "30 days"], ["season", "Season"], ["all", "All"]].map(([k, l]) => `<button type="button" data-ap="rper" data-v="${k}" aria-pressed="${A.recPer === k}">${l}</button>`).join("")}</div>`;
    if (who === "model") { await loadGrades(); if (A.tab === "record") v.innerHTML = html + (A.cfg.model_record_hidden ? `<p class="ap-note" style="margin-top:12px">Hidden from everyone else (Settings in the admin). Admins still see it.</p>` : "") + modelRecord(); return; }
    if (who === "board") return renderBoard(v, html);
    v.innerHTML = html + `<div class="ap-empty">Loading your record…</div>`;
    let r;
    try { r = await api("record", { period: A.recPer }); } catch (e) { if (A.tab === "record") v.innerHTML = html + (e.status === 401 ? signInForm("Your sign-in expired.") : `<div class="ap-err" style="margin:16px 20px">${H(e.message)}</div>`); return; }
    if (A.tab !== "record" || A.recWho !== "mine") return;
    const t = r.total || {};
    html += `<section class="ap-sum four"><div><i>Units</i><b class="${t.net > 0 ? "up" : t.net < 0 ? "dn" : ""}">${H(t.units || "0u")}</b></div><div><i>Record</i><b>${H(t.record || "0-0")}</b></div><div><i>Win rate</i><b>${H(t.pct || "—")}</b></div><div><i>ROI</i><b class="${t.net > 0 ? "up" : t.net < 0 ? "dn" : ""}">${H(t.roi || "—")}</b></div></section>`;
    if ((r.timeline || []).length > 1) html += `<h2 class="ap-h2">UNITS OVER TIME</h2>${unitsChart(r.timeline)}`;
    if ((r.by_type || []).length) html += `<h2 class="ap-h2">BY TYPE</h2><div class="ap-list">${r.by_type.map(x => `<div class="ap-rrow"><span>${H(x.label)}</span><span>${H(x.record)}</span><b class="${x.net > 0 ? "up" : x.net < 0 ? "dn" : ""}">${H(x.units)}</b></div>`).join("")}</div>`;
    html += `<h2 class="ap-h2">RECENT</h2><div class="ap-list">${(r.recent || []).map(b => `<div class="ap-rrow"><span>${H(mn(b.kind === "straight" ? b.label : `${b.legs.length}-leg ${b.kind === "sgp" ? "SGP" : "parlay"}`))}<small>${H(new Date(b.starts_at).toLocaleDateString([], { month: "short", day: "numeric" }))}${b.note ? " · " + H(b.note.replace(/^Auto: /, "")) : ""}</small></span><span>${H(UP(b.status))}</span><b class="${b.result_units > 0 ? "up" : b.result_units < 0 ? "dn" : ""}">${b.result_units == null ? "—" : (b.result_units > 0 ? "+" : "") + mn(num(b.result_units, 2)) + "u"}</b></div>`).join("") || `<div class="ap-empty">No graded bets in this period.</div>`}</div>`;
    v.innerHTML = html;
  }
  async function renderBoard(v, html) {
    v.innerHTML = html + `<div class="ap-empty">Loading the leaderboard…</div>`;
    let r; try { r = await api("leaderboard", { period: A.recPer }); } catch (e) { if (A.tab === "record") v.innerHTML = html + `<div class="ap-err" style="margin:16px 20px">${H(e.message)}</div>`; return; }
    if (A.tab !== "record") return;
    let rank = 0;
    const rows = (r.rows || []).map(x => { if (x.n) rank++; return `<button type="button" class="ap-row" data-ap="bettor" data-v="${H(x.id)}">
        <span class="ap-rank">${x.n ? rank : "–"}</span>
        <span style="min-width:0"><span class="ap-rt">${H(x.name)}${x.house ? '<span class="ap-tag">OUR PICKS</span>' : ""}${x.me && !x.house ? '<span class="ap-tag fin">YOU</span>' : ""}${x.private ? '<span class="ap-tag fin">PRIVATE</span>' : ""}</span>
        <span class="ap-rs">${H(x.n ? x.record : "No graded bets")}${x.n ? ` · ${H(x.pct)} · ROI ${H(x.roi)}` : ""}</span></span>
        <span class="ap-pct ${x.net > 0 ? "" : x.net < 0 ? "dn" : "acc"}">${H(mn(x.units))}</span>${ICON.chev}</button>`; }).join("");
    const meRow = (r.rows || []).find(x => x.me && x.private);
    v.innerHTML = html + `<h2 class="ap-h2">RANKED BY UNITS</h2><div class="ap-list">${rows}</div>
      <p class="ap-note">${meRow ? "You're private: only you see your row. Change it under ? → Your account. " : ""}Members choose whether their bets are public.</p>`;
  }
  async function openBettor(id) {
    openSheet(sheetHead("Loading…", "") + `<div class="ap-empty">Loading…</div>`, "bettor");
    let r; try { r = await api("bettor", { id, period: A.recPer }); } catch (e) { openSheet(sheetHead("Card", "") + `<div class="ap-err" style="margin:16px 20px">${H(e.message)}</div>`, "bettor"); return; }
    if (A.sheet !== "bettor") return;
    const t = r.total || {};
    let html = sheetHead(H(r.name), id === "ballwatcher" ? "Ballwatcher's picks" : "")
      + `<section class="ap-sum four"><div><i>Units</i><b class="${t.net > 0 ? "up" : t.net < 0 ? "dn" : ""}">${H(t.units || "0u")}</b></div><div><i>Record</i><b>${H(t.record || "0-0")}</b></div><div><i>Win rate</i><b>${H(t.pct || "—")}</b></div><div><i>ROI</i><b>${H(t.roi || "—")}</b></div></section>`;
    if ((r.open || []).length) html += `<h2 class="ap-h2">OPEN BETS</h2>${r.open.map(b => betHtml(b, true)).join("")}`;
    if ((r.timeline || []).length > 1) html += `<h2 class="ap-h2">UNITS OVER TIME</h2>${unitsChart(r.timeline)}`;
    html += `<h2 class="ap-h2">RECENT</h2><div class="ap-list">${(r.recent || []).map(b => `<div class="ap-rrow"><span>${H(mn(b.kind === "straight" ? b.label : `${b.legs.length}-leg ${b.kind === "sgp" ? "SGP" : "parlay"}`))}<small>${H(new Date(b.starts_at).toLocaleDateString([], { month: "short", day: "numeric" }))}</small></span><span>${H(UP(b.status))}</span><b class="${b.result_units > 0 ? "up" : b.result_units < 0 ? "dn" : ""}">${b.result_units == null ? "—" : (b.result_units > 0 ? "+" : "") + mn(num(b.result_units, 2)) + "u"}</b></div>`).join("") || `<div class="ap-empty">Nothing graded in this period.</div>`}</div>`;
    openSheet(html, "bettor");
  }
  function unitsChart(tl) {
    const W = 340, Hh = 110, ys = [0, ...tl.map(x => x.cum)], y0 = Math.min(...ys), y1 = Math.max(...ys), sp = (y1 - y0) || 1;
    const X = i => (i / (tl.length - 1)) * W, Y = v => 6 + (Hh - 12) * (1 - (v - y0) / sp);
    const d = tl.map((x, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(x.cum).toFixed(1)}`).join(" ");
    const last = tl[tl.length - 1].cum, col = last >= 0 ? "#4ADE9B" : "#FF7A7A";
    const md = s => new Date(s + "T12:00:00").toLocaleDateString([], { month: "short", day: "numeric" });
    return `<div class="ap-chart"><svg viewBox="0 0 ${W} ${Hh}" preserveAspectRatio="none" role="img" aria-label="Units over time, ending at ${num(last, 2)}"><line x1="0" x2="${W}" y1="${Y(0).toFixed(1)}" y2="${Y(0).toFixed(1)}" stroke="#2A3240" stroke-dasharray="4 4"/><path d="${d}" fill="none" stroke="${col}" stroke-width="2.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg><div><span>${md(tl[0].day)}</span><span>${md(tl[tl.length - 1].day)}</span></div></div>`;
  }
  function modelRecord() {
    const R = state.grades || {};
    const since = A.recPer === "7" ? addDays(todayET(), -6) : A.recPer === "30" ? addDays(todayET(), -29) : A.recPer === "season" ? "2026-08-01" : "0000";
    const add = (a, r) => { if (r) { a.w += r.w || 0; a.l += r.l || 0; a.p += r.p || 0; a.u += r.units || 0; } return a; };
    const z = () => ({ w: 0, l: 0, p: 0, u: 0 });
    const lean = z(), allp = z(), games = z(), rows = [];
    for (const [lg, L] of Object.entries(R.prop_picks || {})) { const a = z(), b = z();
      for (const x of L.days || []) if (x.day >= since) { add(a, (x.rec || {}).LEAN); add(b, (x.rec || {}).ALL); }
      add(lean, a); add(allp, b); if (a.w + a.l + b.w + b.l) rows.push([`${spLabel(lg)} prop leans`, a], [`${spLabel(lg)} every prop pick`, b]); }
    for (const [lg, L] of Object.entries(R.leagues || {})) { const a = z();
      for (const x of L.days || []) if (x.day >= since) for (const mk of ["side", "total"]) add(a, (x[mk] || {}).ALL);
      add(games, a); if (a.w + a.l) rows.push([`${spLabel(lg)} game picks`, a]); }
    const wl = r => `${r.w}-${r.l}${r.p ? "-" + r.p : ""}`, pr = r => r.w + r.l ? Math.round(r.w / (r.w + r.l) * 1000) / 10 + "%" : "—";
    return `<section class="ap-sum"><div><i>Prop leans</i><b class="${lean.w > lean.l ? "up" : ""}">${wl(lean)}</b></div><div><i>Every prop pick</i><b>${wl(allp)}</b></div><div><i>Game picks</i><b>${wl(games)}</b></div></section>
      <section class="ap-sum"><div><i>Lean win rate</i><b>${pr(lean)}</b></div><div><i>Prop win rate</i><b>${pr(allp)}</b></div><div><i>Game win rate</i><b>${pr(games)}</b></div></section>
      <h2 class="ap-h2">BY SPORT</h2><div class="ap-list">${rows.map(([l, r]) => `<div class="ap-rrow"><span>${H(l)}</span><span>${wl(r)}</span><b class="${r.u > 0 ? "up" : r.u < 0 ? "dn" : ""}">${r.w + r.l ? (r.u > 0 ? "+" : "") + mn(num(r.u, 2)) + "u" : "—"}</b></div>`).join("") || `<div class="ap-empty">No graded model picks in this period yet.</div>`}</div>
      <p class="ap-note">Every pick the model makes is graded at DraftKings' line. Leans tested at 57–62%; game lines sit near 50–53%.</p>`;
  }

  /* ---------------------------------------------------------------- events */
  function onClick(e) {
    const el = e.target.closest("[data-ap]"); if (!el) return;
    const a = el.dataset.ap, v = el.dataset.v;
    switch (a) {
      case "tab": if (A.sheet) closeSheet(); if (v === A.tab) { window.scrollTo(0, 0); if (v === "card") renderCard(true); return; } go(v); return;
      case "help": openHelp(); if (signedIn()) api("me").then(m => { A.me = m; }).catch(() => {}); return;
      case "close": closeSheet(); return;
      case "tsp": A.tsp = v; A.tShow = 4; render(); return;
      case "tmore": A.tShow += 10; render(); return;
      case "gsp": A.gsp = v; render(); return;
      case "gkey": A.gkey[A.gsp] = v; render(); return;
      case "psp": A.psp = v; A.pShow = 40; render(); return;
      case "pmk": A.pmk[A.psp] = v; A.pShow = 40; render(); return;
      case "pleans": A.pleans = v === "1"; render(); return;
      case "pmore": A.pShow += 40; render(); return;
      case "item": { const x = A.items[+el.dataset.i]; if (!x) return;
        if (x.t === "g") openGame(x.sp, x.g, x.sl); else if (x.t === "p") openPlayer(x.sp, x.doc, x.L, x.c, x.mk); return; }
      case "pxmk": { const P = A.pctx; if (P) openPlayer(P.sp, P.doc, P.L, P.c, v); return; }
      case "add": { const x = A.items[+el.dataset.i]; if (!x) return; toggleSlip(x.it); const on = inSlip(x.it.key);
        el.classList.toggle("on", on); el.textContent = on ? "✓ In your slip" : `+ Add ${x.it.label} to slip`; return; }
      case "slip": openSlip(); return;
      case "smode": A.slipMode = v; if (v === "parlay") A.par.odds = estParlay(); openSlip(); return;
      case "srm": A.slip.splice(+el.dataset.i, 1); saveSlip(); A.par.odds = ""; openSlip(); return;
      case "sclear": A.slip = []; saveSlip(); openSlip(); return;
      case "log": logSlip(); return;
      case "build": A.bld = A.bld || newBld(); openBuilder(); return;
      case "bmkt": A.bld.mkt = v; if (v === "total" || v === "tt" || v === "prop") A.bld.ou = "over"; bldPrefill(A.bld); refreshBuilder(); return;
      case "bside": A.bld.side = v; bldPrefill(A.bld); refreshBuilder(); return;
      case "bou": { const was = A.bld.ou; A.bld.ou = v; if (A.bld.mkt === "prop" && (v === "atleast") !== (was === "atleast")) { const n = parseFloat(A.bld.line); if (isFinite(n)) A.bld.line = String(v === "atleast" ? Math.ceil(n) : Math.floor(n) + 0.5); }
        else if (A.bld.mkt !== "tt") bldPrefill(A.bld); refreshBuilder(); return; }
      case "bstep": { const b = A.bld, n = parseFloat(b.line); b.line = String(Math.round(((isFinite(n) ? n : 0) + parseFloat(v)) * 10) / 10);
        if (b.dkLine !== undefined && b.dkLine !== "") { if (parseFloat(b.line) === parseFloat(b.dkLine)) b.odds = b.dkOdds || b.odds; else if (b.odds === b.dkOdds) b.odds = ""; }
        refreshBuilder(); return; }
      case "badd": { const it = bldPick(A.bld);
        if (!it) { refreshBuilder(`<div class="ap-err">Finish the pick first: choose the side and line.</div>`); return; }
        if (!amDec(it.odds)) { refreshBuilder(`<div class="ap-err">Enter DraftKings' odds, like -110 or +135.</div>`); return; }
        if (!it.starts_at) { refreshBuilder(`<div class="ap-err">Add the game's start time.</div>`); return; }
        if (!inSlip(it.key)) { A.slip.push(it); saveSlip(); }
        refreshBuilder(`<div class="ap-ok">Added ${H(it.label)}. Build another leg, or review your slip.</div>`); return; }
      case "signin": doSignIn(); return;
      case "signinsheet": openSheet(sheetHead("Sign in", "") + signInForm("Sign in to log bets and track them on My card."), "signin"); return;
      case "signout": api("logout").catch(() => {}); setToken(null); A.card = null; closeSheet(); toast("Signed out"); render(); return;
      case "desk": LS.set("bw_layout", "desk"); location.reload(); return;
      case "del": delBet(el); return;
      case "bettor": openBettor(v); return;
      case "vis": saveAccount({ visibility: v }, v === "public" ? "Your bets are public now and you're on the leaderboard." : "Your bets are private now."); return;
      case "savename": saveAccount({ name: (q("#apNm") || {}).value || "" }, "Name saved."); return;
      case "savepw": { const a = (q("#apPw0") || {}).value || "", b = (q("#apPw1") || {}).value || "";
        api("password", { current: a, new: b }).then(() => openHelp(`<div class="ap-ok">Password changed. You're still signed in here; other devices were signed out.</div>`))
          .catch(e2 => { const m = q("#apAcMsg"); if (m) m.innerHTML = `<div class="ap-err">${H(e2.message)}</div>`; }); return; }
      case "rwho": A.recWho = v; render(); return;
      case "rper": A.recPer = v; render(); return;
    }
  }
  function onInput(e) {
    const f = e.target.dataset && e.target.dataset.f;
    if (e.target.id === "apPq") { A.pq = e.target.value; A.pShow = 40; clearTimeout(onInput.t); onInput.t = setTimeout(render, 180); return; }
    if (!f) return;
    const val = e.target.value;
    if (f === "odds" || f === "units") { const x = A.slip[+e.target.dataset.i]; if (!x) return; x[f] = val; saveSlip();
      const tw = q("#apTw"); if (tw) tw.textContent = A.slipMode === "parlay" ? uTxt(toWin(A.par.units, A.par.odds)) : uTxt(A.slip.reduce((a, y) => a + (toWin(y.units || 1, y.odds) || 0), 0)); return; }
    if (f === "podds" || f === "punits") { A.par[f === "podds" ? "odds" : "units"] = val; const tw = q("#apTw"); if (tw) tw.textContent = uTxt(toWin(A.par.units, A.par.odds)); return; }
    if (f.startsWith("b.") && A.bld) { const k = f.slice(2); if (["line", "odds", "custom", "when", "team", "player"].includes(k)) { A.bld[k] = val; bldPreview(); } }
  }
  function onChange(e) {
    const f = e.target.dataset && e.target.dataset.f; if (!f || !f.startsWith("b.") || !A.bld) return;
    const k = f.slice(2), b = A.bld;
    if (k === "sp") { A.bld = newBld(e.target.value); openBuilder(); return; }
    if (k === "gid") { b.gid = e.target.value; b.player = ""; bldPrefill(b); refreshBuilder(); return; }
    if (k === "stat") { b.stat = e.target.value; bldPrefill(b); refreshBuilder(); return; }
    if (k === "player") { b.player = e.target.value; bldPrefill(b); refreshBuilder(); return; }
  }

  window.BWAPP = { want, start, deskLink, get on() { return A.on; } };
})();
