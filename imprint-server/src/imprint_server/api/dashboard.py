# ruff: noqa: E501
"""Admin dashboard served at /admin.

Returns a self-contained HTML document with inlined CSS and JS.
No build step, no external dependencies beyond Google Fonts CDN.
Fetches data from the existing REST API endpoints using the same
Bearer token the operator provides.

Auth: protected by the same AuthMiddleware as /v1. When auth is
disabled, the dashboard loads without any token prompt.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_LOGO_SVG = """<svg width="36" height="36" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <path d="M 12 6 H 52 A 6 6 0 0 1 58 12 V 30 H 6 V 12 A 6 6 0 0 1 12 6 Z" fill="#0d9488"/>
  <rect x="38" y="10" width="6" height="14" fill="#f5f5f0"/>
  <path d="M 6 34 H 58 V 52 A 6 6 0 0 1 52 58 H 12 A 6 6 0 0 1 6 52 Z" fill="none" stroke="#f5f5f0" stroke-width="2"/>
  <circle cx="13" cy="42" r="2" fill="#5eead4"/>
  <rect x="18" y="40" width="13" height="4" rx="1" fill="#f5f5f0"/>
  <line x1="32" y1="42" x2="36" y2="42" stroke="#f5f5f0" stroke-width="1.4"/>
  <rect x="37" y="40" width="13" height="4" rx="1" fill="#f5f5f0"/>
  <circle cx="13" cy="50" r="2" fill="none" stroke="#f5f5f0" stroke-width="1" opacity=".5"/>
  <rect x="18" y="48" width="13" height="4" rx="1" fill="#f5f5f0" opacity=".4"/>
  <line x1="32" y1="50" x2="36" y2="50" stroke="#f5f5f0" stroke-width="1.4" opacity=".4"/>
  <rect x="37" y="48" width="13" height="4" rx="1" fill="#f5f5f0" opacity=".4"/>
</svg>"""

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>imprint admin</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg:       #070909;
    --bg1:      #0d1212;
    --bg2:      #121919;
    --border:   #1e2e2e;
    --border2:  #243535;
    --teal:     #0d9488;
    --teal-hi:  #14b8a6;
    --teal-dim: #0d948822;
    --ice:      #f5f5f0;
    --ice-dim:  #f5f5f055;
    --muted:    #718080;
    --danger:   #ef4444;
    --warn:     #f59e0b;
    --ok:       #22c55e;
    --radius:   6px;
    --mono:     "DM Mono", monospace;
    --sans:     "Syne", sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--ice);
    font-family: var(--sans);
    font-size: 13px;
    line-height: 1.5;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }

  /* noise overlay */
  body::before {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
    z-index: 0;
    opacity: 0.6;
  }

  /* teal top bar */
  .topbar {
    position: fixed; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--teal), var(--teal-hi), var(--teal));
    z-index: 100;
  }

  /* left sidebar */
  .sidebar {
    position: fixed; left: 0; top: 0; bottom: 0;
    width: 220px;
    background: var(--bg1);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 28px 0 20px;
    z-index: 10;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0 20px 28px;
    border-bottom: 1px solid var(--border);
  }
  .brand-text {
    display: flex; flex-direction: column;
    gap: 1px;
  }
  .brand-name {
    font-family: var(--sans);
    font-weight: 700;
    font-size: 15px;
    letter-spacing: -0.01em;
    color: var(--ice);
  }
  .brand-sub {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .nav {
    flex: 1;
    padding: 20px 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-radius: var(--radius);
    cursor: pointer;
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.01em;
    transition: all 0.15s;
    border: 1px solid transparent;
    user-select: none;
  }
  .nav-item:hover { color: var(--ice); background: var(--bg2); }
  .nav-item.active {
    color: var(--teal-hi);
    background: var(--teal-dim);
    border-color: var(--border2);
  }
  .nav-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; }

  .sidebar-footer {
    padding: 16px 20px 0;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .status-label { font-family: var(--mono); font-size: 10px; color: var(--muted); }
  .status-badge {
    font-family: var(--mono);
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 2px;
    font-weight: 500;
  }
  .badge-ok { background: #14532d44; color: var(--ok); border: 1px solid #16a34a44; }
  .badge-warn { background: #78350f44; color: var(--warn); border: 1px solid #d9770644; }
  .badge-err { background: #450a0a44; color: var(--danger); border: 1px solid #dc262644; }

  .refresh-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 4px;
  }
  .refresh-label { font-family: var(--mono); font-size: 10px; color: var(--muted); }
  .toggle {
    position: relative; width: 28px; height: 15px;
    background: var(--border2); border-radius: 8px; cursor: pointer;
    transition: background 0.2s;
  }
  .toggle.on { background: var(--teal); }
  .toggle::after {
    content: ""; position: absolute;
    top: 2px; left: 2px;
    width: 11px; height: 11px;
    background: var(--ice); border-radius: 50%;
    transition: left 0.2s;
  }
  .toggle.on::after { left: 15px; }

  /* main area */
  .main {
    margin-left: 220px;
    padding: 40px 36px;
    min-height: 100vh;
    position: relative; z-index: 1;
    animation: fade-in 0.4s ease;
  }

  @keyframes fade-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

  .page { display: none; }
  .page.active { display: block; }

  .page-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 28px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }
  .page-title {
    font-family: var(--sans);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--ice);
  }
  .page-subtitle { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.08em; }

  /* stat strip */
  .stat-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 28px;
    animation: fade-in 0.4s ease 0.1s both;
  }
  .stat-card {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .stat-label { font-family: var(--mono); font-size: 9px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
  .stat-value {
    font-family: var(--mono);
    font-size: 26px;
    font-weight: 500;
    color: var(--ice);
    line-height: 1.1;
  }
  .stat-value.teal { color: var(--teal-hi); }
  .stat-sub { font-family: var(--mono); font-size: 9px; color: var(--muted); }

  /* table */
  .section {
    margin-bottom: 28px;
    animation: fade-in 0.4s ease 0.2s both;
  }
  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
  }
  .section-title { font-family: var(--mono); font-size: 10px; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; }
  .section-count { font-family: var(--mono); font-size: 10px; color: var(--teal); }

  .table-wrap {
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th {
    font-family: var(--mono);
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  td {
    padding: 11px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--ice-dim);
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg2); }
  td.primary { color: var(--ice); font-weight: 500; }
  td.accent { color: var(--teal-hi); }
  td.dim { color: var(--muted); }

  /* mode pill */
  .pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 9px;
    padding: 2px 7px;
    border-radius: 2px;
    letter-spacing: 0.06em;
    font-weight: 500;
  }
  .pill-frugal   { background: #164e6344; color: #38bdf8; border: 1px solid #0ea5e944; }
  .pill-balanced { background: #3b273344; color: #f0abfc; border: 1px solid #d946ef44; }
  .pill-eager    { background: #2d1b0044; color: #fcd34d; border: 1px solid #f59e0b44; }
  .pill-default  { background: #1e2e2e44; color: var(--muted); border: 1px solid var(--border); }

  /* empty state */
  .empty {
    padding: 40px;
    text-align: center;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.06em;
  }

  /* loading spinner */
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 1.5px solid var(--border2);
    border-top-color: var(--teal);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* pulse live indicator */
  .live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--ok);
    border-radius: 50%;
    margin-right: 5px;
    box-shadow: 0 0 0 0 #22c55e88;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 #22c55e88; }
    70%  { box-shadow: 0 0 0 5px #22c55e00; }
    100% { box-shadow: 0 0 0 0 #22c55e00; }
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* error banner */
  .error-banner {
    background: #450a0a44;
    border: 1px solid #dc262644;
    color: var(--danger);
    font-family: var(--mono);
    font-size: 11px;
    padding: 10px 14px;
    border-radius: var(--radius);
    margin-bottom: 16px;
    display: none;
  }

  /* auth modal */
  .modal-bg {
    position: fixed; inset: 0;
    background: #000000cc;
    backdrop-filter: blur(4px);
    z-index: 200;
    display: flex;
    align-items: center;
    justify-content: center;
    display: none;
  }
  .modal-bg.open { display: flex; }
  .modal {
    background: var(--bg1);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 32px;
    width: 360px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    animation: fade-in 0.2s ease;
  }
  .modal-title { font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }
  .modal-sub { font-family: var(--mono); font-size: 10px; color: var(--muted); line-height: 1.6; }
  .modal-input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    color: var(--ice);
    font-family: var(--mono);
    font-size: 12px;
    padding: 10px 12px;
    outline: none;
    transition: border-color 0.15s;
  }
  .modal-input:focus { border-color: var(--teal); }
  .modal-btn {
    background: var(--teal);
    color: var(--ice);
    border: none;
    border-radius: var(--radius);
    font-family: var(--sans);
    font-size: 13px;
    font-weight: 600;
    padding: 10px;
    cursor: pointer;
    transition: background 0.15s;
    width: 100%;
  }
  .modal-btn:hover { background: var(--teal-hi); }

  .tok-row {
    display: flex; align-items: center; gap: 8px;
  }
  .tok-badge {
    font-family: var(--mono); font-size: 9px; color: var(--teal);
    background: var(--teal-dim); border: 1px solid var(--border2);
    padding: 2px 6px; border-radius: 2px; white-space: nowrap;
  }
  .tok-clear {
    font-family: var(--mono); font-size: 9px; color: var(--muted);
    cursor: pointer; text-decoration: underline; padding: 0;
    background: none; border: none; color: var(--muted);
  }
  .tok-clear:hover { color: var(--danger); }
</style>
</head>
<body>
<div class="topbar"></div>

<!-- Auth modal -->
<div class="modal-bg" id="auth-modal">
  <div class="modal">
    <div>
      <div class="modal-title">API Key Required</div>
      <div style="margin-top:8px" class="modal-sub">
        Enter a master API key to access the admin dashboard.<br/>
        The key is stored in sessionStorage and cleared on tab close.
      </div>
    </div>
    <input class="modal-input" id="key-input" type="password" placeholder="sk-imp-..." autocomplete="off"/>
    <button class="modal-btn" onclick="submitKey()">Continue</button>
  </div>
</div>

<!-- Sidebar -->
<div class="sidebar">
  <div class="brand">
    LOGO_SVG_PLACEHOLDER
    <div class="brand-text">
      <div class="brand-name">imprint</div>
      <div class="brand-sub">admin console</div>
    </div>
  </div>

  <nav class="nav">
    <div class="nav-item active" onclick="showPage('overview', this)">
      <div class="nav-dot"></div>Overview
    </div>
    <div class="nav-item" onclick="showPage('agents', this)">
      <div class="nav-dot"></div>Agents
    </div>
    <div class="nav-item" onclick="showPage('keys', this)">
      <div class="nav-dot"></div>API Keys
    </div>
  </nav>

  <div class="sidebar-footer">
    <div class="status-row">
      <div class="status-label">system</div>
      <div class="status-badge badge-ok" id="sys-badge">--</div>
    </div>
    <div class="status-row">
      <div class="status-label">store</div>
      <div class="status-badge badge-ok" id="store-badge">--</div>
    </div>
    <div class="status-row">
      <div class="status-label">redis</div>
      <div class="status-badge" id="redis-badge">--</div>
    </div>
    <div class="refresh-row">
      <div class="refresh-label">auto-refresh</div>
      <div class="toggle" id="refresh-toggle" onclick="toggleRefresh(this)"></div>
    </div>
    <div id="tok-row" class="tok-row" style="display:none">
      <div class="tok-badge" id="tok-label">key set</div>
      <button class="tok-clear" onclick="clearKey()">clear</button>
    </div>
  </div>
</div>

<!-- Main -->
<div class="main">

  <!-- Overview -->
  <div class="page active" id="page-overview">
    <div class="page-header">
      <div>
        <div class="page-title">Overview</div>
        <div class="page-subtitle" id="last-refresh">Fetching...</div>
      </div>
    </div>
    <div class="error-banner" id="err-overview"></div>
    <div class="stat-strip" id="stat-strip">
      <div class="stat-card"><div class="stat-label">agents loaded</div><div class="stat-value teal" id="s-agents">--</div></div>
      <div class="stat-card"><div class="stat-label">store</div><div class="stat-value" id="s-store" style="font-size:16px;padding-top:5px">--</div></div>
      <div class="stat-card"><div class="stat-label">redis</div><div class="stat-value" id="s-redis" style="font-size:16px;padding-top:5px">--</div></div>
      <div class="stat-card"><div class="stat-label">db status</div><div class="stat-value" id="s-db" style="font-size:16px;padding-top:5px">--</div></div>
    </div>
    <div class="section">
      <div class="section-head">
        <div class="section-title">Loaded Agents</div>
        <div class="section-count" id="agent-count-label"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Agent ID</th>
            <th>Mode</th>
            <th>Dynamic Scopes</th>
            <th>Scopes</th>
          </tr></thead>
          <tbody id="agents-tbody"><tr><td colspan="4" class="empty"><span class="spinner"></span>Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Agents -->
  <div class="page" id="page-agents">
    <div class="page-header">
      <div>
        <div class="page-title">Agents</div>
        <div class="page-subtitle">All initialized agent configurations</div>
      </div>
    </div>
    <div class="error-banner" id="err-agents"></div>
    <div class="section">
      <div class="section-head">
        <div class="section-title">Agent Registry</div>
        <div class="section-count" id="agents-full-count"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Agent ID</th>
            <th>Mode</th>
            <th>Description</th>
            <th>Scopes</th>
            <th>Dynamic Scopes</th>
          </tr></thead>
          <tbody id="agents-full-tbody"><tr><td colspan="5" class="empty"><span class="spinner"></span>Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Keys -->
  <div class="page" id="page-keys">
    <div class="page-header">
      <div>
        <div class="page-title">API Keys</div>
        <div class="page-subtitle">Active key summary -- hashes only, raw keys are never stored</div>
      </div>
    </div>
    <div class="error-banner" id="err-keys"></div>
    <div class="stat-strip">
      <div class="stat-card"><div class="stat-label">active keys</div><div class="stat-value teal" id="k-count">--</div></div>
      <div class="stat-card"><div class="stat-label">master keys</div><div class="stat-value" id="k-master">--</div></div>
      <div class="stat-card"><div class="stat-label">scoped keys</div><div class="stat-value" id="k-scoped">--</div></div>
      <div class="stat-card"><div class="stat-label">user keys</div><div class="stat-value" id="k-user">--</div></div>
    </div>
    <div class="section">
      <div class="section-head">
        <div class="section-title">Key List</div>
        <div class="section-count" id="key-count-label"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Hash (first 16)</th>
            <th>Label</th>
            <th>Agent Scope</th>
            <th>User</th>
            <th>Status</th>
            <th>Created</th>
          </tr></thead>
          <tbody id="keys-tbody"><tr><td colspan="6" class="empty"><span class="spinner"></span>Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

</div>

<script>
const API = "";
let _token = sessionStorage.getItem("imprint_admin_token") || "";
let _refreshInterval = null;
let _currentPage = "overview";

function headers() {
  const h = {"Content-Type": "application/json"};
  if (_token) h["Authorization"] = "Bearer " + _token;
  return h;
}

async function api(path) {
  const r = await fetch(API + path, {headers: headers()});
  if (r.status === 401) { showModal(); throw new Error("401"); }
  if (!r.ok) throw new Error(r.status + " " + r.statusText);
  return r.json();
}

function showModal() {
  document.getElementById("auth-modal").classList.add("open");
  setTimeout(() => document.getElementById("key-input").focus(), 50);
}

function submitKey() {
  const v = document.getElementById("key-input").value.trim();
  if (!v) return;
  _token = v;
  sessionStorage.setItem("imprint_admin_token", v);
  document.getElementById("auth-modal").classList.remove("open");
  updateTokRow();
  refresh();
}

document.getElementById("key-input").addEventListener("keydown", e => { if (e.key === "Enter") submitKey(); });

function clearKey() {
  _token = "";
  sessionStorage.removeItem("imprint_admin_token");
  updateTokRow();
  showModal();
}

function updateTokRow() {
  const row = document.getElementById("tok-row");
  if (_token) {
    row.style.display = "flex";
    document.getElementById("tok-label").textContent = _token.slice(0, 12) + "...";
  } else {
    row.style.display = "none";
  }
}

function showPage(name, el) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  document.getElementById("page-" + name).classList.add("active");
  el.classList.add("active");
  _currentPage = name;
  refresh();
}

function pill(mode) {
  const m = (mode || "").toLowerCase();
  const cls = m === "frugal" ? "pill-frugal" : m === "balanced" ? "pill-balanced" : m === "eager" ? "pill-eager" : "pill-default";
  return `<span class="pill ${cls}">${m || "default"}</span>`;
}

function badge(ok) {
  return ok
    ? `<span class="status-badge badge-ok">ok</span>`
    : `<span class="status-badge badge-err">err</span>`;
}

function ts(isoStr) {
  if (!isoStr) return "--";
  try { return new Date(isoStr).toLocaleString([], {dateStyle:"short", timeStyle:"short"}); }
  catch { return isoStr; }
}

function setErr(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  if (msg) { el.textContent = msg; el.style.display = "block"; }
  else { el.style.display = "none"; }
}

async function loadHealth() {
  try {
    const h = await api("/health");
    const ok = h.status === "ok";
    document.getElementById("sys-badge").textContent = h.status;
    document.getElementById("sys-badge").className = "status-badge " + (ok ? "badge-ok" : "badge-warn");
    document.getElementById("store-badge").textContent = h.store || "--";
    document.getElementById("redis-badge").textContent = h.redis || "--";
    document.getElementById("redis-badge").className = "status-badge " + (h.redis === "ok" ? "badge-ok" : h.redis === "disabled" ? "badge-warn" : "badge-err");
    document.getElementById("s-agents").textContent = h.agents_loaded ?? "--";
    document.getElementById("s-store").textContent = h.store || "--";
    document.getElementById("s-redis").textContent = h.redis || "--";
    document.getElementById("s-db").innerHTML = badge(h.db_ok);
    document.getElementById("last-refresh").innerHTML = `<span class="live-dot"></span>Last refreshed ` + new Date().toLocaleTimeString();
  } catch (e) {
    if (e.message !== "401") setErr("err-overview", "Health check failed: " + e.message);
  }
}

async function loadAgentsOverview() {
  const tbody = document.getElementById("agents-tbody");
  try {
    const agents = await api("/v1/agents");
    document.getElementById("agent-count-label").textContent = agents.length + " agent" + (agents.length !== 1 ? "s" : "");
    if (!agents.length) { tbody.innerHTML = `<tr><td colspan="4" class="empty">No agents initialized</td></tr>`; return; }
    tbody.innerHTML = agents.map(a => `<tr>
      <td class="primary">${a.agent_id}</td>
      <td>${pill(a.processing_mode)}</td>
      <td class="dim">${a.dynamic_scopes ? badge(true) : ""}</td>
      <td class="dim">${(a.scopes || []).join(", ") || "--"}</td>
    </tr>`).join("");
  } catch (e) {
    if (e.message !== "401") { tbody.innerHTML = `<tr><td colspan="4" class="empty">Failed to load agents</td></tr>`; setErr("err-overview", e.message); }
  }
}

async function loadAgentsFull() {
  const tbody = document.getElementById("agents-full-tbody");
  try {
    const agents = await api("/v1/agents");
    document.getElementById("agents-full-count").textContent = agents.length + " total";
    if (!agents.length) { tbody.innerHTML = `<tr><td colspan="5" class="empty">No agents initialized</td></tr>`; return; }
    tbody.innerHTML = agents.map(a => `<tr>
      <td class="primary">${a.agent_id}</td>
      <td>${pill(a.processing_mode)}</td>
      <td class="dim">${a.agent_description || "--"}</td>
      <td class="dim">${(a.scopes || []).join(", ") || "--"}</td>
      <td class="dim">${a.dynamic_scopes ? badge(true) : "--"}</td>
    </tr>`).join("");
  } catch (e) {
    if (e.message !== "401") { tbody.innerHTML = `<tr><td colspan="5" class="empty">Failed to load agents</td></tr>`; setErr("err-agents", e.message); }
  }
}

async function loadKeys() {
  const tbody = document.getElementById("keys-tbody");
  try {
    const keys = await api("/v1/keys");
    const active = keys.filter(k => k.active);
    const master = active.filter(k => !k.agent_id);
    const scoped = active.filter(k => k.agent_id);
    const user   = active.filter(k => k.user_id);
    document.getElementById("k-count").textContent = active.length;
    document.getElementById("k-master").textContent = master.length;
    document.getElementById("k-scoped").textContent = scoped.length;
    document.getElementById("k-user").textContent = user.length;
    document.getElementById("key-count-label").textContent = keys.length + " total";
    if (!keys.length) { tbody.innerHTML = `<tr><td colspan="6" class="empty">No keys found</td></tr>`; return; }
    tbody.innerHTML = keys.map(k => `<tr>
      <td class="accent">${k.key_hash ? k.key_hash.slice(0, 16) : "--"}</td>
      <td class="dim">${k.label || "--"}</td>
      <td class="dim">${k.agent_id || "<span style='color:var(--muted)'>master</span>"}</td>
      <td class="dim">${k.user_id || "--"}</td>
      <td>${k.active ? badge(true) : `<span class="status-badge badge-err">revoked</span>`}</td>
      <td class="dim">${ts(k.created_at)}</td>
    </tr>`).join("");
  } catch (e) {
    if (e.message !== "401") { tbody.innerHTML = `<tr><td colspan="6" class="empty">Failed to load keys</td></tr>`; setErr("err-keys", e.message); }
  }
}

async function refresh() {
  setErr("err-overview", "");
  setErr("err-agents", "");
  setErr("err-keys", "");
  await loadHealth();
  if (_currentPage === "overview") await loadAgentsOverview();
  else if (_currentPage === "agents") await loadAgentsFull();
  else if (_currentPage === "keys") await loadKeys();
}

function toggleRefresh(el) {
  el.classList.toggle("on");
  if (el.classList.contains("on")) {
    _refreshInterval = setInterval(refresh, 30000);
  } else {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// init
updateTokRow();
if (!_token) {
  // try unauthenticated first (auth_disabled mode)
  fetch("/health").then(r => {
    if (r.status === 200) { refresh(); }
    else { showModal(); }
  }).catch(() => showModal());
} else {
  refresh();
}
</script>
</body>
</html>"""

_DASHBOARD_HTML = _HTML.replace("LOGO_SVG_PLACEHOLDER", _LOGO_SVG)


@router.get(
    "/admin",
    operation_id="admin_dashboard",
    tags=["system"],
    summary="Read-only admin dashboard",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def admin_dashboard() -> HTMLResponse:
    """Serve the read-only admin dashboard.

    Returns a self-contained HTML document. All data is fetched client-side
    from the existing REST API endpoints using the operator's Bearer token.
    Auth is enforced by AuthMiddleware like any other protected route.

    include_in_schema=False: HTML routes don't belong in the OpenAPI spec.
    """
    return HTMLResponse(_DASHBOARD_HTML)
