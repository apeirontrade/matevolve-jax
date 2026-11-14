#!/usr/bin/env python3
"""
Local HTTP viewer (http://127.0.0.1:8550) for the literature-sourced materials catalog
and optional, unranked synthetic placeholder rows.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
from typing import List
from urllib.parse import urlparse

from engine import (
    UniversalCompound,
    generate_placeholder_candidates,
    get_universal_catalog,
)

MAX_PLACEHOLDER_ROWS: int = 20

SOURCED_CATALOG: List[UniversalCompound] = get_universal_catalog()
PLACEHOLDER_ROWS: List[UniversalCompound] = []
PLACEHOLDER_RNG: random.Random = random.Random(42)


def reset_state() -> None:
    """Resets in-memory placeholder list and RNG seed (used by unit tests)."""
    global PLACEHOLDER_ROWS, PLACEHOLDER_RNG
    PLACEHOLDER_ROWS = []
    PLACEHOLDER_RNG = random.Random(42)


def serialize_compound(c: UniversalCompound) -> dict:
    label, score, areas, summary = c.evaluate_heuristics()
    return {
        "name": c.name,
        "formula": c.formula,
        "structure_class": c.structure_class,
        "composition": c.composition,
        "cost_per_kg": round(c.raw_cost_per_kg, 2),
        "density": round(c.density_g_cm3, 2),
        "hardness_gpa": round(c.vickers_hardness_gpa, 1) if c.vickers_hardness_gpa is not None else None,
        "pugh_ratio": round(c.pugh_ratio, 2),
        "poisson_ratio": round(c.poisson_ratio, 3),
        "stiffness_gpa": round(c.youngs_modulus_gpa, 1),
        "thermal_expansion_ppm": round(c.thermal_expansion_ppm, 2),
        "thermal_cond": round(c.thermal_cond_w_mk, 2),
        "band_gap_ev": round(c.band_gap_ev, 2),
        "heuristic_label": label,
        "heuristic_score": round(score, 1),
        "possible_applications": [] if c.placeholder else areas,
        "heuristic_summary": summary,
        "source": c.source,
        "property_sources": c.property_sources,
        "placeholder": bool(c.placeholder),
        "scores": {
            "all": round(score, 1),
            "dielectric_heat": round(c.heuristic_dielectric_heat, 1),
            "elastic_screen": round(c.heuristic_elastic_screen, 1),
            "low_thermal_expansion": round(c.heuristic_low_thermal_expansion, 1),
            "auxetic": round(c.heuristic_auxetic, 1),
        },
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Materials Catalog & Heuristic Scoring Viewer</title>
<style>
  :root {
    --bg: #090d16;
    --panel: #111827;
    --border: #1f2937;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 22px; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 18px; }
  .header h1 { margin: 0; font-size: 20px; color: #fff; }
  .header p { margin: 5px 0 0; color: var(--muted); font-size: 13px; }
  .btn-placeholder {
    background: #1e293b; color: var(--text); border: 1px solid var(--border);
    padding: 10px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 13px;
  }
  .btn-placeholder:hover { border-color: var(--accent); }
  .layout { display: grid; grid-template-columns: 300px 1fr; gap: 18px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
  .lens-btn {
    width: 100%; text-align: left; background: #1e293b; color: var(--text);
    border: 1px solid var(--border); padding: 10px 12px; border-radius: 8px;
    margin-bottom: 8px; cursor: pointer; font-size: 12.5px; font-weight: 500;
  }
  .lens-btn.active { border-color: var(--accent); background: rgba(56,189,248,0.14); color: #fff; font-weight: 600; }
  .slider-group { margin-top: 14px; }
  .slider-group label { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 6px; }
  .slider-group input[type=range] { width: 100%; accent-color: var(--accent); }
  table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  th { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--border); color: var(--muted); }
  td { padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); cursor: pointer; vertical-align: middle; }
  tr:hover td { background: rgba(56, 189, 248, 0.07); }
  .pill { display: inline-block; padding: 3px 8px; border-radius: 99px; font-size: 11px; font-weight: 600; background: rgba(56,189,248,0.15); color: #7dd3fc; }
  .pill-none { background: rgba(148,163,184,0.15); color: #94a3b8; }
  .pill-placeholder { background: rgba(245,158,11,0.2); color: #fcd34d; }
  .app-tag { display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 11px; background: rgba(34,197,94,0.12); color: #86efac; margin: 2px 3px 2px 0; }
  .inspector { margin-top: 16px; background: #0c1322; border: 1px solid var(--accent); border-radius: 10px; padding: 16px; }
  .stat-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-top: 12px; }
  .stat { background: var(--panel); padding: 9px; border-radius: 8px; border: 1px solid var(--border); }
  .stat .k { font-size: 10.5px; color: var(--muted); }
  .stat .v { font-size: 13.5px; font-weight: 700; margin-top: 3px; color: #fff; }
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Materials Catalog & Heuristic Scoring Demo</h1>
      <p>Displays literature-sourced materials and derived elastic/thermal heuristics (standard-library prototype).</p>
    </div>
    <button class="btn-placeholder" onclick="addPlaceholders()">+ Generate 4 Synthetic Placeholders (Max 20)</button>
  </div>

  <div class="layout">
    <div class="card">
      <h3 style="margin:0 0 10px; font-size:14px;">Heuristic Filter Lenses</h3>
      <button class="lens-btn active" onclick="setLens('all', this)">All Heuristics (Max Score)</button>
      <button class="lens-btn" onclick="setLens('dielectric_heat', this)">Dielectric Thermal Conductor</button>
      <button class="lens-btn" onclick="setLens('low_thermal_expansion', this)">Negative / Low Thermal Expansion</button>
      <button class="lens-btn" onclick="setLens('elastic_screen', this)">Elastic Screen: Hv &gt;= 15, B/G &gt;= 1.35, Density &lt;= 5</button>
      <button class="lens-btn" onclick="setLens('auxetic', this)">Auxetic (Negative Poisson Ratio)</button>

      <div class="slider-group">
        <label><span>Max Raw Element Cost ($/kg)</span><span id="v_cost">$500.00/kg</span></label>
        <input type="range" id="s_cost" min="1.0" max="500.0" step="1.0" value="500.0" oninput="render()">
      </div>
      <div class="slider-group">
        <label><span>Max Density (g/cm³)</span><span id="v_dens">6.5 g/cc</span></label>
        <input type="range" id="s_dens" min="2.0" max="6.5" step="0.1" value="6.5" oninput="render()">
      </div>
      <hr style="border:0; border-top:1px solid var(--border); margin:14px 0;">
      <p style="font-size:11.5px; color:var(--muted); line-height:1.5;">
        <b>Notes:</b><br>
        • Rows below threshold (50.0) are labelled <code>no anomaly</code>.<br>
        • Synthetic placeholder rows are unverified random numbers and are listed separately without rankings or application areas.
      </p>
    </div>

    <div class="card">
      <h3 style="margin:0 0 10px; font-size:14px;">Literature-Sourced Catalog (Ranked by Active Heuristic)</h3>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Material</th>
            <th>Raw Cost</th>
            <th>Poisson</th>
            <th>Heuristic Classification</th>
            <th>Possible Application Areas</th>
            <th>Heuristic Score</th>
          </tr>
        </thead>
        <tbody id="tbody-sourced"></tbody>
      </table>

      <div id="placeholder-section" style="display:none; margin-top:22px;">
        <h3 style="margin:0 0 10px; font-size:14px; color:var(--amber);">Synthetic Placeholders (Unverified Random Numbers — Unranked)</h3>
        <table>
          <thead>
            <tr>
              <th>Tag</th>
              <th>Synthetic Name</th>
              <th>Raw Cost</th>
              <th>Poisson</th>
              <th>Heuristic Classification</th>
              <th>Status</th>
              <th>Heuristic Score</th>
            </tr>
          </thead>
          <tbody id="tbody-placeholders"></tbody>
        </table>
      </div>

      <div id="inspector" class="inspector" style="display:none;"></div>
    </div>
  </div>

<script>
let compounds = [];
let currentLens = 'all';
let selected = null;

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function loadData() {
  const res = await fetch('/api/compounds');
  compounds = await res.json();
  selected = compounds[0] || null;
  render();
}

async function addPlaceholders() {
  const res = await fetch('/api/placeholders', {method: 'POST'});
  compounds = await res.json();
  render();
}

function setLens(lens, btn) {
  currentLens = lens;
  document.querySelectorAll('.lens-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
}

function render() {
  const maxCost = parseFloat(document.getElementById('s_cost').value);
  const maxDens = parseFloat(document.getElementById('s_dens').value);
  document.getElementById('v_cost').innerText = '$' + maxCost.toFixed(2) + '/kg';
  document.getElementById('v_dens').innerText = maxDens.toFixed(1) + ' g/cc';

  const sourced = compounds
    .filter(c => !c.placeholder && c.cost_per_kg <= maxCost && c.density <= maxDens)
    .map(c => ({...c, active_score: c.scores[currentLens]}))
    .sort((a, b) => b.active_score - a.active_score);

  const placeholders = compounds
    .filter(c => c.placeholder && c.cost_per_kg <= maxCost && c.density <= maxDens)
    .map(c => ({...c, active_score: c.scores[currentLens]}));

  const tbodySourced = document.getElementById('tbody-sourced');
  tbodySourced.innerHTML = '';
  sourced.forEach((c, idx) => {
    const tr = document.createElement('tr');
    tr.onclick = () => { selected = c; renderInspector(); };
    const pillClass = c.heuristic_label === 'no anomaly' ? 'pill pill-none' : 'pill';
    const appHtml = (c.possible_applications || [])
      .map(a => `<span class="app-tag">${escapeHtml(a)}</span>`)
      .join('') || '<span style="color:var(--muted)">—</span>';
    tr.innerHTML = `
      <td><b>#${idx + 1}</b></td>
      <td><b>${escapeHtml(c.name)}</b><br><span style="color:var(--muted);font-size:11px;">${escapeHtml(c.structure_class)}</span></td>
      <td>$${escapeHtml(c.cost_per_kg.toFixed(2))}</td>
      <td>${escapeHtml(c.poisson_ratio.toFixed(3))}</td>
      <td><span class="${pillClass}">${escapeHtml(c.heuristic_label)}</span></td>
      <td>${appHtml}</td>
      <td><b>${escapeHtml(c.active_score.toFixed(1))}</b></td>
    `;
    tbodySourced.appendChild(tr);
  });

  const pSec = document.getElementById('placeholder-section');
  const tbodyP = document.getElementById('tbody-placeholders');
  tbodyP.innerHTML = '';
  if (placeholders.length > 0) {
    pSec.style.display = 'block';
    placeholders.forEach(c => {
      const tr = document.createElement('tr');
      tr.onclick = () => { selected = c; renderInspector(); };
      tr.innerHTML = `
        <td><span class="pill pill-placeholder">PLACEHOLDER</span></td>
        <td><b>${escapeHtml(c.name)}</b></td>
        <td>$${escapeHtml(c.cost_per_kg.toFixed(2))}</td>
        <td>${escapeHtml(c.poisson_ratio.toFixed(3))}</td>
        <td>${escapeHtml(c.heuristic_label)}</td>
        <td style="color:var(--amber)">Unverified synthetic row (unranked)</td>
        <td>${escapeHtml(c.active_score.toFixed(1))}</td>
      `;
      tbodyP.appendChild(tr);
    });
  } else {
    pSec.style.display = 'none';
  }

  if (selected) renderInspector();
}

function renderInspector() {
  const box = document.getElementById('inspector');
  box.style.display = 'block';
  const recipe = Object.entries(selected.composition)
    .map(([k, v]) => `${escapeHtml(v)} ${escapeHtml(k)}`)
    .join(' + ');
  const propSourcesHtml = Object.entries(selected.property_sources || {})
    .map(([k, v]) => `<li><b>${escapeHtml(k)}:</b> ${escapeHtml(v)}</li>`)
    .join('');

  box.innerHTML = `
    <h3 style="margin:0 0 6px; color:#38bdf8;">
      ${selected.placeholder ? '[UNVERIFIED PLACEHOLDER] ' : ''}Material Details: ${escapeHtml(selected.name)}
    </h3>
    <div style="font-size:13px; color:#e2e8f0; margin-bottom:6px;">
      <b>Heuristic Summary:</b> ${escapeHtml(selected.heuristic_summary)}
    </div>
    <div style="font-size:12px; color:var(--muted); margin-bottom:6px;">
      <b>Formula:</b> ${escapeHtml(selected.formula)} (${recipe}) &nbsp;|&nbsp;
      <b>Primary Source:</b> ${escapeHtml(selected.source)}
    </div>
    ${propSourcesHtml ? `<ul style="font-size:11.5px; color:var(--muted); margin:6px 0 10px; padding-left:18px;">${propSourcesHtml}</ul>` : ''}
    <div class="stat-grid">
      <div class="stat"><div class="k">Raw Cost Est.</div><div class="v">$${escapeHtml(selected.cost_per_kg.toFixed(2))}/kg</div></div>
      <div class="stat"><div class="k">Density</div><div class="v">${escapeHtml(selected.density)} g/cc</div></div>
      <div class="stat"><div class="k">Est. Hv / Pugh</div><div class="v">${selected.hardness_gpa !== null && selected.hardness_gpa !== undefined ? escapeHtml(selected.hardness_gpa) + ' GPa' : 'n/a'} / ${escapeHtml(selected.pugh_ratio)}</div></div>
      <div class="stat"><div class="k">Poisson Ratio</div><div class="v">${escapeHtml(selected.poisson_ratio)}</div></div>
      <div class="stat"><div class="k">Thermal Exp (CTE)</div><div class="v">${escapeHtml(selected.thermal_expansion_ppm)} ppm/K</div></div>
      <div class="stat"><div class="k">Thermal Cond / Eg</div><div class="v">${escapeHtml(selected.thermal_cond)} W / ${escapeHtml(selected.band_gap_ev)} eV</div></div>
    </div>
  `;
}

loadData();
</script>
</body>
</html>
"""


class UniversalStudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, status_code: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/compounds":
            data = [serialize_compound(c) for c in (SOURCED_CATALOG + PLACEHOLDER_ROWS)]
            self._send_json(200, data)
        else:
            self._send_json(404, {"error": "Not found", "path": parsed.path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/placeholders", "/api/evolve"):
            remaining = max(0, MAX_PLACEHOLDER_ROWS - len(PLACEHOLDER_ROWS))
            if remaining > 0:
                batch = generate_placeholder_candidates(
                    count=min(4, remaining),
                    rng=PLACEHOLDER_RNG,
                    start_index=len(PLACEHOLDER_ROWS) + 1,
                )
                PLACEHOLDER_ROWS.extend(batch)
            data = [serialize_compound(c) for c in (SOURCED_CATALOG + PLACEHOLDER_ROWS)]
            self._send_json(200, data)
        else:
            self._send_json(404, {"error": "Not found", "path": parsed.path})


if __name__ == "__main__":
    port = 8550
    server = HTTPServer(("127.0.0.1", port), UniversalStudioHandler)
    print(f"Materials Catalog Viewer running at http://127.0.0.1:{port}")
    server.serve_forever()
