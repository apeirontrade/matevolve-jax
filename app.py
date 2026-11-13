#!/usr/bin/env python3
"""
MatEvolve-JAX Universal Anomaly & Compound Discovery Studio
Industry-agnostic web studio (http://localhost:8550) that:
  - Scans compounds across 12 fundamental physics properties
  - Automatically flags "Freaks of Nature" (Physics Trade-Off Breakers)
  - Maps each discovered anomaly to the industries that need it
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from engine import (
    UniversalCompound,
    get_universal_catalog,
    breed_universal_anomalies_rsi,
)

LIVE_CATALOG = get_universal_catalog()
GEN_COUNTER = 0


def serialize_compound(c: UniversalCompound) -> dict:
    title, score, industries, reason = c.detect_breakthrough_profile()
    return {
        "name": c.name,
        "formula": c.formula,
        "structure_class": c.structure_class,
        "composition": c.composition,
        "cost_per_kg": round(c.raw_cost_per_kg, 2),
        "density": round(c.density_g_cm3, 2),
        "hardness_gpa": round(c.vickers_hardness_gpa, 1),
        "pugh_ratio": round(c.pugh_ratio, 2),
        "poisson_ratio": round(c.poisson_ratio, 2),
        "stiffness_gpa": round(c.youngs_modulus_gpa, 1),
        "thermal_expansion_ppm": round(c.thermal_expansion_ppm, 2),
        "thermal_cond": round(c.thermal_cond_w_mk, 1),
        "band_gap_ev": round(c.band_gap_ev, 2),
        "ionic_cond": round(c.ionic_cond_ms_cm, 1),
        "seebeck": round(c.seebeck_uv_k, 1),
        "surface_idx": round(c.surface_repulsion_idx, 1),
        "anomaly_title": title,
        "anomaly_score": round(score, 1),
        "industries": industries,
        "why_weird": reason,
        "scores": {
            "all": round(score, 1),
            "dielectric_heat": round(c.anomaly_dielectric_heat, 1),
            "tough_superhard": round(c.anomaly_tough_superhard, 1),
            "zero_thermal": round(c.anomaly_zero_negative_thermal, 1),
            "energy_quantum": round(c.anomaly_energy_quantum, 1),
            "auxetic_shock": round(c.anomaly_auxetic_shock, 1),
            "pfas_free": round(c.anomaly_surface_pfas_free, 1),
        },
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MatEvolve-JAX | Universal Physics Anomaly & Compound Discovery Studio</title>
<style>
  :root {
    --bg: #090d16;
    --panel: #111827;
    --border: #1f2937;
    --accent: #38bdf8;
    --purple: #a855f7;
    --green: #22c55e;
    --amber: #f59e0b;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { margin: 0; background: var(--bg); color: var(--text); padding: 22px; }
  .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 18px; }
  .header h1 { margin: 0; font-size: 21px; color: #fff; }
  .header p { margin: 5px 0 0; color: var(--muted); font-size: 13.5px; }
  .btn-evolve {
    background: linear-gradient(135deg, #2563eb, #9333ea);
    color: #fff; border: none; padding: 12px 20px; border-radius: 8px;
    font-weight: 600; cursor: pointer; font-size: 13.5px; box-shadow: 0 4px 16px rgba(147,51,234,0.35);
  }
  .btn-evolve:hover { opacity: 0.92; transform: translateY(-1px); }
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
  td { padding: 11px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); cursor: pointer; vertical-align: middle; }
  tr:hover td { background: rgba(56, 189, 248, 0.07); }
  .pill { display: inline-block; padding: 3px 8px; border-radius: 99px; font-size: 11px; font-weight: 600; background: rgba(168,85,247,0.18); color: #e9d5ff; }
  .ind-tag { display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 11px; background: rgba(56,189,248,0.12); color: #7dd3fc; margin: 2px 3px 2px 0; }
  .inspector { margin-top: 16px; background: #0c1322; border: 1px solid var(--accent); border-radius: 10px; padding: 16px; }
  .stat-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-top: 12px; }
  .stat { background: var(--panel); padding: 9px; border-radius: 8px; border: 1px solid var(--border); }
  .stat .k { font-size: 10.5px; color: var(--muted); }
  .stat .v { font-size: 14px; font-weight: 700; margin-top: 3px; color: #fff; }
</style>
</head>
<body>
  <div class="header">
    <div>
      <h1>🌌 MatEvolve-JAX: Universal "Freak of Nature" & Anomaly Discovery Engine</h1>
      <p>No pre-picked industry required: scans fundamental physics laws to find compounds that break classical trade-offs, then maps who needs them.</p>
    </div>
    <button class="btn-evolve" onclick="runEvolution()">⚡ Run Unsupervised RSI Evolution (+4 Anomalies)</button>
  </div>

  <div class="layout">
    <div class="card">
      <h3 style="margin:0 0 10px; font-size:14px;">🔭 Discovery Lens (Physics Paradoxes)</h3>
      <button class="lens-btn active" onclick="setLens('all', this)">🚨 All Physics Trade-Off Breakers (Auto)</button>
      <button class="lens-btn" onclick="setLens('dielectric_heat', this)">💎 Insulator That Conducts Heat</button>
      <button class="lens-btn" onclick="setLens('zero_thermal', this)">❄️ Zero / Negative Thermal Expansion</button>
      <button class="lens-btn" onclick="setLens('tough_superhard', this)">🛡️ Superhard Yet Ductile (No Shatter)</button>
      <button class="lens-btn" onclick="setLens('energy_quantum', this)">🔋 Fast-Ion & Waste-Heat Harvesters</button>
      <button class="lens-btn" onclick="setLens('pfas_free', this)">💧 PFAS-Free Non-Stick / Superlubricity</button>

      <div class="slider-group">
        <label><span>Max Ingredient Cost ($/kg)</span><span id="v_cost">$12.00/kg</span></label>
        <input type="range" id="s_cost" min="1.0" max="15.0" step="0.5" value="12.0" oninput="render()">
      </div>
      <div class="slider-group">
        <label><span>Max Weight / Density (g/cm³)</span><span id="v_dens">6.5 g/cc</span></label>
        <input type="range" id="s_dens" min="2.2" max="6.5" step="0.1" value="6.5" oninput="render()">
      </div>
      <hr style="border:0; border-top:1px solid var(--border); margin:14px 0;">
      <p style="font-size:11.5px; color:var(--muted); line-height:1.5;">
        <b>How Unsupervised Discovery Works:</b><br>
        Instead of locking into one industry, the engine tests <b>12 fundamental physics properties</b> at once and flags compounds that break normal rules of nature—then automatically lists which industries will pay for them.
      </p>
    </div>

    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Compound</th>
            <th>Cost/kg</th>
            <th>Physics Trade-Off Broken ("Freak Property")</th>
            <th>Industries Unlocked Automatically</th>
            <th>Anomaly Score</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>

      <div id="inspector" class="inspector" style="display:none;"></div>
    </div>
  </div>

<script>
let compounds = [];
let currentLens = 'all';
let selected = null;

async function loadData() {
  const res = await fetch('/api/compounds');
  compounds = await res.json();
  selected = compounds[0];
  render();
}

async function runEvolution() {
  const res = await fetch('/api/evolve', {method: 'POST'});
  compounds = await res.json();
  selected = compounds[compounds.length - 1];
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

  const filtered = compounds
    .filter(c => c.cost_per_kg <= maxCost && c.density <= maxDens)
    .map(c => ({...c, active_score: c.scores[currentLens]}))
    .sort((a, b) => b.active_score - a.active_score);

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  filtered.forEach((c, idx) => {
    const tr = document.createElement('tr');
    tr.onclick = () => { selected = c; renderInspector(); };
    const indHtml = c.industries.map(ind => `<span class="ind-tag">${ind}</span>`).join('');
    tr.innerHTML = `
      <td><b>#${idx+1}</b></td>
      <td><b>${c.name}</b><br><span style="color:var(--muted);font-size:11px;">${c.structure_class}</span></td>
      <td style="color:#86efac; font-weight:600;">$${c.cost_per_kg.toFixed(2)}</td>
      <td><span class="pill">${c.anomaly_title}</span></td>
      <td>${indHtml}</td>
      <td><b style="color:#38bdf8; font-size:14px;">${c.active_score.toFixed(1)}</b></td>
    `;
    tbody.appendChild(tr);
  });
  if (selected) renderInspector();
}

function renderInspector() {
  const box = document.getElementById('inspector');
  box.style.display = 'block';
  const recipe = Object.entries(selected.composition).map(([k,v]) => `${v} ${k}`).join(' + ');
  box.innerHTML = `
    <h3 style="margin:0 0 6px; color:#38bdf8;">🔬 12-Property Physics Fingerprint: ${selected.name}</h3>
    <div style="font-size:13px; color:#e2e8f0; margin-bottom:6px;">
      <b>Why It's a Physics Anomaly:</b> ${selected.why_weird}
    </div>
    <div style="font-size:12px; color:var(--muted);">
      <b>Atomic Formula:</b> ${selected.formula} (${recipe}) &nbsp;|&nbsp;
      <b>Unlocked Markets:</b> ${selected.industries.join(' • ')}
    </div>
    <div class="stat-grid">
      <div class="stat"><div class="k">Raw Cost</div><div class="v" style="color:#22c55e">$${selected.cost_per_kg.toFixed(2)}/kg</div></div>
      <div class="stat"><div class="k">Density (Weight)</div><div class="v">${selected.density} g/cc</div></div>
      <div class="stat"><div class="k">Hardness / Pugh</div><div class="v">${selected.hardness_gpa} GPa / ${selected.pugh_ratio}</div></div>
      <div class="stat"><div class="k">Thermal Expansion</div><div class="v">${selected.thermal_expansion_ppm > 0 ? '+' : ''}${selected.thermal_expansion_ppm} ppm/K</div></div>
      <div class="stat"><div class="k">Heat Cond / BandGap</div><div class="v">${selected.thermal_cond} W / ${selected.band_gap_ev} eV</div></div>
      <div class="stat"><div class="k">Ion / Seebeck / Slip</div><div class="v">${selected.ionic_cond} mS | ${selected.surface_idx}%</div></div>
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

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/compounds":
            data = [serialize_compound(c) for c in LIVE_CATALOG]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        global GEN_COUNTER
        if self.path == "/api/evolve":
            GEN_COUNTER += 1
            LIVE_CATALOG.extend(breed_universal_anomalies_rsi(GEN_COUNTER, count=4))
            data = [serialize_compound(c) for c in LIVE_CATALOG]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))


if __name__ == "__main__":
    port = 8550
    server = HTTPServer(("127.0.0.1", port), UniversalStudioHandler)
    print(f"MatEvolve-JAX Universal Anomaly Studio running at http://localhost:{port}")
    server.serve_forever()
