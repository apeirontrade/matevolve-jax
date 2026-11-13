# MatEvolve-JAX (Prototype): Literature Catalog & Heuristic Scoring Demo

This repository contains a dependency-free Python prototype (`engine.py` and `app.py`) that computes derived elastic properties, raw elemental weight-fraction costs, and thresholded heuristic scores over a small, hand-entered catalog of literature-sourced inorganic materials.

Despite the repository name (`matevolve-jax`), **the current codebase uses only the Python standard library**. It does not import JAX or JAX-MD, does not load Google DeepMind's GNoME dataset, does not run on TPUs, and does not perform evolutionary search or materials discovery. Those items are future goals listed in the [Roadmap](#roadmap) below.

---

## What Exists in the Code Today

1. **Literature-Sourced Catalog (`engine.py`):**
   - A dataclass (`MaterialEntry`) storing composition (in atomic proportions), density ($\rho$), bulk modulus ($B$), shear modulus ($G$), linear coefficient of thermal expansion ($\alpha$), room-temperature thermal conductivity ($\kappa$), and electronic band gap ($E_g$).
   - Every catalog entry carries a `source` summary and a `property_sources` dictionary mapping each numeric property to a literature DOI or handbook reference.
2. **Derived Isotropic Mechanics & Cost Formulas (`engine.py`):**
   - Isotropic Young's modulus: $E = \frac{9BG}{3B + G}$
   - Isotropic Poisson's ratio: $\nu = \frac{3B - 2G}{2(3B + G)}$
   - Pugh's ratio: $B / G$
   - Chen's empirical Vickers hardness estimate: $H_v = \max\left(1.0,\; 2(k^2 G)^{0.585} - 3\right)$ where $k = G / B$ (Chen et al., *Intermetallics* 19, 1275, 2011, DOI: `10.1016/j.intermet.2011.03.026`).
   - Raw elemental cost per kilogram computed from atomic masses and coarse commodity/supplier element prices.
3. **Four Thresholded Heuristic Filters (`engine.py`):**
   - `heuristic_dielectric_heat`: Scores electrical insulators ($E_g \ge 3.0\text{ eV}$) with high thermal conductivity ($\kappa \ge 50\text{ W/(m}\cdot\text{K)}$).
   - `heuristic_low_thermal_expansion`: Scores materials with negative or near-zero linear thermal expansion ($|\alpha| \le 2.0\text{ ppm/K}$ or $\alpha < 0$).
   - `heuristic_damage_tolerant_hard`: Scores materials with high elastic hardness estimate ($H_v \ge 15.0\text{ GPa}$), moderate-to-high Pugh ratio ($B/G \ge 1.35$), and density $\rho \le 5.0\text{ g/cm}^3$.
   - `heuristic_auxetic`: Scores materials with a strictly negative isotropic Poisson's ratio ($\nu < 0.0$).
   - If a material does not reach the threshold (`50.0`) on any heuristic, `evaluate_heuristics()` returns `"no anomaly"` and attaches no application areas.
4. **Opt-In Seeded Placeholder Generator (`engine.py`):**
   - `generate_placeholder_candidates(count, rng)` generates synthetic, unverified test rows using a caller-supplied `random.Random` instance.
   - Placeholder rows are flagged with `placeholder=True`, carry no literature citations or application areas, are never ranked alongside sourced rows, and are only printed on the CLI when `--placeholders N` is passed.
5. **Local HTTP Viewer (`app.py`):**
   - Serves an HTML table at `http://127.0.0.1:8550` displaying the sourced catalog and (in a separate unranked table) up to 20 opt-in placeholder rows.
6. **Unit Test Suite (`test_engine.py`):**
   - Standard-library `unittest` suite covering formula cost calculations, Ti-6Al-4V weight fractions, elastic identities, element table coverage, deterministic placeholder generation, baseline non-anomaly classification, and HTTP status/cap behavior.

---

## Known Limitations

- **Coarse Elemental Cost Model:** `raw_cost_per_kg` multiplies atomic weight fractions by a static dictionary of approximate elemental prices (`ELEMENT_COST_USD_PER_KG`). It ignores the chemical form in which an element is actually purchased (for example, boric acid or boron oxide vs. high-purity amorphous elemental boron powder at ~$300/kg), purity requirements, and processing/sintering costs.
- **Chen's Hardness Formula on Nanolaminates & Metals:** Chen's empirical correlation ($H_v$ from $B$ and $G$) was fit to brittle covalent and ionic crystals. In layered MAX/MAB phases (such as $\text{Ti}_3\text{SiC}_2$) and ductile metals, basal dislocation slip and kink-band formation yield much lower experimental indentation hardness (~4–5 GPa for $\text{Ti}_3\text{SiC}_2$) than the purely elastic estimate (~22.5 GPa) predicted from $B$ and $G$.
- **Isotropic Polycrystalline Averages:** Single-crystal elastic and thermal expansion anisotropy is collapsed into scalar Voigt–Reuss–Hill or polycrystalline averages.
- **Hand-Entered Catalog Size:** The repository currently evaluates 9 literature-sourced entries rather than an external database.

---

## Quickstart

Run the CLI table (sourced catalog only):

```bash
python3 engine.py
```

Optionally print a separate table of deterministic synthetic placeholder rows:

```bash
python3 engine.py --placeholders 4 --seed 42
```

Run the unit test suite:

```bash
python3 -m unittest discover -v
```

Start the local web viewer on `http://127.0.0.1:8550`:

```bash
python3 app.py
```

---

## Roadmap (Planned, Not Yet Implemented)

- **External Dataset Ingestion:** Load crystal structures and DFT elastic tensors from the Materials Project API and Google DeepMind's GNoME dataset.
- **Property Prediction & Force Fields:** Integrate pre-trained universal interatomic potentials and `JAX-MD` for finite-temperature mechanical and thermal trajectories.
- **Evolutionary Search:** Implement a closed-loop mutation, structure-relaxation, and selection pipeline (`FunSearch`-style search) backed by physical energy evaluators rather than random number generation.
- **Cloud TPU Execution:** Port batched evaluations to `JAX` (`jit`/`vmap`) for parallel execution on Cloud TPUs.

---

## License

MIT License. See [LICENSE](LICENSE).
