# MatEvolve-JAX (Prototype): Literature Catalog & Heuristic Scoring Demo

This repository contains a dependency-free Python prototype (`engine.py` and `app.py`) that computes derived elastic properties, raw elemental weight-fraction costs, and thresholded heuristic scores over a small, hand-entered catalog of literature-sourced inorganic materials.

Despite the repository name (`matevolve-jax`), **the current codebase uses only the Python standard library**. It does not import JAX or JAX-MD, does not load Google DeepMind's GNoME dataset, does not run on TPUs, and does not perform evolutionary search or materials discovery. Those items are future goals listed in the [Roadmap](#roadmap) below.

---

## What Exists in the Code Today

1. **Literature-Sourced Catalog (`engine.py`):**
   - A dataclass (`UniversalCompound`) storing composition (in atomic proportions), density ($\rho$), bulk modulus ($B$), shear modulus ($G$), linear coefficient of thermal expansion ($\alpha$), room-temperature thermal conductivity ($\kappa$), and electronic band gap ($E_g$).
   - Every catalog entry carries a `source` summary and a `property_sources` dictionary mapping each numeric property to a literature DOI (with table/figure/page reference) or handbook reference.
2. **Derived Isotropic Mechanics & Cost Formulas (`engine.py`):**
   - Isotropic Young's modulus: $E = \frac{9BG}{3B + G}$
   - Isotropic Poisson's ratio: $\nu = \frac{3B - 2G}{2(3B + G)}$
   - Pugh's ratio: $B / G$
   - Chen's empirical Vickers hardness estimate: $H_v = \max\left(1.0,\; 2(k^2 G)^{0.585} - 3\right)$ where $k = G / B$ (Chen et al., *Intermetallics* 19, 1275, 2011, DOI: `10.1016/j.intermet.2011.03.026`).
   - Raw elemental cost per kilogram computed from atomic masses and coarse commodity/supplier element prices.
3. **Four Thresholded Heuristic Filters (`engine.py`):**
   - `heuristic_dielectric_heat`: Scores electrical insulators ($E_g \ge 3.0\text{ eV}$) with high thermal conductivity ($\kappa \ge 50\text{ W/(m}\cdot\text{K)}$).
   - `heuristic_low_thermal_expansion`: Scores materials with negative or near-zero linear thermal expansion ($\alpha \le 2.0\text{ ppm/K}$).
   - `heuristic_damage_tolerant_hard`: Scores materials (`High-modulus moderate-Pugh phase`) with high shear stiffness ($G \ge 110.0\text{ GPa}$), moderate-to-high Pugh ratio ($B/G \ge 1.35$), and density $\rho \le 5.0\text{ g/cm}^3$ (does not use Chen's hardness estimate, which overpredicts machinable MAX phases).
   - `heuristic_auxetic`: Scores materials with a strictly negative isotropic Poisson's ratio ($\nu < 0.0$).
   - If a material does not reach the threshold (`50.0`) on any heuristic, `evaluate_heuristics()` returns `"no anomaly"` and attaches no application areas.
4. **Opt-In Seeded Placeholder Generator (`engine.py`):**
   - `generate_placeholder_candidates(count, rng)` generates synthetic, unverified test rows using a caller-supplied `random.Random` instance.
   - Placeholder rows are flagged with `placeholder=True`, carry no literature citations or application areas, are never ranked alongside sourced rows, and are only printed on the CLI when `--placeholders N` is passed.
5. **Local HTTP Viewer (`app.py`):**
   - Serves an HTML table at `http://127.0.0.1:8550` displaying the sourced catalog and (in a separate unranked table) up to 20 opt-in placeholder rows.
6. **Unit Test Suite & DOI Verification (`test_engine.py`, `tools/check_dois.py`):**
   - Standard-library `unittest` suite covering formula cost calculations, Ti-6Al-4V weight fractions, elastic identities, element table coverage, deterministic placeholder generation, baseline non-anomaly classification, HTTP status/cap behavior, and live DOI resolution against Crossref and DataCite (skipped automatically when offline).

---

## Known Limitations

- **Coarse Elemental Cost Model:** `raw_cost_per_kg` multiplies atomic weight fractions by a static dictionary of approximate elemental prices (`ELEMENT_COST_USD_PER_KG`). It ignores the chemical form in which an element is actually purchased (for example, boric acid or boron oxide vs. high-purity amorphous elemental boron powder at ~$300/kg), purity requirements, and processing/sintering costs.
- **Chen's Hardness Formula on Nanolaminates & Metals:** Chen's empirical correlation ($H_v$ from $B$ and $G$) was fit to brittle covalent and ionic crystals. In layered MAX/MAB phases (such as $\text{Ti}_3\text{SiC}_2$) and ductile metals, basal dislocation slip and kink-band formation yield much lower experimental indentation hardness (~4–5 GPa for $\text{Ti}_3\text{SiC}_2$) than the purely elastic estimate (~22.5 GPa) predicted from $B$ and $G$.
- **Isotropic Polycrystalline Averages:** Single-crystal elastic and thermal expansion anisotropy is collapsed into scalar Voigt–Reuss–Hill or polycrystalline averages.
- **Hand-Entered Catalog Size:** The repository currently evaluates 7 literature-sourced entries rather than an external database.

---

## Quickstart

Run the CLI table (sourced catalog only):

```bash
python3 engine.py
```

Output of `python3 engine.py`:

```text
======================================================================================================================
  LITERATURE-SOURCED MATERIALS CATALOG: DERIVED PROPERTIES & HEURISTIC SCORES
======================================================================================================================
Rank | Material                                   | Cost/kg   | Poisson | Top Heuristic Classification         | Score
----------------------------------------------------------------------------------------------------------------------
#1   | AlN (Wurtzite Aluminum Nitride)            | $   1.72 |  +0.233 | Dielectric thermal conductor         |  97.9
     └─ Possible application areas: Electronic packaging substrates, Power-electronics heat spreaders
     └─ Source: Slack et al., J. Phys. Chem. Solids 48(7), 641-647 (1987), DOI:10.1016/0022-3697(87)90153-3; McNeil et al., J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x
#2   | ZrW2O8 (Alpha-Zirconium Tungstate)         | $  28.52 |  +0.303 | Negative / low thermal expansion     |  94.6
     └─ Possible application areas: Athermal optical mounts, Low-expansion composite fillers
     └─ Source: Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; Kennedy & White, Solid State Commun. 134(4), 271-276 (2005), DOI:10.1016/j.ssc.2005.01.031
#3   | SiO2 (Alpha-Cristobalite)                  | $   1.13 |  -0.164 | Auxetic (negative Poisson ratio)     |  82.8
     └─ Possible application areas: Indentation-resistant pads, Acoustic/mechanical dampers
     └─ Source: Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650; Swainson & Dove, Phys. Chem. Miner. 22, 61-65 (1995), DOI:10.1007/BF00202681
#4   | MgSiN2 (Magnesium Silicon Nitride)         | $   1.84 |  +0.239 | High-modulus moderate-Pugh phase     |  62.4
     └─ Possible application areas: Thermal-shock-resistant structural parts, Machinable high-temperature components
     └─ Source: Bruls et al., J. Phys. Chem. Solids 61(8), 1285-1292 (2000), DOI:10.1016/S0022-3697(99)00416-3; Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7; de Boer et al., Phys. Status Solidi RRL 9(4), 250-254 (2015), DOI:10.1002/pssr.201510043
#5   | Ti3SiC2 (Titanium Silicon Carbide MAX Phas | $   8.94 |  +0.204 | High-modulus moderate-Pugh phase     |  62.3
     └─ Possible application areas: Thermal-shock-resistant structural parts, Machinable high-temperature components
     └─ Source: Barsoum, Prog. Solid State Chem. 28, 201-281 (2000), DOI:10.1016/S0079-6786(00)00006-6; Finkel, Barsoum & El-Raghy, J. Appl. Phys. 87(4), 1701-1703 (2000), DOI:10.1063/1.372080; Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8
#6   | Ti-6Al-4V (Grade 5 Titanium Alloy Baseline | $  11.61 |  +0.324 | no anomaly                           |   0.0
     └─ Source: ASM Aerospace Specification Metals (MTP641); Boyer et al., Materials Properties Handbook: Titanium Alloys (ASM International, 1994)
#7   | B4C (Boron Carbide Ceramic Baseline)       | $ 235.11 |  +0.184 | no anomaly                           |   0.0
     └─ Source: Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), DOI:10.1111/j.1551-2916.2011.04865.x; Dodd, Saunders & James, J. Mater. Sci. 37(13), 2731-2736 (2002), DOI:10.1023/A:1015825318086; Werheit, J. Phys.: Condens. Matter 18(47), 10655-10662 (2006), DOI:10.1088/0953-8984/18/47/011
======================================================================================================================
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
