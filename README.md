# MatEvolve-JAX (Prototype): Literature Catalog & Heuristic Scoring Demo

This repository contains a dependency-free Python prototype (`engine.py` and `app.py`) that computes derived elastic properties, raw elemental weight-fraction costs, and thresholded heuristic scores over a small, hand-entered catalog of literature-sourced inorganic materials.

Despite the repository name (`matevolve-jax`), **the current codebase uses only the Python standard library**. It does not import JAX or JAX-MD, does not load Google DeepMind's GNoME dataset, does not run on TPUs, and does not perform evolutionary search or materials discovery. Those items are future goals listed in the [Roadmap](#roadmap) below.

---

## What Exists in the Code Today

1. **Literature-Sourced Catalog (`engine.py`):**
   - A dataclass (`MaterialEntry`) storing composition (in atomic proportions), density ($\rho$), bulk modulus ($B$), shear modulus ($G$), linear coefficient of thermal expansion ($\alpha$), room-temperature thermal conductivity ($\kappa$), and electronic band gap ($E_g$).
   - Every catalog entry carries a `source` summary and a `property_sources` dictionary mapping each numeric property to a literature DOI (with table/figure/page reference) or handbook reference.
2. **Derived Isotropic Mechanics & Cost Formulas (`engine.py`):**
   - Isotropic Young's modulus: $E = \frac{9BG}{3B + G}$
   - Isotropic Poisson's ratio: $\nu = \frac{3B - 2G}{2(3B + G)}$
   - Pugh's ratio: $B / G$
   - Chen's empirical Vickers hardness estimate: $H_v = 2(k^2 G)^{0.585} - 3$ where $k = G / B$ (Chen et al., *Intermetallics* 19, 1275, 2011, DOI: `10.1016/j.intermet.2011.03.026`). Returns `None` (`n/a`) when $G/B > 1$ or $\nu < 0$ (outside the valid domain of Chen's correlation, such as auxetic $\alpha$-cristobalite).
   - Raw elemental cost per kilogram computed from atomic masses (CIAAW 2024, DOI: `10.1515/pac-2019-0603`) and coarse commodity/nominal element prices (`ELEMENT_COST_USD_PER_KG`, documented in `ELEMENT_COST_SOURCES` with USGS Mineral Commodity Summaries 2026, DOI: `10.3133/mcs2026`, or `UNSOURCED` tags). Cost is for display only and does not feed any heuristic.
3. **Four Heuristic Filters (`engine.py`):**
   - `heuristic_dielectric_heat`: Scores electrical insulators ($E_g \ge 3.0\text{ eV}$) with high thermal conductivity ($\kappa \ge 50\text{ W/(m}\cdot\text{K)}$).
   - `heuristic_elastic_screen`: Screen labeled `Elastic screen: Hv estimate >= 15, B/G >= 1.35, density <= 5` requiring Chen's hardness estimate $H_v \ge 15.0\text{ GPa}$, Pugh ratio $B/G \ge 1.35$, and density $\rho \le 5.0\text{ g/cm}^3$. Explicitly returns `0.0` for MAX phases, MAB phases, and metallic alloys (where Chen's brittle-crystal correlation is invalid) and attaches no application areas.
   - `heuristic_low_thermal_expansion`: Scores materials with linear thermal expansion $\alpha \le 2.0\text{ ppm/K}$ (linearly rewarding increasingly negative thermal expansion below $+2.0\text{ ppm/K}$ rather than closeness to zero).
   - `heuristic_auxetic`: Scores materials with a strictly negative isotropic Poisson's ratio ($\nu < 0.0$; Evans, *Endeavour* 15, 170, 1991, DOI: `10.1016/0160-9327(91)90123-S`).
   - Each heuristic either passes its documented property cut-offs (producing a score in 50–100) or does not (returning `0.0`). When no heuristic cut-off is met, `evaluate_heuristics()` returns `"no anomaly"` (`no cut-off met`) and attaches no application areas.
   - **Design-Choice Cut-Offs:** The numerical cut-offs ($B/G \ge 1.35$, $H_v \ge 15\text{ GPa}$, $\rho \le 5.0\text{ g/cm}^3$, $E_g \ge 3.0\text{ eV}$, $\kappa \ge 50\text{ W/(m}\cdot\text{K)}$, $\alpha \le 2.0\text{ ppm/K}$) and linear scaling anchors are prototype design choices with no literature source.
4. **Opt-In Seeded Placeholder Generator (`engine.py`):**
   - `generate_placeholder_candidates(count, rng)` generates up to 20 synthetic, unverified test rows using a caller-supplied `random.Random` instance.
   - Placeholder rows are flagged with `placeholder=True`, are not evaluated by the heuristics (`evaluate_heuristics()` returns `"not evaluated (placeholder)"`, score `None`, and no application areas), are never ranked alongside sourced rows, and are only printed on the CLI when `--placeholders N` ($0 \le N \le 20$) is passed.
5. **Local HTTP Viewer (`app.py`):**
   - Serves an HTML table at `http://127.0.0.1:8550` (`CatalogHandler`) displaying the sourced catalog (grouped by heuristic cut-off) and, in a separate unranked table, up to 20 opt-in placeholder rows.
6. **Unit Test Suite & DOI Verification (`test_engine.py`, `tools/check_dois.py`):**
   - Standard-library `unittest` suite covering formula cost calculations, Ti-6Al-4V weight fractions, elastic identities, element and citation source coverage (`DOI`, `URL`, `mp-id`, or `UNSOURCED`), input validation in `MaterialEntry.__post_init__`, deterministic un-evaluated placeholder generation, MAX/MAB exclusion from the hardness screen, `None` hardness for $\alpha$-cristobalite, baseline non-anomaly classification, HTTP status/cap behavior, and live DOI resolution against Crossref and DataCite (skipped automatically when offline).

---

## Known Limitations

- **Coarse Elemental Cost Model:** `raw_cost_per_kg` multiplies atomic weight fractions by a static dictionary of approximate elemental prices (`ELEMENT_COST_USD_PER_KG`). It ignores the chemical form in which an element is actually purchased (for example, refined borates priced at ~$0.80–0.81/kg in USGS MCS 2026 vs. the unsourced $300/kg nominal figure for elemental boron in the table), purity requirements, and processing/sintering costs.
- **Chen's Hardness Formula on Nanolaminates, Metals & Auxetics:** Chen's empirical correlation ($H_v$ from $B$ and $G$) was fit to brittle covalent and ionic crystals. In layered MAX/MAB phases (such as $\text{Ti}_3\text{SiC}_2$) and ductile metals, basal dislocation slip and kink-band formation yield much lower experimental indentation hardness (~4–5 GPa for $\text{Ti}_3\text{SiC}_2$) than the purely elastic formula (`22.0 GPa` at $G = 138\text{ GPa}$, or `23.3 GPa` at $G = 142\text{ GPa}$) calculated from $B$ and $G$; `heuristic_elastic_screen` therefore excludes MAX/MAB phases and metals, and `vickers_hardness_gpa` returns `None` when $G/B > 1$ or $\nu < 0$.
- **Isotropic Polycrystalline Averages:** Single-crystal elastic and thermal expansion anisotropy is collapsed into scalar Voigt–Reuss–Hill or polycrystalline averages.
- **Hand-Entered Catalog Size:** The repository currently evaluates 9 literature-sourced entries rather than an external database.

---

## Quickstart

Run the CLI table (sourced catalog only):

```bash
python3 engine.py
```

Output of `python3 engine.py`:

```text
========================================================================================================================
  LITERATURE-SOURCED MATERIALS CATALOG: DERIVED PROPERTIES & HEURISTIC CUT-OFFS
========================================================================================================================
Material                                     | Cost/kg   | Poisson | Est. Hv | Top Heuristic Classification         | Score
------------------------------------------------------------------------------------------------------------------------
[Rows passing a documented heuristic cut-off (grouped by heuristic)]
AlN (Wurtzite Aluminum Nitride)              | $   1.72 |  +0.240 |   17.4G | Dielectric thermal conductor         |  97.9
     +- Possible application areas: Electronic packaging substrates, Power-electronics heat spreaders
     +- Source: McNeil, Grimsditch & French, J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x, as tabulated in Reeber & Wang, MRS Internet J. Nitride Semicond. Res. 6, e3 (2001), DOI:10.1557/S1092578300000156; Slack et al., J. Phys. Chem. Solids 48(7), 641-647 (1987), DOI:10.1016/0022-3697(87)90153-3, as cited in Bruls, PhD Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906; Yim & Paff, J. Appl. Phys. 45(3), 1456-1457 (1974), DOI:10.1063/1.1663432; Yim et al., J. Appl. Phys. 44(1), 292-296 (1973), DOI:10.1063/1.1661876
ZrW2O8 (Alpha-Zirconium Tungstate)           | $  28.52 |  +0.303 |    3.2G | Negative / low thermal expansion     |  94.6
     +- Possible application areas: Low-expansion composite fillers
     +- Source: Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; Kennedy & White, Solid State Commun. 134(4), 271-276 (2005), DOI:10.1016/j.ssc.2005.01.031; Kennedy, PhD Thesis, Dalhousie Univ. (2005), http://hdl.handle.net/10222/54702; Ouyang, Xu & Ching, Phys. Rev. B 65, 113110 (2002), DOI:10.1103/PhysRevB.65.113110
SiO2 (Alpha-Cristobalite)                    | $   1.13 |  -0.164 |     n/a | Auxetic (negative Poisson ratio)     |  82.8
     +- Possible application areas: Indentation-resistant pads, Acoustic/mechanical dampers
     +- Source: Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650, as tabulated at http://www.numericana.com/answer/elastic.htm; Pluth, Smith & Faber, J. Appl. Phys. 57, 1045-1049 (1985), DOI:10.1063/1.334545; Kunugi et al., J. Am. Ceram. Soc. 55, 580 (1972), DOI:10.1111/j.1151-2916.1972.tb13442.x, as tabulated in Mizokami, Togo & Tanaka, Phys. Rev. B 97, 224306 (2018), DOI:10.1103/PhysRevB.97.224306; Milton et al., Materials 16, 1382 (2023), DOI:10.3390/ma16041382
------------------------------------------------------------------------------------------------------------------------
[no cut-off met]
Ti-6Al-4V (Grade 5 Titanium Alloy Baseline)  | $  11.61 |  +0.324 |    3.3G | no anomaly                           |   0.0
     +- Source: ASM Aerospace Specification Metals Inc., Titanium Ti-6Al-4V (Grade 5) Annealed Datasheet, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf; Driver et al., Philos. Mag. 96(22), 2311-2327 (2016), DOI:10.1080/14786435.2016.1198054
B4C (Boron Carbide Ceramic Baseline)         | $ 235.11 |  +0.181 |   31.7G | no anomaly                           |   0.0
     +- Source: Gieske, Aselage & Emin, AIP Conf. Proc. 231, 376-379 (1991), DOI:10.1063/1.40854, as tabulated in Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), DOI:10.1111/j.1551-2916.2011.04865.x; Yakel, J. Appl. Cryst. 6, 471-472 (1973), DOI:10.1107/S0021889873009246; Mahagin et al., HEDL-TME 73-78 (1973), DOI:10.2172/4389842, https://www.osti.gov/servlets/purl/4389842; Werheit, J. Phys.: Condens. Matter 18(47), 10655-10662 (2006), DOI:10.1088/0953-8984/18/47/011
Ti3SiC2 (Titanium Silicon Carbide MAX Phase) | $   8.94 |  +0.197 |   23.3G | no anomaly                           |   0.0
     +- Source: Finkel, Barsoum & El-Raghy, J. Appl. Phys. 87(4), 1701-1703 (2000), DOI:10.1063/1.372080, as tabulated in Ganguly, PhD Thesis, Drexel Univ. (2006), Table 3.4 p. 61, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/Adrish_Ganguly.pdf; Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8
Cr2AlC (Chromium Aluminum Carbide MAX Phase) | $   6.76 |  +0.197 |   19.1G | no anomaly                           |   0.0
     +- Source: Hettinger et al., Phys. Rev. B 72, 115120 (2005), DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf; Scabarozi et al., J. Appl. Phys. 105, 013543 (2009), DOI:10.1063/1.3021465
Fe2AlB2 (Iron Aluminum Boride MAB Phase)     | $  41.21 |  +0.228 |   16.1G | no anomaly                           |   0.0
     +- Source: Bai et al., J. Am. Ceram. Soc. 103(10), 5837-5851 (2020), DOI:10.1111/jace.17205; Lamichhane et al., Phys. Rev. Materials 2, 084408 (2018), DOI:10.1103/PhysRevMaterials.2.084408, https://arxiv.org/pdf/1805.06373; Verger et al., J. Appl. Phys. 124, 205108 (2018), DOI:10.1063/1.5054379; Jeitschko, Acta Cryst. B 25, 163 (1969), DOI:10.1107/S0567740869001944
MgSiN2 (Magnesium Silicon Nitride)           | $   1.84 |  +0.246 |   14.8G | no anomaly                           |   0.0
     +- Source: Bruls, Ph.D. Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf; Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7; Bruls et al., J. Phys. Chem. Solids 61(8), 1285-1292 (2000), DOI:10.1016/S0022-3697(99)00416-3; de Boer et al., Phys. Status Solidi RRL 9(4), 250-254 (2015), DOI:10.1002/pssr.201510043
========================================================================================================================
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
