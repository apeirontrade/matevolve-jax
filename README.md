# MatEvolve-JAX: Multi-Objective Discovery of Low-Cost Structural Hybrids & Composites

**MatEvolve-JAX** is an open-source computational materials discovery and Recursive Self-Improvement (RSI) screening pipeline designed for **Google Cloud TPUs** (`JAX`, `JAX-MD`, and `FunSearch`-style evolutionary search).

It screens and evolves inorganic crystal structures (starting from Google DeepMind's **GNoME** 380,000+ stable materials catalog and **MAX/MAB-phase atomic hybrids**) across four practical, non-medical manufacturing objectives:

1. **Ultra-Low Raw Material Cost ($\$/\text{kg}$):** Prioritizes earth-abundant, supply-chain-safe elements ($\text{Fe, Al, Si, B, C, N, Mg, Mn, Ca}$) over expensive rare-earth or precious metals.
2. **Additive Manufacturing (3D Printing) Printability:** Evaluates thermal shock resistance and solidification cracking susceptibility for laser powder-bed fusion (LPBF) and nano-reinforced composite resins.
3. **Ballistic & Impact Toughness (Hardness + Ductility):** Identifies "MAX/MAB-phase" atomic hybrids and cermet reinforcements that break the traditional hardness-brittleness trade-off (high Vickers hardness $H_v$ paired with ductile Pugh's ratio $B/G$).
4. **Vehicle Lightweighting (Specific Stiffness & Strength):** Maximizes stiffness-to-weight ($E/\rho$) and strength-to-weight ratios for structural automotive and aerospace frames.

---

## Architecture

- ** vectorized Multi-Objective Evaluator (`engine.py`):** Computes Pareto-optimal frontiers across cost, density, bulk/shear moduli ($B, G$), Vickers hardness ($H_v$), Pugh's ductility ratio ($B/G$), and thermal shock resistance.
- **Evolutionary RSI Loop (`FunSearch`-style Mutator):** Iteratively mutates stoichiometry, crystal layering motifs, and scoring heuristics to discover novel low-cost quaternary and ternary hybrid candidates.
- **TPU Scaling (`JAX` / `vmap` / `pmap`):** Structured for high-throughput batched evaluation and `JAX-MD` molecular dynamics validation on Google Cloud TPU v4/v5e pods via the **TPU Research Cloud (TRC)**.

## Quickstart

```bash
python3 engine.py
```

## License

MIT License (Open Science release in accordance with Google TPU Research Cloud guidelines).
