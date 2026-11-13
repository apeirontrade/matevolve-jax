# MatEvolve-JAX: Universal Physics Anomaly & Compound Discovery Engine

**MatEvolve-JAX** is an industry-agnostic computational materials discovery and Recursive Self-Improvement (RSI) screening pipeline designed for **Google Cloud TPUs** (`JAX`, `JAX-MD`, and `FunSearch`-style evolutionary search).

Instead of locking into a single industry, **MatEvolve-JAX** evaluates candidate crystal compounds (starting from Google DeepMind's **GNoME** 380,000+ stable materials catalog, **MAX/MAB-phase atomic hybrids**, and **2D van der Waals monolayers**) across a **12-dimensional fundamental physics fingerprint** and automatically detects **"Freaks of Nature" (Physics Trade-Off Breakers)** at ultra-low ingredient cost ($\$/\text{kg}$), mapping each discovery to the industries it unlocks.

---

## The 6 Automated Physics Anomaly Detectors

1. **💎 Dielectric Heat Superhighways:** Electrical insulators ($\text{Band Gap} > 3.0\text{ eV}$) that conduct heat like metals ($> 80\text{ W/m}\cdot\text{K}$) via coherent phonon transport.
   - *Unlocked Industries:* 3D AI Microchip Cooling, High-Power Lasers, EV Power Inverters, Quantum Computing Substrates.
2. **❄️ Zero & Negative Thermal Expansion (NTE) Frameworks:** Rigid crystal lattices that exhibit near-zero or negative thermal expansion ($\text{CTE} \le 0.5\text{ ppm/K}$) across extreme temperature swings.
   - *Unlocked Industries:* Space Telescope Mirrors, Zero-Warp 3D Printing Additives, EUV Semiconductor Lithography, Precision Robotics.
3. **🛡️ Shatter-Proof Superhard Atomic Hybrids:** Materials that break the classical hardness–brittleness trade-off—combining high Vickers hardness ($H_v$) and low density with high Pugh ductility ratios ($B/G > 1.55$).
   - *Unlocked Industries:* Multi-Hit Ballistic Armor, Crack-Free Metal 3D Printing, Hypersonic Leading Edges, Eternal Cutting/Mining Tools.
4. **🔋 Solid-State Super-Ionic & Phonon-Glass Energy Harvesters:** Earth-abundant crystal channels with rapid room-temperature ion transport ($\text{mS/cm}$) or high Seebeck thermoelectric conversion with ultralow lattice thermal conductivity.
   - *Unlocked Industries:* Non-Flammable Solid-State Sodium/Magnesium Batteries, Waste-Heat-to-Electricity Generators, Grid Storage.
5. **🌀 Auxetic Shockwave Metamaterials:** Crystal cages with unusually low or negative Poisson's ratio ($\nu$) that resist indentation and disperse mechanical/acoustic shockwaves.
   - *Unlocked Industries:* Blast & Impact Shielding, Acoustic/Sonar Cloaking, Aerospace Structural Damping.
6. **💧 PFAS-Free Superlubricity & Non-Stick Surfaces:** Fluorine-free 2D and hybrid surfaces exhibiting extreme shear slip and hydrophobicity without toxic "forever chemicals."
   - *Unlocked Industries:* Teflon/PFAS Replacement Coatings, Zero-Wear Wind/Space Bearings, Anti-Icing Aircraft Skins.

---

## Quickstart

### 1. Run CLI Anomaly Screen
```bash
python3 engine.py
```

### 2. Launch Interactive Web Discovery Studio
```bash
python3 app.py
# Open http://localhost:8550 in your browser
```

## License

MIT License (Open Science release in accordance with Google TPU Research Cloud guidelines).
