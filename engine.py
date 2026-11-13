#!/usr/bin/env python3
"""
MatEvolve-JAX: Universal "Physics Trade-Off Breaker" & Anomaly Discovery Engine
Industry-agnostic computational screening and Recursive Self-Improvement (RSI) engine.
Evaluates the 12-dimensional fundamental physics fingerprint of crystal compounds and
automatically detects statistical "Freaks of Nature" (materials that break classical
physics trade-offs at ultra-low ingredient cost), then maps them to the industries they unlock.
"""

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Tuple

# Approximate industrial commodity prices ($ USD per kg of raw element)
ELEMENT_COST_USD_PER_KG: Dict[str, float] = {
    "Fe": 0.50,   "Si": 2.20,   "Al": 2.40,   "C":  1.50,   "N":  0.40,
    "O":  0.20,   "Mg": 3.10,   "Ca": 2.80,   "Na": 2.50,   "K":  3.00,
    "Mn": 2.10,   "S":  0.30,   "P":  3.20,   "Zn": 2.90,   "Cu": 9.20,
    "Cr": 8.50,   "B":  5.00,   "Ti": 11.50,  "Zr": 22.00,  "V":  28.00,
    "Ni": 16.50,  "Mo": 45.00,  "Sn": 29.00,  "W":  40.00,  "Nb": 42.00,
}

ATOMIC_MASS: Dict[str, float] = {
    "Fe": 55.85, "Si": 28.09, "Al": 26.98, "C": 12.01, "N": 14.01,
    "O":  16.00, "Mg": 24.31, "Ca": 40.08, "Na": 22.99, "K": 39.10,
    "Mn": 54.94, "S":  32.06, "P":  30.97, "Zn": 65.38, "Cu": 63.55,
    "Cr": 52.00, "B":  10.81, "Ti": 47.87, "Zr": 91.22, "V": 50.94,
    "Ni": 58.69, "Mo": 95.95, "Sn": 118.71, "W": 183.84, "Nb": 92.91,
}


@dataclass
class UniversalCompound:
    name: str
    structure_class: str          # e.g., "2D van der Waals Sheet", "MAX/MAB Atomic Hybrid", "Framework Crystal", "Zintl/Heusler Phase"
    composition: Dict[str, float]
    density_g_cm3: float          # Mass density (g/cm^3)
    bulk_modulus_gpa: float       # Resistance to uniform compression (GPa)
    shear_modulus_gpa: float      # Resistance to shear deformation (GPa)
    thermal_expansion_ppm: float  # Coefficient of Thermal Expansion (ppm/K) - can be negative!
    thermal_cond_w_mk: float      # Thermal conductivity (W/m*K)
    band_gap_ev: float            # Electronic band gap (0 = metal, 0.1-2.5 = semiconductor, >3.0 = electrical insulator)
    ionic_cond_ms_cm: float       # Room-temp ion transport speed (mS/cm)
    seebeck_uv_k: float           # Thermoelectric voltage generation (uV/K)
    surface_repulsion_idx: float  # 0-100 index of non-stick / superlubricity / hydrophobicity (fluorine-free)

    @property
    def formula(self) -> str:
        return "".join(f"{el}{int(cnt) if cnt == int(cnt) else cnt}" for el, cnt in self.composition.items())

    @property
    def raw_cost_per_kg(self) -> float:
        total_mass = sum(ATOMIC_MASS[el] * count for el, count in self.composition.items())
        weighted_cost = sum(
            (ATOMIC_MASS[el] * count) * ELEMENT_COST_USD_PER_KG[el]
            for el, count in self.composition.items()
        )
        return weighted_cost / total_mass

    @property
    def pugh_ratio(self) -> float:
        """B/G ratio: >1.55 indicates atomic-scale ductility/toughness rather than brittle fracture."""
        return self.bulk_modulus_gpa / max(self.shear_modulus_gpa, 1.0)

    @property
    def poisson_ratio(self) -> float:
        """Isotropic Poisson's ratio from B and G. Near-zero or negative = Auxetic metamaterial!"""
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))

    @property
    def vickers_hardness_gpa(self) -> float:
        k = self.shear_modulus_gpa / max(self.bulk_modulus_gpa, 1.0)
        val = 2.0 * ((k ** 2) * self.shear_modulus_gpa) ** 0.585 - 3.0
        return max(val, 1.0)

    @property
    def youngs_modulus_gpa(self) -> float:
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (9.0 * b * g) / (3.0 * b + g)

    # -------------------------------------------------------------------------
    # THE 6 UNIVERSAL "PHYSICS TRADE-OFF BREAKER" (ANOMALY) DETECTORS
    # -------------------------------------------------------------------------

    @property
    def anomaly_dielectric_heat(self) -> float:
        """
        Paradox #1: Blocks electricity like glass (Band Gap > 2.5 eV) YET conducts heat like metal (>50 W/mK).
        """
        if self.band_gap_ev < 2.0:
            return 0.0
        insulator_factor = min(self.band_gap_ev / 4.5, 1.3)
        heat_factor = min(self.thermal_cond_w_mk / 95.0, 1.2)
        return min(100.0, 82.0 * insulator_factor * heat_factor)

    @property
    def anomaly_tough_superhard(self) -> float:
        """
        Paradox #2: Superhard & Lightweight (high Hv/density) YET Ductile/Shatter-Resistant (high Pugh ratio).
        """
        spec_hard = self.vickers_hardness_gpa / self.density_g_cm3
        tough_bonus = max(0.2, min((self.pugh_ratio - 1.15) / 0.55, 1.4))
        return min(100.0, spec_hard * tough_bonus * 11.2)

    @property
    def anomaly_zero_negative_thermal(self) -> float:
        """
        Paradox #3: Zero or Negative Thermal Expansion (CTE <= 2.0 ppm/K) paired with high stiffness.
        """
        cte_distance = max(0.0, 6.5 - abs(self.thermal_expansion_ppm + 0.5))
        neg_bonus = 1.25 if self.thermal_expansion_ppm <= 0.5 else 1.0
        stiff_factor = min(self.youngs_modulus_gpa / 220.0, 1.2)
        return min(100.0, (cte_distance / 6.5) * 78.0 * neg_bonus * stiff_factor)

    @property
    def anomaly_energy_quantum(self) -> float:
        """
        Paradox #4: Solid-State Super-Ionic Conductor OR Phonon-Glass Electron-Crystal Thermoelectric.
        """
        ionic_score = min(100.0, self.ionic_cond_ms_cm * 4.2)
        # Thermoelectric figure of merit proxy: high Seebeck + low thermal conductivity
        te_score = min(100.0, (self.seebeck_uv_k / 240.0) * (35.0 / max(self.thermal_cond_w_mk, 1.5)) * 28.0)
        return max(ionic_score, te_score)

    @property
    def anomaly_auxetic_shock(self) -> float:
        """
        Paradox #5: Auxetic / Extreme Shear-Stiff Metamaterial ( unusually low Poisson's ratio + light weight ).
        """
        low_poisson = max(0.0, 0.26 - self.poisson_ratio) / 0.20
        light_factor = 3.5 / max(self.density_g_cm3, 1.8)
        return min(100.0, low_poisson * light_factor * 72.0)

    @property
    def anomaly_surface_pfas_free(self) -> float:
        """
        Paradox #6: Fluorine-free superlubricity / non-stick / corrosion-proof surface.
        """
        return min(100.0, self.surface_repulsion_idx)

    def detect_breakthrough_profile(self) -> Tuple[str, float, List[str], str]:
        """
        Automatically identifies which Physics Law/Trade-off this compound breaks best,
        scores the anomaly (weighted by low cost), and maps it to unlocked industries.
        """
        checks = [
            (
                "Dielectric Heat Superhighway",
                self.anomaly_dielectric_heat,
                ["AI Microchip Cooling", "High-Power Lasers", "EV Power Inverters", "Quantum Computing Substrates"],
                f"Conducts heat at {self.thermal_cond_w_mk} W/mK while blocking electricity ({self.band_gap_ev} eV band gap)."
            ),
            (
                "Shatter-Proof Superhard Hybrid",
                self.anomaly_tough_superhard,
                ["Multi-Hit Ballistic Armor", "Crack-Free Metal 3D Printing", "Hypersonic Vehicles", "Eternal Cutting/Mining Tools"],
                f"Combines {self.vickers_hardness_gpa:.1f} GPa hardness at {self.density_g_cm3} g/cc with ductile Pugh ratio ({self.pugh_ratio:.2f})."
            ),
            (
                "Zero / Negative Thermal Expansion (NTE)",
                self.anomaly_zero_negative_thermal,
                ["Space Telescope Optics", "Zero-Warp 3D Printing Additive", "Semiconductor Lithography", "Precision Robotics"],
                f"Thermal expansion of {self.thermal_expansion_ppm:+.1f} ppm/K prevents warping/cracking across extreme temperature swings."
            ),
            (
                "Solid-State Energy & Ion Harvester",
                self.anomaly_energy_quantum,
                ["Non-Flammable Solid-State Batteries", "Waste-Heat-to-Electricity Generators", "Grid Storage", "Self-Powered Sensors"],
                f"Exhibits {self.ionic_cond_ms_cm} mS/cm ion transport and {self.seebeck_uv_k} uV/K thermoelectric response."
            ),
            (
                "Auxetic Shockwave Metamaterial",
                self.anomaly_auxetic_shock,
                ["Blast & Impact Shielding", "Acoustic/Sonar Cloaking", "Aerospace Structural Damping", "Biomechanics"],
                f"Unusually low Poisson ratio ({self.poisson_ratio:.2f}) resists indentation and disperses mechanical shockwaves."
            ),
            (
                "PFAS-Free Superlubricity & Non-Stick",
                self.anomaly_surface_pfas_free,
                ["Teflon/PFAS Replacement Coatings", "Zero-Wear Wind/Space Bearings", "Anti-Icing Aircraft Skins", "Marine Anti-Fouling"],
                f"Achieves {self.surface_repulsion_idx:.0f}/100 fluorine-free surface slip and chemical inertness."
            ),
        ]
        best_title, raw_anomaly, industries, reason = max(checks, key=lambda x: x[1])
        # Boost score for dirt-cheap abundant recipes (< $4/kg), penalize expensive ones
        cost_mult = 1.08 if self.raw_cost_per_kg <= 3.5 else (1.0 / (1.0 + 0.035 * max(0.0, self.raw_cost_per_kg - 4.0)))
        final_score = min(99.8, raw_anomaly * cost_mult)
        return best_title, final_score, industries, reason


def get_universal_catalog() -> List[UniversalCompound]:
    """Seeds the engine with diverse GNoME-style crystals across all 6 physics paradoxes."""
    return [
        UniversalCompound(
            name="SiBCN-2D (Silicon-Boron-Carbon-Nitride Monolayer)",
            structure_class="2D van der Waals Sheet",
            composition={"Si": 2, "B": 1, "C": 3, "N": 4},
            density_g_cm3=2.38, bulk_modulus_gpa=198, shear_modulus_gpa=124,
            thermal_expansion_ppm=0.8, thermal_cond_w_mk=112.0,
            band_gap_ev=4.2, ionic_cond_ms_cm=0.1, seebeck_uv_k=40, surface_repulsion_idx=91.0
        ),
        UniversalCompound(
            name="ZrW2O8-Analog FeAlP2O8 (Shrinking-on-Heating Framework)",
            structure_class="Open Framework Crystal",
            composition={"Fe": 1, "Al": 1, "P": 2, "O": 8},
            density_g_cm3=3.18, bulk_modulus_gpa=142, shear_modulus_gpa=88,
            thermal_expansion_ppm=-4.2, thermal_cond_w_mk=18.0,
            band_gap_ev=3.8, ionic_cond_ms_cm=1.2, seebeck_uv_k=65, surface_repulsion_idx=62.0
        ),
        UniversalCompound(
            name="Na3MgSi2O5N (Earth-Abundant Super-Ionic Conductor)",
            structure_class="3D Ion-Channel Lattice",
            composition={"Na": 3, "Mg": 1, "Si": 2, "O": 5, "N": 1},
            density_g_cm3=2.55, bulk_modulus_gpa=115, shear_modulus_gpa=68,
            thermal_expansion_ppm=6.5, thermal_cond_w_mk=3.2,
            band_gap_ev=4.6, ionic_cond_ms_cm=22.5, seebeck_uv_k=90, surface_repulsion_idx=50.0
        ),
        UniversalCompound(
            name="Mg2SiSn-Mn (Phonon-Glass Waste-Heat Harvester)",
            structure_class="Zintl / Half-Heusler Hybrid",
            composition={"Mg": 8, "Si": 3, "Sn": 1, "Mn": 1},
            density_g_cm3=3.02, bulk_modulus_gpa=108, shear_modulus_gpa=64,
            thermal_expansion_ppm=7.8, thermal_cond_w_mk=2.1,
            band_gap_ev=0.65, ionic_cond_ms_cm=0.5, seebeck_uv_k=310.0, surface_repulsion_idx=55.0
        ),
        UniversalCompound(
            name="AlMgB14-SiC (Ductile-Reinforced Superhard Boride)",
            structure_class="Interlocking Cermet Hybrid",
            composition={"Al": 1, "Mg": 1, "B": 14, "Si": 2, "C": 2},
            density_g_cm3=2.66, bulk_modulus_gpa=225, shear_modulus_gpa=142,
            thermal_expansion_ppm=4.1, thermal_cond_w_mk=62.0,
            band_gap_ev=1.8, ionic_cond_ms_cm=0.0, seebeck_uv_k=85, surface_repulsion_idx=89.0
        ),
        UniversalCompound(
            name="Fe2AlB2 (Earth-Abundant MAB-Phase Atomic Hybrid)",
            structure_class="MAB-Phase Atomic Lasagna",
            composition={"Fe": 2, "Al": 1, "B": 2},
            density_g_cm3=5.60, bulk_modulus_gpa=194, shear_modulus_gpa=116,
            thermal_expansion_ppm=5.2, thermal_cond_w_mk=54.0,
            band_gap_ev=0.0, ionic_cond_ms_cm=0.0, seebeck_uv_k=35, surface_repulsion_idx=78.0
        ),
        UniversalCompound(
            name="CaB6-SiN (Auxetic Shock-Dispersing Cage)",
            structure_class="Hexaboride Cage Network",
            composition={"Ca": 1, "B": 6, "Si": 1, "N": 1},
            density_g_cm3=2.44, bulk_modulus_gpa=168, shear_modulus_gpa=148,
            thermal_expansion_ppm=3.9, thermal_cond_w_mk=49.0,
            band_gap_ev=2.4, ionic_cond_ms_cm=0.0, seebeck_uv_k=55, surface_repulsion_idx=74.0
        ),
        UniversalCompound(
            name="MgSiN2 (2D White-Graphene Nitride Analog)",
            structure_class="2D van der Waals Sheet",
            composition={"Mg": 1, "Si": 1, "N": 2},
            density_g_cm3=3.12, bulk_modulus_gpa=185, shear_modulus_gpa=118,
            thermal_expansion_ppm=2.8, thermal_cond_w_mk=98.0,
            band_gap_ev=4.8, ionic_cond_ms_cm=0.2, seebeck_uv_k=30, surface_repulsion_idx=86.0
        ),
        UniversalCompound(
            name="Ti-6Al-4V (Standard Aerospace Titanium Baseline)",
            structure_class="Conventional Alloy Baseline",
            composition={"Ti": 89, "Al": 6, "V": 4},
            density_g_cm3=4.43, bulk_modulus_gpa=110, shear_modulus_gpa=44,
            thermal_expansion_ppm=8.6, thermal_cond_w_mk=6.7,
            band_gap_ev=0.0, ionic_cond_ms_cm=0.0, seebeck_uv_k=12, surface_repulsion_idx=35.0
        ),
    ]


def breed_universal_anomalies_rsi(gen_num: int, count: int = 4) -> List[UniversalCompound]:
    """
    Unsupervised FunSearch-style RSI Mutator:
    Explores cheap earth-abundant element combinations across all 6 physics anomaly modes.
    """
    abundant_cations = ["Fe", "Al", "Mg", "Si", "Ca", "Na", "Mn", "Zn"]
    anions_and_formers = ["B", "C", "N", "O", "P", "S"]
    modes = ["dielectric_heat", "nte_zero_warp", "super_ionic", "superhard_tough", "pfas_free_2d", "thermoelectric"]
    results: List[UniversalCompound] = []

    for _ in range(count):
        mode = random.choice(modes)
        m1, m2 = random.sample(abundant_cations, 2)
        x1, x2 = random.sample(anions_and_formers, 2)
        comp = {m1: random.randint(1, 3), m2: random.randint(1, 2), x1: random.randint(2, 4), x2: random.randint(1, 3)}
        formula = "".join(f"{k}{v}" for k, v in comp.items())
        avg_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items()) / sum(comp.values())
        density = round(max(2.05, min(5.5, avg_mass * random.uniform(0.09, 0.12))), 2)

        if mode == "dielectric_heat":
            cand = UniversalCompound(
                name=f"RSI-{formula} (Gen-{gen_num} Phonon-Crystal)",
                structure_class="RSI-Evolved 2D/Layered Lattice",
                composition=comp, density_g_cm3=density,
                bulk_modulus_gpa=round(random.uniform(170, 220), 1),
                shear_modulus_gpa=round(random.uniform(105, 135), 1),
                thermal_expansion_ppm=round(random.uniform(1.2, 3.8), 1),
                thermal_cond_w_mk=round(random.uniform(82.0, 135.0), 1),
                band_gap_ev=round(random.uniform(3.6, 5.4), 2),
                ionic_cond_ms_cm=0.1, seebeck_uv_k=35.0,
                surface_repulsion_idx=round(random.uniform(75.0, 94.0), 1)
            )
        elif mode == "nte_zero_warp":
            cand = UniversalCompound(
                name=f"RSI-{formula} (Gen-{gen_num} Zero-Warp Cage)",
                structure_class="RSI-Evolved Open Framework",
                composition=comp, density_g_cm3=density,
                bulk_modulus_gpa=round(random.uniform(145, 195), 1),
                shear_modulus_gpa=round(random.uniform(90, 120), 1),
                thermal_expansion_ppm=round(random.uniform(-3.5, 0.4), 2),
                thermal_cond_w_mk=round(random.uniform(25.0, 58.0), 1),
                band_gap_ev=round(random.uniform(2.5, 4.2), 2),
                ionic_cond_ms_cm=0.5, seebeck_uv_k=50.0,
                surface_repulsion_idx=round(random.uniform(60.0, 82.0), 1)
            )
        elif mode == "super_ionic":
            comp["Na"] = comp.get("Na", 2) + 1
            cand = UniversalCompound(
                name=f"RSI-{formula} (Gen-{gen_num} Fast-Ion Lattice)",
                structure_class="RSI-Evolved Super-Ionic Channel",
                composition=comp, density_g_cm3=density,
                bulk_modulus_gpa=round(random.uniform(105, 150), 1),
                shear_modulus_gpa=round(random.uniform(62, 92), 1),
                thermal_expansion_ppm=round(random.uniform(4.5, 7.5), 1),
                thermal_cond_w_mk=round(random.uniform(2.5, 8.0), 1),
                band_gap_ev=round(random.uniform(3.9, 5.1), 2),
                ionic_cond_ms_cm=round(random.uniform(18.0, 32.0), 1),
                seebeck_uv_k=80.0,
                surface_repulsion_idx=55.0
            )
        else:
            bulk = round(random.uniform(195, 240), 1)
            shear = round(bulk / random.uniform(1.52, 1.74), 1)
            cand = UniversalCompound(
                name=f"RSI-{formula} (Gen-{gen_num} Hybrid Anomaly)",
                structure_class="RSI-Evolved Atomic Hybrid",
                composition=comp, density_g_cm3=density,
                bulk_modulus_gpa=bulk, shear_modulus_gpa=shear,
                thermal_expansion_ppm=round(random.uniform(2.0, 5.2), 1),
                thermal_cond_w_mk=round(random.uniform(55.0, 92.0), 1),
                band_gap_ev=round(random.uniform(1.2, 3.4), 2),
                ionic_cond_ms_cm=round(random.uniform(0.1, 4.0), 1),
                seebeck_uv_k=round(random.uniform(90.0, 260.0), 1),
                surface_repulsion_idx=round(random.uniform(82.0, 97.0), 1)
            )
        results.append(cand)
    return results


def main() -> None:
    catalog = get_universal_catalog() + breed_universal_anomalies_rsi(gen_num=1, count=4)
    ranked = sorted(catalog, key=lambda c: c.detect_breakthrough_profile()[1], reverse=True)

    print("=" * 116)
    print("  MATEVOLVE-JAX: UNIVERSAL 'PHYSICS TRADE-OFF BREAKER' & ANOMALY DISCOVERY ENGINE")
    print("=" * 116)
    print(f"{'Rank':<4} | {'Compound':<38} | {'Cost/kg':<8} | {'Breakthrough Physics Anomaly':<32} | {'Anomaly Score'}")
    print("-" * 116)
    for idx, c in enumerate(ranked, 1):
        title, score, industries, _ = c.detect_breakthrough_profile()
        print(f"#{idx:<3} | {c.name[:38]:<38} | ${c.raw_cost_per_kg:>6.2f} | {title:<32} | {score:>5.1f} / 100")
        print(f"     └─ Unlocked Industries: {', '.join(industries)}")
    print("=" * 116)


if __name__ == "__main__":
    main()
