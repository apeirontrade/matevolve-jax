#!/usr/bin/env python3
"""
MatEvolve-JAX (Prototype): Literature Catalog & Heuristic Scoring Demo
Evaluates derived elastic properties, elemental weight-fraction cost estimates,
and thresholded property heuristics over a hand-entered catalog of literature-sourced
inorganic materials.
"""

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Tuple

# Approximate industrial / supplier elemental prices ($ USD per kg of raw element).
# Note: This coarse model multiplies atomic weight fractions by elemental prices and
# ignores chemical precursor form, purity grade, and synthesis/processing costs.
# Boron price ($300/kg) reflects elemental amorphous boron powder supplier listings (~$280-350/kg):
#   - https://www.chemicalbook.com/price/Boron-1.htm
#   - https://sdhshchem.en.made-in-china.com/product/vZFTQAojfIaB/China-Elemental-Amorphous-Boron-Powder-Price-CAS-7440-42-8.html
ELEMENT_COST_USD_PER_KG: Dict[str, float] = {
    "Fe": 0.50,   "Si": 2.20,   "Al": 2.40,   "C":  1.50,   "N":  0.40,
    "O":  0.20,   "Mg": 3.10,   "Ca": 2.80,   "Na": 2.50,   "K":  3.00,
    "Mn": 2.10,   "S":  0.30,   "P":  3.20,   "Zn": 2.90,   "Cu": 9.20,
    "Cr": 8.50,   "B":  300.00, "Ti": 11.50,  "Zr": 22.00,  "V":  28.00,
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
    structure_class: str
    composition: Dict[str, float]
    density_g_cm3: float
    bulk_modulus_gpa: float
    shear_modulus_gpa: float
    thermal_expansion_ppm: float
    thermal_cond_w_mk: float
    band_gap_ev: float
    source: str = ""
    property_sources: Dict[str, str] = field(default_factory=dict)
    placeholder: bool = False

    @property
    def formula(self) -> str:
        return "".join(
            f"{el}{int(cnt) if cnt == int(cnt) else cnt}"
            for el, cnt in self.composition.items()
        )

    def weight_fractions(self) -> Dict[str, float]:
        """Returns mass fraction (0.0 to 1.0) of each element from atomic proportions."""
        total_mass = sum(ATOMIC_MASS[el] * count for el, count in self.composition.items())
        return {
            el: (ATOMIC_MASS[el] * count) / total_mass
            for el, count in self.composition.items()
        }

    @property
    def raw_cost_per_kg(self) -> float:
        wf = self.weight_fractions()
        return sum(frac * ELEMENT_COST_USD_PER_KG[el] for el, frac in wf.items())

    @property
    def pugh_ratio(self) -> float:
        """B/G ratio (Pugh, Phil. Mag. 45, 823, 1954)."""
        return self.bulk_modulus_gpa / max(self.shear_modulus_gpa, 1.0)

    @property
    def poisson_ratio(self) -> float:
        """Isotropic Poisson's ratio from B and G: (3B - 2G) / (2(3B + G))."""
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))

    @property
    def vickers_hardness_gpa(self) -> float:
        """Chen's empirical Vickers hardness correlation: Hv = 2*(k^2 * G)^0.585 - 3, k = G/B."""
        k = self.shear_modulus_gpa / max(self.bulk_modulus_gpa, 1.0)
        val = 2.0 * ((k ** 2) * self.shear_modulus_gpa) ** 0.585 - 3.0
        return max(val, 1.0)

    @property
    def youngs_modulus_gpa(self) -> float:
        """Isotropic Young's modulus from B and G: 9BG / (3B + G)."""
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (9.0 * b * g) / (3.0 * b + g)

    @property
    def anomaly_dielectric_heat(self) -> float:
        if self.band_gap_ev < 2.0:
            return 0.0
        insulator_factor = min(self.band_gap_ev / 4.5, 1.3)
        heat_factor = min(self.thermal_cond_w_mk / 95.0, 1.2)
        return min(100.0, 82.0 * insulator_factor * heat_factor)

    @property
    def anomaly_tough_superhard(self) -> float:
        spec_hard = self.vickers_hardness_gpa / self.density_g_cm3
        tough_bonus = max(0.2, min((self.pugh_ratio - 1.15) / 0.55, 1.4))
        return min(100.0, spec_hard * tough_bonus * 11.2)

    @property
    def anomaly_zero_negative_thermal(self) -> float:
        cte_distance = max(0.0, 6.5 - abs(self.thermal_expansion_ppm + 0.5))
        neg_bonus = 1.25 if self.thermal_expansion_ppm <= 0.5 else 1.0
        stiff_factor = min(self.youngs_modulus_gpa / 220.0, 1.2)
        return min(100.0, (cte_distance / 6.5) * 78.0 * neg_bonus * stiff_factor)

    @property
    def anomaly_auxetic_shock(self) -> float:
        low_poisson = max(0.0, 0.26 - self.poisson_ratio) / 0.20
        light_factor = 3.5 / max(self.density_g_cm3, 1.8)
        return min(100.0, low_poisson * light_factor * 72.0)

    def detect_breakthrough_profile(self) -> Tuple[str, float, List[str], str]:
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
                "Auxetic Shockwave Metamaterial",
                self.anomaly_auxetic_shock,
                ["Blast & Impact Shielding", "Acoustic/Sonar Cloaking", "Aerospace Structural Damping", "Biomechanics"],
                f"Unusually low Poisson ratio ({self.poisson_ratio:.2f}) resists indentation and disperses mechanical shockwaves."
            ),
        ]
        best_title, raw_anomaly, industries, reason = max(checks, key=lambda x: x[1])
        cost_mult = 1.08 if self.raw_cost_per_kg <= 3.5 else (1.0 / (1.0 + 0.035 * max(0.0, self.raw_cost_per_kg - 4.0)))
        final_score = min(99.8, raw_anomaly * cost_mult)
        return best_title, final_score, industries, reason


def get_universal_catalog() -> List[UniversalCompound]:
    """
    Returns literature-sourced catalog entries where every numeric property is tied
    to an explicit citation in `property_sources`.
    """
    return [
        UniversalCompound(
            name="Ti-6Al-4V (Grade 5 Titanium Alloy Baseline)",
            structure_class="Alpha-Beta Titanium Alloy (Baseline)",
            # Atomic percentages reproducing 90.0 wt% Ti, 6.0 wt% Al, 4.0 wt% V
            composition={"Ti": 86.2, "Al": 10.2, "V": 3.6},
            density_g_cm3=4.43,
            bulk_modulus_gpa=110.0,
            shear_modulus_gpa=44.0,
            thermal_expansion_ppm=8.6,
            thermal_cond_w_mk=6.7,
            band_gap_ev=0.0,
            source="ASM Aerospace Specification Metals (MTP641); Boyer et al., Materials Properties Handbook: Titanium Alloys (ASM International, 1994)",
            property_sources={
                "composition": "ASTM B265 Grade 5 (90 wt% Ti, 6 wt% Al, 4 wt% V -> 86.2 at% Ti, 10.2 at% Al, 3.6 at% V)",
                "density_g_cm3": "ASM MatWeb MTP641 (4.43 g/cm^3)",
                "bulk_modulus_gpa": "Boyer et al., ASM Titanium Alloys Handbook, 1994 (~110 GPa)",
                "shear_modulus_gpa": "ASM MatWeb MTP641 (44.0 GPa)",
                "thermal_expansion_ppm": "ASM MatWeb MTP641 (8.6e-6 /K at 20-100 C)",
                "thermal_cond_w_mk": "ASM MatWeb MTP641 (6.7 W/m-K)",
                "band_gap_ev": "Metallic alloy conductor (0.0 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="B4C (Boron Carbide Ceramic Baseline)",
            structure_class="Rhombohedral Covalent Ceramic (Baseline)",
            composition={"B": 4, "C": 1},
            density_g_cm3=2.52,
            bulk_modulus_gpa=247.0,
            shear_modulus_gpa=197.0,
            thermal_expansion_ppm=5.6,
            thermal_cond_w_mk=30.0,
            band_gap_ev=2.09,
            source="Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), DOI:10.1111/j.1551-2916.2011.04865.x; Werheit, J. Phys.: Condens. Matter 18, S1065 (2006), DOI:10.1088/0953-8984/18/22/S01",
            property_sources={
                "density_g_cm3": "Domnich et al. 2011, DOI:10.1111/j.1551-2916.2011.04865.x (2.52 g/cm^3)",
                "bulk_modulus_gpa": "McClellan et al., J. Am. Ceram. Soc. 84, 1049 (2001), DOI:10.1111/j.1151-2916.2001.tb00789.x (247 GPa)",
                "shear_modulus_gpa": "McClellan et al. 2001, DOI:10.1111/j.1151-2916.2001.tb00789.x (197 GPa)",
                "thermal_expansion_ppm": "Munro, J. Res. NIST 105(5), 709-720 (2000) / Domnich et al. 2011 (5.6e-6 /K)",
                "thermal_cond_w_mk": "Domnich et al. 2011, DOI:10.1111/j.1551-2916.2011.04865.x (~30 W/m-K at 300 K)",
                "band_gap_ev": "Werheit 2006, DOI:10.1088/0953-8984/18/22/S01 (2.09 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="AlN (Wurtzite Aluminum Nitride)",
            structure_class="Wurtzite Wide-Bandgap Ceramic",
            composition={"Al": 1, "N": 1},
            density_g_cm3=3.26,
            bulk_modulus_gpa=202.0,
            shear_modulus_gpa=131.0,
            thermal_expansion_ppm=4.5,
            thermal_cond_w_mk=285.0,
            band_gap_ev=6.20,
            source="Slack et al., J. Phys. Chem. Solids 48(7), 641-647 (1987), DOI:10.1016/0022-3697(87)90153-3; McNeil et al., J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x",
            property_sources={
                "density_g_cm3": "Slack et al. 1987, DOI:10.1016/0022-3697(87)90153-3 (3.26 g/cm^3)",
                "bulk_modulus_gpa": "McNeil et al. 1993, DOI:10.1111/j.1151-2916.1993.tb03730.x (202 GPa)",
                "shear_modulus_gpa": "McNeil et al. 1993, DOI:10.1111/j.1151-2916.1993.tb03730.x (131 GPa)",
                "thermal_expansion_ppm": "Slack & Bartram, J. Appl. Phys. 46, 89 (1975), DOI:10.1063/1.321373 (4.5e-6 /K at 300 K)",
                "thermal_cond_w_mk": "Slack et al. 1987, DOI:10.1016/0022-3697(87)90153-3 (285 W/m-K single crystal at 300 K)",
                "band_gap_ev": "Yim et al., J. Appl. Phys. 44, 292 (1973), DOI:10.1063/1.1661876 (6.2 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="Ti3SiC2 (Titanium Silicon Carbide MAX Phase)",
            structure_class="MAX-Phase Nanolaminate",
            composition={"Ti": 3, "Si": 1, "C": 2},
            density_g_cm3=4.52,
            bulk_modulus_gpa=187.0,
            shear_modulus_gpa=138.0,
            thermal_expansion_ppm=9.1,
            thermal_cond_w_mk=37.0,
            band_gap_ev=0.0,
            source="Barsoum, Prog. Solid State Chem. 28, 201-281 (2000), DOI:10.1016/S0079-6786(00)00006-6; Finkel et al., J. Appl. Phys. 87, 1701 (2000), DOI:10.1063/1.372080",
            property_sources={
                "density_g_cm3": "Barsoum 2000, DOI:10.1016/S0079-6786(00)00006-6 (4.52 g/cm^3)",
                "bulk_modulus_gpa": "Finkel et al. 2000, DOI:10.1063/1.372080 (187 GPa)",
                "shear_modulus_gpa": "Finkel et al. 2000, DOI:10.1063/1.372080 (138 GPa)",
                "thermal_expansion_ppm": "Barsoum et al., J. Am. Ceram. Soc. 82(9), 2545 (1999), DOI:10.1111/j.1151-2916.1999.tb02117.x (9.1e-6 /K)",
                "thermal_cond_w_mk": "Barsoum 2000, DOI:10.1016/S0079-6786(00)00006-6 (37 W/m-K at 300 K)",
                "band_gap_ev": "Metallic nanolaminate conductor, Barsoum 2000 (0.0 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="Cr2AlC (Chromium Aluminum Carbide MAX Phase)",
            structure_class="MAX-Phase Nanolaminate",
            composition={"Cr": 2, "Al": 1, "C": 1},
            density_g_cm3=5.24,
            bulk_modulus_gpa=166.0,
            shear_modulus_gpa=105.0,
            thermal_expansion_ppm=12.0,
            thermal_cond_w_mk=22.0,
            band_gap_ev=0.0,
            source="Tian et al., Acta Materialia 54(16), 4273-4283 (2006), DOI:10.1016/j.actamat.2006.05.020; Sun et al., Solid State Commun. 129(9), 589-592 (2004), DOI:10.1016/j.ssc.2003.12.008",
            property_sources={
                "density_g_cm3": "Tian et al. 2006, DOI:10.1016/j.actamat.2006.05.020 (5.24 g/cm^3)",
                "bulk_modulus_gpa": "Sun et al. 2004, DOI:10.1016/j.ssc.2003.12.008 (166 GPa)",
                "shear_modulus_gpa": "Sun et al. 2004 / Tian et al. 2006, DOI:10.1016/j.actamat.2006.05.020 (105 GPa)",
                "thermal_expansion_ppm": "Tian et al. 2006, DOI:10.1016/j.actamat.2006.05.020 (12.0e-6 /K)",
                "thermal_cond_w_mk": "Tian et al. 2006, DOI:10.1016/j.actamat.2006.05.020 (22 W/m-K at 300 K)",
                "band_gap_ev": "Metallic nanolaminate conductor, Tian et al. 2006 (0.0 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="Fe2AlB2 (Iron Aluminum Boride MAB Phase)",
            structure_class="MAB-Phase Nanolaminate",
            composition={"Fe": 2, "Al": 1, "B": 2},
            density_g_cm3=5.77,
            bulk_modulus_gpa=174.0,
            shear_modulus_gpa=114.0,
            thermal_expansion_ppm=10.6,
            thermal_cond_w_mk=7.5,
            band_gap_ev=0.0,
            source="Ade & Hillebrecht, Angew. Chem. Int. Ed. 54, 11568 (2015), DOI:10.1002/anie.201505525; Li et al., J. Am. Ceram. Soc. 100(9), 4180 (2017), DOI:10.1111/jace.14945; Verger et al., J. Appl. Phys. 124, 215108 (2018), DOI:10.1063/1.5055964",
            property_sources={
                "density_g_cm3": "Ade & Hillebrecht 2015, DOI:10.1002/anie.201505525 (5.77 g/cm^3)",
                "bulk_modulus_gpa": "Li et al. 2017, DOI:10.1111/jace.14945 / Materials Project mp-20780 (174 GPa)",
                "shear_modulus_gpa": "Li et al. 2017, DOI:10.1111/jace.14945 / Materials Project mp-20780 (114 GPa)",
                "thermal_expansion_ppm": "Verger et al. 2018, DOI:10.1063/1.5055964 (10.6e-6 /K)",
                "thermal_cond_w_mk": "Kota et al., Int. Mater. Rev. 65(4), 226-255 (2020), DOI:10.1080/09506608.2019.1637090 (~7.5 W/m-K)",
                "band_gap_ev": "Metallic conductor, Ade & Hillebrecht 2015 (0.0 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="MgSiN2 (Magnesium Silicon Nitride)",
            structure_class="Wurtzite-Derived Ternary Nitride",
            composition={"Mg": 1, "Si": 1, "N": 2},
            density_g_cm3=3.13,
            bulk_modulus_gpa=177.0,
            shear_modulus_gpa=112.0,
            thermal_expansion_ppm=4.5,
            thermal_cond_w_mk=23.0,
            band_gap_ev=4.80,
            source="Bruls et al., J. Phys. Chem. Solids 61, 1285 (2000), DOI:10.1016/S0022-3697(99)00416-3; Bruls et al., J. Eur. Ceram. Soc. 21, 263 (2001), DOI:10.1016/S0955-2219(00)00210-7; de Boer et al., Phys. Rev. B 93, 155205 (2016), DOI:10.1103/PhysRevB.93.155205",
            property_sources={
                "density_g_cm3": "Bruls et al. 2000, DOI:10.1016/S0022-3697(99)00416-3 (3.128 g/cm^3)",
                "bulk_modulus_gpa": "Bruls et al. 2001, DOI:10.1016/S0955-2219(00)00210-7 / Materials Project mp-3677 (177 GPa)",
                "shear_modulus_gpa": "Bruls et al. 2001, DOI:10.1016/S0955-2219(00)00210-7 / Materials Project mp-3677 (112 GPa)",
                "thermal_expansion_ppm": "Bruls et al. 2000, DOI:10.1016/S0022-3697(99)00416-3 (4.5e-6 /K at 300 K)",
                "thermal_cond_w_mk": "Bruls, Ph.D. Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906 (23 W/m-K experimental sintered max)",
                "band_gap_ev": "de Boer et al. 2016, DOI:10.1103/PhysRevB.93.155205 (4.8 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="ZrW2O8 (Alpha-Zirconium Tungstate)",
            structure_class="Open Framework Oxide",
            composition={"Zr": 1, "W": 2, "O": 8},
            density_g_cm3=5.07,
            bulk_modulus_gpa=74.5,
            shear_modulus_gpa=33.8,
            thermal_expansion_ppm=-8.7,
            thermal_cond_w_mk=0.51,
            band_gap_ev=2.84,
            source="Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; Kennedy & White, Solid State Commun. 134(4), 271 (2005), DOI:10.1016/j.ssc.2005.01.031",
            property_sources={
                "density_g_cm3": "Mary et al. 1996, DOI:10.1126/science.272.5258.90 (5.07 g/cm^3)",
                "bulk_modulus_gpa": "Drymiotis et al. 2004, DOI:10.1103/PhysRevLett.93.025502 (74.5 GPa at 300 K)",
                "shear_modulus_gpa": "Drymiotis et al. 2004, DOI:10.1103/PhysRevLett.93.025502 (33.8 GPa at 300 K)",
                "thermal_expansion_ppm": "Mary et al. 1996, DOI:10.1126/science.272.5258.90 (-8.7e-6 /K isotropic from 0.3 to 1050 K)",
                "thermal_cond_w_mk": "Kennedy & White 2005, DOI:10.1016/j.ssc.2005.01.031 (0.51 W/m-K at 300 K)",
                "band_gap_ev": "Materials Project mp-18781, DOI:10.17188/1193603 (2.84 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="SiO2 (Alpha-Cristobalite)",
            structure_class="Tetrahedral Silica Polymorph",
            composition={"Si": 1, "O": 2},
            density_g_cm3=2.33,
            bulk_modulus_gpa=16.4,
            shear_modulus_gpa=39.1,
            thermal_expansion_ppm=10.3,
            thermal_cond_w_mk=1.40,
            band_gap_ev=8.90,
            source="Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650; Swainson & Dove, Phys. Chem. Miner. 22, 61 (1995), DOI:10.1007/BF00202681",
            property_sources={
                "density_g_cm3": "Yeganeh-Haeri et al. 1992, DOI:10.1126/science.257.5070.650 (2.33 g/cm^3)",
                "bulk_modulus_gpa": "Yeganeh-Haeri et al. 1992, DOI:10.1126/science.257.5070.650 (16.4 GPa VRH aggregate at 300 K)",
                "shear_modulus_gpa": "Yeganeh-Haeri et al. 1992, DOI:10.1126/science.257.5070.650 (39.1 GPa VRH aggregate at 300 K -> Poisson ratio -0.164)",
                "thermal_expansion_ppm": "Swainson & Dove 1995, DOI:10.1007/BF00202681 (~10.3e-6 /K at 300 K)",
                "thermal_cond_w_mk": "CRC Handbook of Chemistry and Physics, 104th Ed. (~1.4 W/m-K for crystalline/vitreous silica at 300 K)",
                "band_gap_ev": "Laughlin, Phys. Rev. B 22, 3021 (1980), DOI:10.1103/PhysRevB.22.3021 (8.9 eV)",
            },
            placeholder=False,
        ),
    ]


def breed_universal_anomalies_rsi(gen_num: int, count: int = 4) -> List[UniversalCompound]:
    """Temporary wrapper replaced in Step 4."""
    return []


def main() -> None:
    catalog = get_universal_catalog()
    for c in catalog:
        print(f"{c.name}: cost=${c.raw_cost_per_kg:.2f}/kg, source={c.source}")


if __name__ == "__main__":
    main()
