#!/usr/bin/env python3
"""
MatEvolve-JAX (Prototype): Literature Catalog & Heuristic Scoring Demo
Evaluates derived elastic properties, elemental weight-fraction cost estimates,
and thresholded property heuristics over a hand-entered catalog of literature-sourced
inorganic materials.
"""

import argparse
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Tuple

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

# Minimum heuristic score (0-100 scale) required to flag a property anomaly.
HEURISTIC_THRESHOLD: float = 50.0


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
        """
        Chen's empirical Vickers hardness correlation: Hv = 2*(k^2 * G)^0.585 - 3, k = G/B
        (Chen et al., Intermetallics 19, 1275, 2011, DOI:10.1016/j.intermet.2011.03.026).
        Note: Fit on brittle covalent/ionic crystals; overestimates indentation hardness
        in layered MAX/MAB phases where basal dislocation slip occurs.
        """
        k = self.shear_modulus_gpa / max(self.bulk_modulus_gpa, 1.0)
        val = 2.0 * ((k ** 2) * self.shear_modulus_gpa) ** 0.585 - 3.0
        return max(val, 1.0)

    @property
    def youngs_modulus_gpa(self) -> float:
        """Isotropic Young's modulus from B and G: 9BG / (3B + G)."""
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (9.0 * b * g) / (3.0 * b + g)

    # -------------------------------------------------------------------------
    # THRESHOLDED PROPERTY HEURISTICS (0 - 100 SCALE)
    # -------------------------------------------------------------------------

    @property
    def heuristic_dielectric_heat(self) -> float:
        """
        Heuristic 1 (Dielectric thermal conductor):
        Threshold: requires electrical insulator (band_gap_ev >= 3.0 eV) AND
        high thermal conductivity (thermal_cond_w_mk >= 50.0 W/(m*K)).
        Returns 0.0 if either threshold is not met.
        """
        if self.band_gap_ev < 3.0 or self.thermal_cond_w_mk < 50.0:
            return 0.0
        bg_term = min((self.band_gap_ev - 3.0) / 3.5, 1.0)
        tc_term = min((self.thermal_cond_w_mk - 50.0) / 235.0, 1.0)
        return round(50.0 + 25.0 * bg_term + 25.0 * tc_term, 1)

    @property
    def heuristic_damage_tolerant_hard(self) -> float:
        """
        Heuristic 2 (High-modulus moderate-Pugh phase):
        Threshold: requires shear_modulus_gpa >= 110.0 GPa, Pugh ratio B/G >= 1.35
        (above brittle ceramics like B4C at 1.25), and density_g_cm3 <= 5.0 g/cm^3.
        Does not use Chen's hardness estimate, which overpredicts indentation hardness
        for machinable MAX phases such as Ti3SiC2.
        Returns 0.0 if any threshold is not met.
        """
        if (
            self.shear_modulus_gpa < 110.0
            or self.pugh_ratio < 1.35
            or self.density_g_cm3 > 5.0
        ):
            return 0.0
        g_term = min((self.shear_modulus_gpa - 110.0) / 70.0, 1.0)
        pugh_term = min((self.pugh_ratio - 1.35) / 0.40, 1.0)
        return round(50.0 + 30.0 * g_term + 20.0 * pugh_term, 1)

    @property
    def heuristic_low_thermal_expansion(self) -> float:
        """
        Heuristic 3 (Negative or near-zero thermal expansion):
        Threshold: requires linear thermal expansion thermal_expansion_ppm <= 2.0 ppm/K.
        Returns 0.0 if thermal_expansion_ppm > 2.0 ppm/K.
        """
        if self.thermal_expansion_ppm > 2.0:
            return 0.0
        cte_term = min((2.0 - self.thermal_expansion_ppm) / 12.0, 1.0)
        return round(50.0 + 50.0 * cte_term, 1)

    @property
    def heuristic_auxetic(self) -> float:
        """
        Heuristic 4 (Auxetic material):
        Threshold: uses the strict definition of auxetic (poisson_ratio < 0.0).
        Returns 0.0 if poisson_ratio >= 0.0.
        """
        if self.poisson_ratio >= 0.0:
            return 0.0
        aux_term = min(abs(self.poisson_ratio) / 0.25, 1.0)
        return round(50.0 + 50.0 * aux_term, 1)

    def evaluate_heuristics(self) -> Tuple[str, float, List[str], str]:
        """
        Evaluates all four thresholded heuristics.
        If none reaches HEURISTIC_THRESHOLD (50.0), returns ("no anomaly", 0.0, [], ...).
        Placeholder rows never receive application areas.
        """
        candidates = [
            (
                "Dielectric thermal conductor",
                self.heuristic_dielectric_heat,
                ["Electronic packaging substrates", "Power-electronics heat spreaders"],
                f"Band gap {self.band_gap_ev} eV (>=3.0 eV) with thermal conductivity {self.thermal_cond_w_mk} W/m-K (>=50 W/m-K).",
            ),
            (
                "High-modulus moderate-Pugh phase",
                self.heuristic_damage_tolerant_hard,
                ["Thermal-shock-resistant structural parts", "Machinable high-temperature components"],
                f"Shear modulus {self.shear_modulus_gpa:.1f} GPa (>=110 GPa, E={self.youngs_modulus_gpa:.1f} GPa) with Pugh ratio {self.pugh_ratio:.2f} (>=1.35) and density {self.density_g_cm3} g/cm^3 (<=5.0).",
            ),
            (
                "Negative / low thermal expansion",
                self.heuristic_low_thermal_expansion,
                ["Athermal optical mounts", "Low-expansion composite fillers"],
                f"Linear thermal expansion {self.thermal_expansion_ppm:+.1f} ppm/K (<=2.0 ppm/K).",
            ),
            (
                "Auxetic (negative Poisson ratio)",
                self.heuristic_auxetic,
                ["Indentation-resistant pads", "Acoustic/mechanical dampers"],
                f"Negative isotropic Poisson ratio ({self.poisson_ratio:+.3f} < 0.0).",
            ),
        ]
        best_label, best_score, areas, summary = max(candidates, key=lambda item: item[1])
        if best_score < HEURISTIC_THRESHOLD:
            return (
                "no anomaly",
                0.0,
                [],
                "No heuristic threshold met (all heuristic scores < 50.0).",
            )
        if self.placeholder:
            return best_label, best_score, [], f"[UNVERIFIED PLACEHOLDER] {summary}"
        return best_label, best_score, areas, summary


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
            bulk_modulus_gpa=241.0,
            shear_modulus_gpa=193.0,
            thermal_expansion_ppm=5.6,
            thermal_cond_w_mk=30.0,
            band_gap_ev=2.09,
            source="Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), DOI:10.1111/j.1551-2916.2011.04865.x; Dodd, Saunders & James, J. Mater. Sci. 37(13), 2731-2736 (2002), DOI:10.1023/A:1015825318086; Werheit, J. Phys.: Condens. Matter 18(47), 10655-10662 (2006), DOI:10.1088/0953-8984/18/47/011",
            property_sources={
                "density_g_cm3": "Domnich et al. 2011, Sec. II p. 3606, DOI:10.1111/j.1551-2916.2011.04865.x (2.52 g/cm^3)",
                "bulk_modulus_gpa": "Dodd, Saunders & James 2002, Abstract & Table I p. 2733, DOI:10.1023/A:1015825318086 (241 GPa at 295 K)",
                "shear_modulus_gpa": "Dodd, Saunders & James 2002, Abstract & Table I p. 2733, DOI:10.1023/A:1015825318086 (193 GPa at 295 K)",
                "thermal_expansion_ppm": "Domnich et al. 2011, Sec. II, DOI:10.1111/j.1551-2916.2011.04865.x (5.6e-6 /K)",
                "thermal_cond_w_mk": "Domnich et al. 2011, Sec. II p. 3610, DOI:10.1111/j.1551-2916.2011.04865.x (~30 W/m-K at 300 K)",
                "band_gap_ev": "Werheit 2006, Abstract & Sec. 2 Fig. 1 p. 10656, DOI:10.1088/0953-8984/18/47/011 (2.09 eV)",
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
                "density_g_cm3": "Slack et al. 1987, Table 1 p. 642, DOI:10.1016/0022-3697(87)90153-3 (3.26 g/cm^3)",
                "bulk_modulus_gpa": "McNeil et al. 1993, Table II p. 1134, DOI:10.1111/j.1151-2916.1993.tb03730.x (202 GPa)",
                "shear_modulus_gpa": "McNeil et al. 1993, Table II p. 1134, DOI:10.1111/j.1151-2916.1993.tb03730.x (131 GPa)",
                "thermal_expansion_ppm": "Slack & Bartram, J. Appl. Phys. 46(1), 89-98 (1975), Table II p. 91, DOI:10.1063/1.321373 (4.5e-6 /K at 300 K)",
                "thermal_cond_w_mk": "Slack et al. 1987, Abstract & Sec. 4 p. 645, DOI:10.1016/0022-3697(87)90153-3 (285 W/m-K single crystal at 300 K)",
                "band_gap_ev": "Yim et al., J. Appl. Phys. 44(1), 292-296 (1973), Abstract & Fig. 4 p. 295, DOI:10.1063/1.1661876 (6.2 eV)",
            },
            placeholder=False,
        ),
        UniversalCompound(
            name="Ti3SiC2 (Titanium Silicon Carbide MAX Phase)",
            structure_class="MAX-Phase Nanolaminate",
            composition={"Ti": 3, "Si": 1, "C": 2},
            density_g_cm3=4.52,
            bulk_modulus_gpa=187.0,
            shear_modulus_gpa=142.0,
            thermal_expansion_ppm=9.1,
            thermal_cond_w_mk=37.0,
            band_gap_ev=0.0,
            source="Finkel, Barsoum & El-Raghy, J. Appl. Phys. 87(4), 1701-1703 (2000), DOI:10.1063/1.372080, as tabulated in Ganguly, PhD Thesis, Drexel Univ. (2006), Table 3.4 p. 61, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/Adrish_Ganguly.pdf; Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8",
            property_sources={
                "composition": "Stoichiometric polycrystal Ti3SiC2: Finkel, Barsoum & El-Raghy 2000, DOI:10.1063/1.372080; Barsoum et al. 1999, DOI:10.1016/S0022-3697(98)00313-8",
                "density_g_cm3": "4.52 g/cm^3 (polycrystal at 300 K): UNSOURCED in paywalled Barsoum, Prog. Solid State Chem. 28, 201-281 (2000), DOI:10.1016/S0079-6786(00)00006-6; cf. measured 4.49 Mg/m^3 (99% of theoretical 4.535 Mg/m^3) and ~4.5 g/cm^3 in Ganguly, PhD Thesis, Drexel Univ. (2006), pp. 29 & 36, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/Adrish_Ganguly.pdf",
                "bulk_modulus_gpa": "187 GPa (measured, polycrystal, room temperature ~300 K, ultrasonic): Finkel, Barsoum & El-Raghy, J. Appl. Phys. 87(4), 1701-1703 (2000), DOI:10.1063/1.372080, as tabulated in Ganguly, PhD Thesis, Drexel Univ. (2006), Table 3.4 p. 61, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/Adrish_Ganguly.pdf",
                "shear_modulus_gpa": "142 GPa (measured, polycrystal, room temperature ~300 K, ultrasonic): Finkel, Barsoum & El-Raghy, J. Appl. Phys. 87(4), 1701-1703 (2000), DOI:10.1063/1.372080, as tabulated in Ganguly, PhD Thesis, Drexel Univ. (2006), Table 3.4 p. 61, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/Adrish_Ganguly.pdf",
                "thermal_expansion_ppm": "9.1 +/- 0.2 ppm/K (measured, polycrystal, dilatometric mean over 25-1000 C): Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8, abstract at https://impact.ornl.gov/en/publications/thermal-properties-of-tisub3subsicsub2sub/",
                "thermal_cond_w_mk": "37 W/(m*K) (measured, polycrystal, room temperature 300 K): Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8, abstract at https://impact.ornl.gov/en/publications/thermal-properties-of-tisub3subsicsub2sub/",
                "band_gap_ev": "0.0 eV, metal, no gap (measured, polycrystal, electronic thermal transport dominated by delocalized electrons): Barsoum et al., J. Phys. Chem. Solids 60(4), 429-439 (1999), DOI:10.1016/S0022-3697(98)00313-8, abstract at https://impact.ornl.gov/en/publications/thermal-properties-of-tisub3subsicsub2sub/",
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
            band_gap_ev=5.60,
            source="Bruls et al., J. Phys. Chem. Solids 61(8), 1285-1292 (2000), DOI:10.1016/S0022-3697(99)00416-3; Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7; de Boer et al., Phys. Status Solidi RRL 9(4), 250-254 (2015), DOI:10.1002/pssr.201510043",
            property_sources={
                "density_g_cm3": "Bruls et al. 2000, Sec. 3.2 p. 1288, DOI:10.1016/S0022-3697(99)00416-3 (3.128 g/cm^3)",
                "bulk_modulus_gpa": "Bruls et al. 2001, Table 2 p. 266, DOI:10.1016/S0955-2219(00)00210-7 (177 GPa)",
                "shear_modulus_gpa": "Bruls et al. 2001, Table 2 p. 266, DOI:10.1016/S0955-2219(00)00210-7 (112 GPa)",
                "thermal_expansion_ppm": "Bruls et al. 2000, Sec. 3.4 Table 3 p. 1290, DOI:10.1016/S0022-3697(99)00416-3 (4.5e-6 /K at 300 K)",
                "thermal_cond_w_mk": "Bruls, Ph.D. Thesis, TU Eindhoven (2000), Ch. 7 p. 125, DOI:10.6100/IR535906 (23 W/m-K experimental sintered max)",
                "band_gap_ev": "de Boer et al. 2015, Abstract & Sec. 3 p. 252, DOI:10.1002/pssr.201510043 (5.6 eV experimental X-ray spectroscopy)",
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
            band_gap_ev=3.61,
            source="Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; Kennedy & White, Solid State Commun. 134(4), 271-276 (2005), DOI:10.1016/j.ssc.2005.01.031",
            property_sources={
                "density_g_cm3": "Mary et al. 1996, Note 11 p. 92, DOI:10.1126/science.272.5258.90 (5.07 g/cm^3)",
                "bulk_modulus_gpa": "Drymiotis et al. 2004, Table I p. 025502-2, DOI:10.1103/PhysRevLett.93.025502 (74.5 GPa at 300 K)",
                "shear_modulus_gpa": "Drymiotis et al. 2004, Table I p. 025502-2, DOI:10.1103/PhysRevLett.93.025502 (33.8 GPa at 300 K)",
                "thermal_expansion_ppm": "Mary et al. 1996, Abstract p. 90, DOI:10.1126/science.272.5258.90 (-8.7e-6 /K isotropic from 0.3 to 1050 K)",
                "thermal_cond_w_mk": "Kennedy & White 2005, Abstract & Fig. 1 p. 273, DOI:10.1016/j.ssc.2005.01.031 (0.51 W/m-K at 300 K)",
                "band_gap_ev": "Materials Project mp-18778 (Zr(WO4)2 cubic P2_13 summary record), DOI:10.17188/1193473 (3.61 eV DFT)",
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
            source="Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650; Swainson & Dove, Phys. Chem. Miner. 22, 61-65 (1995), DOI:10.1007/BF00202681",
            property_sources={
                "density_g_cm3": "Yeganeh-Haeri et al. 1992, Table 1 p. 651, DOI:10.1126/science.257.5070.650 (2.33 g/cm^3)",
                "bulk_modulus_gpa": "Yeganeh-Haeri et al. 1992, Table 2 p. 651, DOI:10.1126/science.257.5070.650 (16.4 GPa VRH aggregate at 300 K)",
                "shear_modulus_gpa": "Yeganeh-Haeri et al. 1992, Table 2 p. 651, DOI:10.1126/science.257.5070.650 (39.1 GPa VRH aggregate at 300 K -> Poisson ratio -0.164)",
                "thermal_expansion_ppm": "Swainson & Dove 1995, Fig. 2 p. 63, DOI:10.1007/BF00202681 (~10.3e-6 /K at 300 K)",
                "thermal_cond_w_mk": "CRC Handbook of Chemistry and Physics, 104th Ed. (~1.4 W/m-K for crystalline/vitreous silica at 300 K)",
                "band_gap_ev": "Laughlin, Phys. Rev. B 22(6), 3021-3029 (1980), Sec. III p. 3025, DOI:10.1103/PhysRevB.22.3021 (8.9 eV)",
            },
            placeholder=False,
        ),
    ]


def generate_placeholder_candidates(
    count: int = 4,
    rng: Optional[random.Random] = None,
    start_index: int = 1,
) -> List[UniversalCompound]:
    """
    Generates synthetic, unverified placeholder rows using a deterministic
    `random.Random` instance for UI/pipeline testing.
    Every returned entry has `placeholder=True` and carries no literature citations.
    """
    if count <= 0:
        return []
    if rng is None:
        rng = random.Random(0)

    cations = ["Fe", "Al", "Mg", "Si", "Ca", "Na", "Mn", "Zn"]
    anions = ["B", "C", "N", "O", "P", "S"]
    results: List[UniversalCompound] = []

    for offset in range(count):
        idx = start_index + offset
        m1, m2 = rng.sample(cations, 2)
        x1, x2 = rng.sample(anions, 2)
        comp: Dict[str, float] = {
            m1: float(rng.randint(1, 3)),
            m2: float(rng.randint(1, 2)),
            x1: float(rng.randint(1, 4)),
            x2: float(rng.randint(1, 3)),
        }
        formula = "".join(f"{el}{int(cnt)}" for el, cnt in comp.items())
        avg_mass = sum(ATOMIC_MASS[el] * n for el, n in comp.items()) / sum(comp.values())
        density = round(max(2.1, min(6.0, avg_mass * rng.uniform(0.085, 0.125))), 2)
        bulk = round(rng.uniform(60.0, 210.0), 1)
        shear = round(rng.uniform(30.0, 140.0), 1)
        cte = round(rng.uniform(-2.0, 12.0), 2)
        t_cond = round(rng.uniform(2.0, 90.0), 1)
        b_gap = round(rng.choice([0.0, rng.uniform(0.5, 5.0)]), 2)

        results.append(
            UniversalCompound(
                name=f"Placeholder-{formula} (Synthetic #{idx})",
                structure_class="Synthetic Placeholder (Unverified)",
                composition=comp,
                density_g_cm3=density,
                bulk_modulus_gpa=bulk,
                shear_modulus_gpa=shear,
                thermal_expansion_ppm=cte,
                thermal_cond_w_mk=t_cond,
                band_gap_ev=b_gap,
                source="Synthetic placeholder (generated via random.Random; not experimental or DFT data)",
                property_sources={},
                placeholder=True,
            )
        )
    return results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate literature-sourced materials catalog and optional synthetic placeholders."
    )
    parser.add_argument(
        "--placeholders",
        type=int,
        default=0,
        help="Number of synthetic placeholder rows to generate in a separate unranked table (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic placeholder generation (default: 42).",
    )
    args = parser.parse_args(argv)

    catalog = get_universal_catalog()
    ranked = sorted(catalog, key=lambda c: c.evaluate_heuristics()[1], reverse=True)

    print("=" * 118)
    print("  LITERATURE-SOURCED MATERIALS CATALOG: DERIVED PROPERTIES & HEURISTIC SCORES")
    print("=" * 118)
    print(
        f"{'Rank':<4} | {'Material':<42} | {'Cost/kg':<9} | {'Poisson':<7} | "
        f"{'Top Heuristic Classification':<36} | {'Score'}"
    )
    print("-" * 118)
    for idx, c in enumerate(ranked, 1):
        label, score, areas, _ = c.evaluate_heuristics()
        print(
            f"#{idx:<3} | {c.name[:42]:<42} | ${c.raw_cost_per_kg:>7.2f} | "
            f"{c.poisson_ratio:>+7.3f} | {label:<36} | {score:>5.1f}"
        )
        if areas:
            print(f"     └─ Possible application areas: {', '.join(areas)}")
        print(f"     └─ Source: {c.source}")
    print("=" * 118)

    if args.placeholders > 0:
        rng = random.Random(args.seed)
        placeholders = generate_placeholder_candidates(count=args.placeholders, rng=rng)
        print("\n" + "=" * 118)
        print("  SYNTHETIC PLACEHOLDER CANDIDATES (UNVERIFIED RANDOM NUMBERS - UNRANKED)")
        print("=" * 118)
        for p in placeholders:
            label, score, _, _ = p.evaluate_heuristics()
            print(
                f"[PLACEHOLDER] {p.name:<38} | formula={p.formula:<12} | "
                f"cost=${p.raw_cost_per_kg:>6.2f}/kg | heuristic={label} ({score:.1f})"
            )
        print("=" * 118)


if __name__ == "__main__":
    main()
