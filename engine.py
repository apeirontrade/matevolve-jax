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

# Approximate elemental prices ($ USD per kg) used ONLY for coarse display estimates.
# Cost does not feed any heuristic.
# Where available, commodity reference prices and forms are cited to USGS Mineral Commodity
# Summaries 2026 (DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf);
# elements not priced in elemental form by USGS MCS 2026 (or where the nominal figure differs
# from bulk mineral concentrate/compound prices) are explicitly tagged UNSOURCED in
# ELEMENT_COST_SOURCES below.
ELEMENT_COST_USD_PER_KG: Dict[str, float] = {
    "Fe": 0.50,   "Si": 2.20,   "Al": 2.40,   "C":  1.50,   "N":  0.40,
    "O":  0.20,   "Mg": 3.10,   "Ca": 2.80,   "Na": 2.50,   "K":  3.00,
    "Mn": 2.10,   "S":  0.30,   "P":  3.20,   "Zn": 2.90,   "Cu": 9.20,
    "Cr": 8.50,   "B":  300.00, "Ti": 11.50,  "Zr": 22.00,  "V":  28.00,
    "Ni": 16.50,  "Mo": 45.00,  "Sn": 29.00,  "W":  40.00,  "Nb": 42.00,
}

ELEMENT_COST_SOURCES: Dict[str, str] = {
    "Al": "USGS Mineral Commodity Summaries 2026, p. 22, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Aluminum ingot, U.S. market spot: 127.8 cents/lb = $2.82/kg in 2024; 175 cents/lb = $3.86/kg in 2025e; table nominal $2.40/kg UNSOURCED)",
    "B": "UNSOURCED ($300.00/kg elemental amorphous boron figure is unsourced; USGS MCS 2026 p. 42, DOI:10.3133/mcs2026 prices refined borates/boric acid import/export at $810/t = $0.81/kg in 2024 and $800/t = $0.80/kg in 2025e, not elemental boron)",
    "C": "USGS Mineral Commodity Summaries 2026, p. 80, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Natural flake graphite imports $1,420/t = $1.42/kg in 2024, $1,200/t = $1.20/kg in 2025e; table nominal $1.50/kg UNSOURCED)",
    "Ca": "UNSOURCED ($2.80/kg calcium metal figure is unsourced; USGS MCS 2026 p. 106, DOI:10.3133/mcs2026 prices quicklime CaO at $197/t = $0.20/kg in 2024, not elemental Ca metal)",
    "Cr": "USGS Mineral Commodity Summaries 2026, p. 52, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Chromium metal imports: $15,615/t = $15.62/kg in 2024, $15,000/t = $15.00/kg in 2025e; table nominal $8.50/kg UNSOURCED)",
    "Cu": "USGS Mineral Commodity Summaries 2026, p. 58, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Copper cathode COMEX: 414 cents/lb = $9.13/kg in 2024, 460 cents/lb = $10.14/kg in 2025e; table nominal $9.20/kg)",
    "Fe": "USGS Mineral Commodity Summaries 2026, p. 94, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (No. 1 heavy melting steel scrap: $355/t = $0.36/kg in 2024, $330/t = $0.33/kg in 2025e; table nominal $0.50/kg UNSOURCED)",
    "K": "UNSOURCED ($3.00/kg elemental potassium metal; USGS MCS 2026 p. 136, DOI:10.3133/mcs2026 prices muriate of potash K2O equivalent at $360/t = $0.36/kg in 2024)",
    "Mg": "USGS Mineral Commodity Summaries 2026, p. 112, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Primary magnesium metal ingot, China/Europe: $2,700-$3,000/t = $2.70-$3.00/kg in 2024; table nominal $3.10/kg UNSOURCED)",
    "Mn": "USGS Mineral Commodity Summaries 2026, p. 114, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Electrolytic manganese metal imports; table nominal $2.10/kg UNSOURCED)",
    "Mo": "USGS Mineral Commodity Summaries 2026, p. 120, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Molybdenum contained in technical-grade molybdic oxide: $47.20/kg in 2024; table nominal $45.00/kg UNSOURCED)",
    "N": "UNSOURCED ($0.40/kg nitrogen; USGS MCS 2026 p. 124, DOI:10.3133/mcs2026 prices anhydrous ammonia NH3 Gulf Coast at $495/short ton = $0.55/kg in 2024, $510/st = $0.56/kg in 2025e)",
    "Na": "UNSOURCED ($2.50/kg sodium metal; USGS MCS 2026 p. 162, DOI:10.3133/mcs2026 prices soda ash Na2CO3 at $205.51/t = $0.21/kg in 2024, not metallic Na)",
    "Nb": "USGS Mineral Commodity Summaries 2026, p. 122, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Niobium contained in ferroniobium imports; table nominal $42.00/kg UNSOURCED)",
    "Ni": "USGS Mineral Commodity Summaries 2026, p. 122/126, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Nickel LME cash: $16,812/t = $16.81/kg in 2024; table nominal $16.50/kg UNSOURCED)",
    "O": "UNSOURCED ($0.20/kg industrial oxygen estimate)",
    "P": "UNSOURCED ($3.20/kg elemental yellow phosphorus; USGS MCS 2026 p. 130, DOI:10.3133/mcs2026 prices marketable phosphate rock at $101/t = $0.10/kg in 2024, not elemental P)",
    "S": "UNSOURCED ($0.30/kg elemental sulfur; USGS MCS 2026 p. 170, DOI:10.3133/mcs2026 prices elemental sulfur f.o.b. plant at ~$73/t = $0.07/kg in 2024)",
    "Si": "USGS Mineral Commodity Summaries 2026, p. 158, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Silicon metal <99.99% Si: 158 cents/lb = $3.48/kg in 2024, 140 cents/lb = $3.09/kg in 2025e; table nominal $2.20/kg UNSOURCED)",
    "Sn": "USGS Mineral Commodity Summaries 2026, p. 180, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Refined tin metal LME/New York dealer: ~$30/kg in 2024; table nominal $29.00/kg UNSOURCED)",
    "Ti": "USGS Mineral Commodity Summaries 2026, p. 182, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Titanium sponge metal imports, unit value: $12.10/kg in 2024, $11.50/kg in 2025e)",
    "V": "USGS Mineral Commodity Summaries 2026, p. 190, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Ferrovanadium U.S. 78-82% V: $29.39/kg V in 2024, $28.70/kg V in 2025e; V2O5 Chinese 98%: $5.72/lb = $12.61/kg in 2024; table nominal $28.00/kg UNSOURCED)",
    "W": "USGS Mineral Commodity Summaries 2026, p. 188, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Tungsten concentrate / ammonium paratungstate APT ~$330-$340/mtu WO3 in 2024; table nominal $40.00/kg W metal powder UNSOURCED)",
    "Zn": "USGS Mineral Commodity Summaries 2026, p. 196, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Special High Grade zinc metal LME/North American: ~134 cents/lb = $2.95/kg in 2024; table nominal $2.90/kg)",
    "Zr": "USGS Mineral Commodity Summaries 2026, p. 198, DOI:10.3133/mcs2026, https://pubs.usgs.gov/periodicals/mcs2026/mcs2026.pdf (Unwrought zirconium powder/sponge imports: $29/kg in 2024, $25/kg in 2025e; table nominal $22.00/kg UNSOURCED)",
}

# Standard abridged atomic weights rounded to two decimal places:
# CIAAW 2024 (https://ciaaw.org/abridged-atomic-weights.htm; Prohaska et al., Pure Appl. Chem.
# 94(5), 573-600, 2022, DOI:10.1515/pac-2019-0603).
ATOMIC_MASS: Dict[str, float] = {
    "Fe": 55.85, "Si": 28.09, "Al": 26.98, "C": 12.01, "N": 14.01,
    "O":  16.00, "Mg": 24.31, "Ca": 40.08, "Na": 22.99, "K": 39.10,
    "Mn": 54.94, "S":  32.06, "P":  30.97, "Zn": 65.38, "Cu": 63.55,
    "Cr": 52.00, "B":  10.81, "Ti": 47.87, "Zr": 91.22, "V": 50.94,
    "Ni": 58.69, "Mo": 95.95, "Sn": 118.71, "W": 183.84, "Nb": 92.91,
}


@dataclass
class MaterialEntry:
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

    def __post_init__(self) -> None:
        if not self.composition:
            raise ValueError("composition must not be empty.")
        for el, count in self.composition.items():
            if el not in ATOMIC_MASS or el not in ELEMENT_COST_USD_PER_KG:
                raise ValueError(f"Unknown element in composition: {el!r}")
            if count <= 0:
                raise ValueError(f"Element count for {el!r} must be positive, got {count}.")
        if self.density_g_cm3 <= 0:
            raise ValueError(f"density_g_cm3 must be positive, got {self.density_g_cm3}.")
        if self.bulk_modulus_gpa <= 0:
            raise ValueError(f"bulk_modulus_gpa must be positive, got {self.bulk_modulus_gpa}.")
        if self.shear_modulus_gpa <= 0:
            raise ValueError(f"shear_modulus_gpa must be positive, got {self.shear_modulus_gpa}.")

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
        return self.bulk_modulus_gpa / self.shear_modulus_gpa

    @property
    def poisson_ratio(self) -> float:
        """Isotropic Poisson's ratio from B and G: (3B - 2G) / (2(3B + G))."""
        b, g = self.bulk_modulus_gpa, self.shear_modulus_gpa
        return (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))

    @property
    def vickers_hardness_gpa(self) -> Optional[float]:
        """
        Chen's empirical Vickers hardness correlation: Hv = 2*(k^2 * G)^0.585 - 3, k = G/B
        (Chen et al., Intermetallics 19, 1275, 2011, DOI:10.1016/j.intermet.2011.03.026).
        Returns None when G/B > 1.0 or poisson_ratio < 0.0 (outside the valid domain of
        Chen's correlation, e.g. auxetic alpha-cristobalite).
        Note: Fit on brittle covalent/ionic crystals; overestimates indentation hardness
        in layered MAX/MAB phases and metals where dislocation slip occurs.
        """
        k = self.shear_modulus_gpa / self.bulk_modulus_gpa
        if k > 1.0 or self.poisson_ratio < 0.0:
            return None
        return 2.0 * ((k ** 2) * self.shear_modulus_gpa) ** 0.585 - 3.0

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
    def heuristic_elastic_screen(self) -> float:
        """
        Heuristic 2 (Elastic screen: Hv estimate >= 15, B/G >= 1.35, density <= 5):
        Threshold: requires Chen's Hv estimate vickers_hardness_gpa >= 15.0 GPa,
        Pugh ratio B/G >= 1.35 (above brittle ceramics like B4C at 1.24), and
        density_g_cm3 <= 5.0 g/cm^3.
        Returns 0.0 for MAX phases, MAB phases, and metallic alloys (where Chen's
        brittle-crystal correlation is invalid), or if any threshold is not met.
        """
        cls_lower = self.structure_class.lower()
        if any(token in cls_lower for token in ("max", "mab", "alloy", "metal")):
            return 0.0
        hv = self.vickers_hardness_gpa
        if (
            hv is None
            or hv < 15.0
            or self.pugh_ratio < 1.35
            or self.density_g_cm3 > 5.0
        ):
            return 0.0
        hv_term = min((hv - 15.0) / 20.0, 1.0)
        pugh_term = min((self.pugh_ratio - 1.35) / 0.40, 1.0)
        return round(50.0 + 30.0 * hv_term + 20.0 * pugh_term, 1)

    @property
    def heuristic_low_thermal_expansion(self) -> float:
        """
        Heuristic 3 (Negative / low thermal expansion):
        Threshold: requires linear thermal expansion thermal_expansion_ppm <= 2.0 ppm/K.
        Note: The formula (2.0 - thermal_expansion_ppm) / 12.0 linearly rewards
        increasingly negative thermal expansion below +2.0 ppm/K (reaching 100 at
        -10.0 ppm/K) rather than closeness to zero (|alpha| -> 0).
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
        Threshold: uses the strict definition of auxetic (poisson_ratio < 0.0;
        Evans, Endeavour 15(4), 170-174, 1991, DOI:10.1016/0160-9327(91)90123-S).
        Returns 0.0 if poisson_ratio >= 0.0.
        """
        if self.poisson_ratio >= 0.0:
            return 0.0
        aux_term = min(abs(self.poisson_ratio) / 0.25, 1.0)
        return round(50.0 + 50.0 * aux_term, 1)

    def evaluate_heuristics(self) -> Tuple[str, Optional[float], List[str], str]:
        """
        Evaluates all four heuristics. Each heuristic either passes its documented
        property cut-offs (producing a score in 50-100) or fails them (returning 0.0).
        When no cut-off is met (all scores 0.0), returns ("no anomaly", 0.0, [], ...).
        When placeholder is True, heuristics are not evaluated and returns
        ("not evaluated (placeholder)", None, [], ...).
        """
        if self.placeholder:
            return (
                "not evaluated (placeholder)",
                None,
                [],
                "Synthetic placeholder row (heuristics not evaluated on random numbers).",
            )
        candidates = [
            (
                "Dielectric thermal conductor",
                self.heuristic_dielectric_heat,
                ["Electronic packaging substrates", "Power-electronics heat spreaders"],
                f"Band gap {self.band_gap_ev} eV (>=3.0 eV) with thermal conductivity {self.thermal_cond_w_mk} W/m-K (>=50 W/m-K).",
            ),
            (
                "Elastic screen: Hv estimate >= 15, B/G >= 1.35, density <= 5",
                self.heuristic_elastic_screen,
                [],
                (
                    f"Chen Hv estimate {self.vickers_hardness_gpa:.1f} GPa (>=15.0 GPa) "
                    f"with Pugh ratio {self.pugh_ratio:.2f} (>=1.35) and density {self.density_g_cm3} g/cm^3 (<=5.0)."
                    if self.vickers_hardness_gpa is not None
                    else "Chen Hv estimate n/a (G/B > 1 or nu < 0)."
                ),
            ),
            (
                "Negative / low thermal expansion",
                self.heuristic_low_thermal_expansion,
                ["Low-expansion composite fillers"],
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
        if best_score == 0.0:
            return (
                "no anomaly",
                0.0,
                [],
                "No heuristic cut-off met (passing a heuristic's documented cut-offs yields a score in 50-100).",
            )
        return best_label, best_score, areas, summary


def get_catalog() -> List[MaterialEntry]:
    """
    Returns literature-sourced catalog entries where every numeric property is tied
    to an explicit citation in `property_sources`.
    """
    return [
        MaterialEntry(
            name="Ti-6Al-4V (Grade 5 Titanium Alloy Baseline)",
            structure_class="Alpha-Beta Titanium Alloy (Baseline)",
            # Atomic percentages reproducing 90.0 wt% Ti, 6.0 wt% Al, 4.0 wt% V
            composition={"Ti": 86.2, "Al": 10.2, "V": 3.6},
            density_g_cm3=4.43,
            bulk_modulus_gpa=113.0,
            shear_modulus_gpa=45.0,
            thermal_expansion_ppm=8.6,
            thermal_cond_w_mk=6.7,
            band_gap_ev=0.0,
            source="ASM Aerospace Specification Metals Inc., Titanium Ti-6Al-4V (Grade 5) Annealed Datasheet, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf; Driver et al., Philos. Mag. 96(22), 2311-2327 (2016), DOI:10.1080/14786435.2016.1198054",
            property_sources={
                "composition": "Nominal Grade 5 composition 90 wt% Ti, 6 wt% Al, 4 wt% V (converted via CIAAW atomic weights to 86.2 at% Ti, 10.2 at% Al, 3.6 at% V): ASM Aerospace Specification Metals Datasheet p. 1, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
                "density_g_cm3": "4.43 g/cm^3 (measured, annealed polycrystal alloy at room temperature): ASM Aerospace Specification Metals Datasheet p. 1, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
                "bulk_modulus_gpa": "113.0 GPa (measured, polycrystal at room temperature ~295 K, resonant ultrasound spectroscopy paired with G = 45.0 GPa): Driver et al., Philos. Mag. 96(22), 2311-2327 (2016), Table 2, DOI:10.1080/14786435.2016.1198054 (UNSOURCED in paywalled Table 2; cf. E = 113.8 GPa, G = 44 GPa, nu = 0.342 at https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf)",
                "shear_modulus_gpa": "45.0 GPa (measured, polycrystal at room temperature ~295 K, resonant ultrasound spectroscopy paired with B = 113.0 GPa): Driver et al., Philos. Mag. 96(22), 2311-2327 (2016), Table 2, DOI:10.1080/14786435.2016.1198054 (UNSOURCED in paywalled Table 2; cf. G = 44 GPa at https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf)",
                "thermal_expansion_ppm": "8.6 ppm/K (measured, annealed polycrystal alloy, linear mean over 20-100 C): ASM Aerospace Specification Metals Datasheet p. 1, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
                "thermal_cond_w_mk": "6.7 W/(m*K) (measured, annealed polycrystal alloy at room temperature): ASM Aerospace Specification Metals Datasheet p. 1, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
                "band_gap_ev": "0.0 eV, metal, no gap (measured annealed polycrystal alloy, electrical resistivity 0.000178 ohm-cm at room temperature): ASM Aerospace Specification Metals Datasheet p. 1, https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="B4C (Boron Carbide Ceramic Baseline)",
            structure_class="Rhombohedral Covalent Ceramic (Baseline)",
            composition={"B": 4, "C": 1},
            density_g_cm3=2.52,
            bulk_modulus_gpa=247.0,
            shear_modulus_gpa=200.0,
            thermal_expansion_ppm=5.65,
            thermal_cond_w_mk=30.0,
            band_gap_ev=2.09,
            source="Gieske, Aselage & Emin, AIP Conf. Proc. 231, 376-379 (1991), DOI:10.1063/1.40854, as tabulated in Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), DOI:10.1111/j.1551-2916.2011.04865.x; Yakel, J. Appl. Cryst. 6, 471-472 (1973), DOI:10.1107/S0021889873009246; Mahagin et al., HEDL-TME 73-78 (1973), DOI:10.2172/4389842, https://www.osti.gov/servlets/purl/4389842; Werheit, J. Phys.: Condens. Matter 18(47), 10655-10662 (2006), DOI:10.1088/0953-8984/18/47/011",
            property_sources={
                "composition": "B4C (20 at.% C), carbon-rich limit of the rhombohedral boron carbide solid-solution range (~9 to 20 at.% C): Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), abstract, DOI:10.1111/j.1551-2916.2011.04865.x; Yakel, J. Appl. Cryst. 6, 471-472 (1973), abstract, DOI:10.1107/S0021889873009246",
                "density_g_cm3": "2.52 g/cm^3 (theoretical crystal density at 300 K): UNSOURCED in paywalled Domnich et al. 2011, DOI:10.1111/j.1551-2916.2011.04865.x; cf. measured 2.49 g/cm^3 at 0.8 vol.% porosity (~2.51 g/cm^3 theoretical) for natural B4C specimen B-H8 in Mahagin et al., HEDL-TME 73-78 (1973), Table I p. 11, DOI:10.2172/4389842, https://www.osti.gov/servlets/purl/4389842",
                "bulk_modulus_gpa": "247 GPa (measured, polycrystal at room temperature ~300 K, ultrasonic paired with G = 200 GPa): Gieske, Aselage & Emin, AIP Conf. Proc. 231, 376-379 (1991), DOI:10.1063/1.40854, as tabulated in Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), Table IV, DOI:10.1111/j.1551-2916.2011.04865.x (UNSOURCED in paywalled full text)",
                "shear_modulus_gpa": "200 GPa (measured, polycrystal at room temperature ~300 K, ultrasonic paired with B = 247 GPa): Gieske, Aselage & Emin, AIP Conf. Proc. 231, 376-379 (1991), DOI:10.1063/1.40854, as tabulated in Domnich et al., J. Am. Ceram. Soc. 94(11), 3605-3628 (2011), Table IV, DOI:10.1111/j.1551-2916.2011.04865.x (UNSOURCED in paywalled full text)",
                "thermal_expansion_ppm": "5.65 ppm/K (measured, polycrystal/powder, high-temperature XRD mean linear thermal expansion over 12-940 C): Yakel, J. Appl. Cryst. 6, 471-472 (1973), abstract, DOI:10.1107/S0021889873009246",
                "thermal_cond_w_mk": "30 W/(m*K) (approx., measured on hot-pressed polycrystal, 26-37 W/(m*K) at 293-300 K depending on sample; exact rounded 30 UNSOURCED): Mahagin, Bates & Baker, HEDL-TME 73-78 (1973), Table I p. 11 & Table A.I pp. A-1 to A-9, DOI:10.2172/4389842, https://www.osti.gov/servlets/purl/4389842",
                "band_gap_ev": "2.09 eV (measured optical absorption gap at room temperature; reported gaps for B4C range from 0.48 to >3 eV): Werheit, J. Phys.: Condens. Matter 18(47), 10655-10662 (2006), abstract, DOI:10.1088/0953-8984/18/47/011",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="AlN (Wurtzite Aluminum Nitride)",
            structure_class="Wurtzite Wide-Bandgap Ceramic",
            composition={"Al": 1, "N": 1},
            density_g_cm3=3.26,
            bulk_modulus_gpa=211.0,
            shear_modulus_gpa=133.0,
            thermal_expansion_ppm=4.5,
            thermal_cond_w_mk=285.0,
            band_gap_ev=6.20,
            source="McNeil, Grimsditch & French, J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x, as tabulated in Reeber & Wang, MRS Internet J. Nitride Semicond. Res. 6, e3 (2001), DOI:10.1557/S1092578300000156; Slack et al., J. Phys. Chem. Solids 48(7), 641-647 (1987), DOI:10.1016/0022-3697(87)90153-3, as cited in Bruls, PhD Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906; Yim & Paff, J. Appl. Phys. 45(3), 1456-1457 (1974), DOI:10.1063/1.1663432; Yim et al., J. Appl. Phys. 44(1), 292-296 (1973), DOI:10.1063/1.1661876",
            property_sources={
                "composition": "Stoichiometric wurtzite P6_3mc AlN: Reeber & Wang, MRS Internet J. Nitride Semicond. Res. 6, e3 (2001), p. 1, DOI:10.1557/S1092578300000156, https://link.springer.com/content/pdf/10.1557/S1092578300000156.pdf",
                "density_g_cm3": "3.26 g/cm^3 (measured/theoretical polycrystal 3.258e3 kg/m^3 at 293 K): Bruls, PhD Thesis, TU Eindhoven (2000), Table 6-2 p. 144, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "bulk_modulus_gpa": "211.0 GPa (derived Voigt-Reuss-Hill isotropic average B = 210.7 GPa from measured single-crystal Brillouin scattering C11=410.5, C12=148.5, C13=98.9, C33=388.5, C44=124.6 GPa at room temperature ~298 K): McNeil, Grimsditch & French, J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x, as tabulated in Reeber & Wang, MRS Internet J. Nitride Semicond. Res. 6, e3 (2001), Table 1 p. 3, DOI:10.1557/S1092578300000156, https://link.springer.com/content/pdf/10.1557/S1092578300000156.pdf",
                "shear_modulus_gpa": "133.0 GPa (derived Voigt-Reuss-Hill isotropic average G = 133.0 GPa from measured single-crystal Brillouin scattering C11=410.5, C12=148.5, C13=98.9, C33=388.5, C44=124.6 GPa at room temperature ~298 K): McNeil, Grimsditch & French, J. Am. Ceram. Soc. 76(5), 1132-1136 (1993), DOI:10.1111/j.1151-2916.1993.tb03730.x, as tabulated in Reeber & Wang, MRS Internet J. Nitride Semicond. Res. 6, e3 (2001), Table 1 p. 3, DOI:10.1557/S1092578300000156, https://link.springer.com/content/pdf/10.1557/S1092578300000156.pdf",
                "thermal_expansion_ppm": "4.5 ppm/K (measured wide-range mean over 20-800 C on powder by X-ray diffraction, between alpha_parallel = 4.2 ppm/K and alpha_perp = 5.3 ppm/K; isotropic average 4.93 ppm/K): Yim & Paff, J. Appl. Phys. 45(3), 1456-1457 (1974), abstract, DOI:10.1063/1.1663432, https://api.crossref.org/works/10.1063/1.1663432",
                "thermal_cond_w_mk": "285 W/(m*K) (measured, single crystal at 300 K; note that sintered ceramic substrates are lower): Slack et al., J. Phys. Chem. Solids 48(7), 641-647 (1987), DOI:10.1016/0022-3697(87)90153-3, as tabulated in Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 8 p. 191 & Ch. 9 p. 225, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "band_gap_ev": "6.2 eV (measured, epitaxial single-crystal films on sapphire by optical absorption at 300 K): Yim et al., J. Appl. Phys. 44(1), 292-296 (1973), abstract, DOI:10.1063/1.1661876, https://api.crossref.org/works/10.1063/1.1661876",
            },
            placeholder=False,
        ),
        MaterialEntry(
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
        MaterialEntry(
            name="Cr2AlC (Chromium Aluminum Carbide MAX Phase)",
            structure_class="MAX-Phase Nanolaminate",
            composition={"Cr": 2, "Al": 1, "C": 1},
            density_g_cm3=5.24,
            bulk_modulus_gpa=138.0,
            shear_modulus_gpa=105.0,
            thermal_expansion_ppm=12.6,
            thermal_cond_w_mk=22.0,
            band_gap_ev=0.0,
            source="Hettinger et al., Phys. Rev. B 72, 115120 (2005), DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf; Scabarozi et al., J. Appl. Phys. 105, 013543 (2009), DOI:10.1063/1.3021465",
            property_sources={
                "composition": "Stoichiometric polycrystal Cr2AlC: Hettinger et al., Phys. Rev. B 72, 115120 (2005), Sec. II p. 1 & Table I p. 5, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
                "density_g_cm3": "5.24 g/cm^3 (measured, polycrystal at room temperature ~300 K): Hettinger et al., Phys. Rev. B 72, 115120 (2005), Table I p. 5, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
                "bulk_modulus_gpa": "138 GPa (measured, polycrystal at room temperature ~300 K, ultrasonic pulse-echo paired with G = 105 GPa): Hettinger et al., Phys. Rev. B 72, 115120 (2005), Table I p. 5, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
                "shear_modulus_gpa": "105 GPa (measured, polycrystal at room temperature ~300 K, ultrasonic pulse-echo paired with B = 138 GPa): Hettinger et al., Phys. Rev. B 72, 115120 (2005), Table I p. 5, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
                "thermal_expansion_ppm": "12.6 ppm/K (measured, polycrystal, high-temperature XRD mean over 25-800 C): UNSOURCED in paywalled Scabarozi et al., J. Appl. Phys. 105, 013543 (2009), DOI:10.1063/1.3021465; cf. ~12e-6 K^-1 read in Hettinger et al. 2005, Sec. IV.C p. 5, DOI:10.1103/PhysRevB.72.115120",
                "thermal_cond_w_mk": "~22 W/(m*K) (measured, polycrystal at room temperature ~300 K): Hettinger et al., Phys. Rev. B 72, 115120 (2005), Sec. III p. 2, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
                "band_gap_ev": "0.0, metal, no gap (measured, polycrystal, metallic electrical resistivity dropping linearly with decreasing temperature): Hettinger et al., Phys. Rev. B 72, 115120 (2005), Sec. IV.A p. 2 & Fig. 1, DOI:10.1103/PhysRevB.72.115120, https://research.coe.drexel.edu/mse/layeredsolids/wp-content/uploads/2011/01/HettingerPRB-2005_NSF.pdf",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="Fe2AlB2 (Iron Aluminum Boride MAB Phase)",
            structure_class="MAB-Phase Nanolaminate",
            composition={"Fe": 2, "Al": 1, "B": 2},
            density_g_cm3=5.75,
            bulk_modulus_gpa=162.4,
            shear_modulus_gpa=107.7,
            thermal_expansion_ppm=10.3,
            thermal_cond_w_mk=7.8,
            band_gap_ev=0.0,
            source="Bai et al., J. Am. Ceram. Soc. 103(10), 5837-5851 (2020), DOI:10.1111/jace.17205; Lamichhane et al., Phys. Rev. Materials 2, 084408 (2018), DOI:10.1103/PhysRevMaterials.2.084408, https://arxiv.org/pdf/1805.06373; Verger et al., J. Appl. Phys. 124, 205108 (2018), DOI:10.1063/1.5054379; Jeitschko, Acta Cryst. B 25, 163 (1969), DOI:10.1107/S0567740869001944",
            property_sources={
                "composition": "Orthorhombic Cmmm Fe2AlB2 (AlB2Fe2): Jeitschko, Acta Cryst. B 25, 163-165 (1969), DOI:10.1107/S0567740869001944, as cited in Lamichhane et al. 2018 p. 1, DOI:10.1103/PhysRevMaterials.2.084408; Materials Project mp-3805, https://optimade.materialsproject.org/v1/structures/mp-3805",
                "density_g_cm3": "5.75 g/cm^3 (measured X-ray density from single-crystal diffraction at 200 K): Lamichhane et al., Phys. Rev. Materials 2, 084408 (2018), Table I p. 3, DOI:10.1103/PhysRevMaterials.2.084408, https://arxiv.org/pdf/1805.06373",
                "bulk_modulus_gpa": "162.4 GPa (measured, polycrystal at 298 K, ultrasonic): Bai et al., J. Am. Ceram. Soc. 103(10), 5837-5851 (2020), abstract, DOI:10.1111/jace.17205, https://api.crossref.org/works/10.1111/jace.17205",
                "shear_modulus_gpa": "107.7 GPa (measured, polycrystal at 298 K, ultrasonic): Bai et al., J. Am. Ceram. Soc. 103(10), 5837-5851 (2020), abstract, DOI:10.1111/jace.17205, https://api.crossref.org/works/10.1111/jace.17205",
                "thermal_expansion_ppm": "10.3(2) ppm/K (measured, powder, high-temperature XRD average over 308-773 K): Verger et al., J. Appl. Phys. 124, 205108 (2018), abstract, DOI:10.1063/1.5054379",
                "thermal_cond_w_mk": "7.8 W/(m*K) (measured, polycrystal at 298 K): Bai et al., J. Am. Ceram. Soc. 103(10), 5837-5851 (2020), abstract, DOI:10.1111/jace.17205, https://api.crossref.org/works/10.1111/jace.17205",
                "band_gap_ev": "0.0 eV, metal, no gap (measured single-crystal metallic resistivity ~60 uOhm*cm at 300 K): Lamichhane et al., Phys. Rev. Materials 2, 084408 (2018), Sec. III.E p. 8 & Fig. 8, DOI:10.1103/PhysRevMaterials.2.084408, https://arxiv.org/pdf/1805.06373",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="MgSiN2 (Magnesium Silicon Nitride)",
            structure_class="Wurtzite-Derived Ternary Nitride",
            composition={"Mg": 1, "Si": 1, "N": 2},
            density_g_cm3=3.14,
            bulk_modulus_gpa=183.0,
            shear_modulus_gpa=112.0,
            thermal_expansion_ppm=4.4,
            thermal_cond_w_mk=23.0,
            band_gap_ev=5.60,
            source="Bruls, Ph.D. Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf; Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7; Bruls et al., J. Phys. Chem. Solids 61(8), 1285-1292 (2000), DOI:10.1016/S0022-3697(99)00416-3; de Boer et al., Phys. Status Solidi RRL 9(4), 250-254 (2015), DOI:10.1002/pssr.201510043",
            property_sources={
                "composition": "Stoichiometric orthorhombic Pna2_1 MgSiN2 powder/polycrystal: Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 3 Table 3-4 p. 70, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "density_g_cm3": "3.14 g/cm^3 (measured X-ray/neutron powder diffraction 3.141(1) g/cm^3 at 296 K in Table 3-4 p. 70, and 3.142e3 kg/m^3 at 293 K in Table 6-2 p. 144): Bruls, PhD Thesis, TU Eindhoven (2000), DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "bulk_modulus_gpa": "183.0 GPa (derived via B = E/(3*(1-2*nu)) = 183.1 GPa from measured polycrystal E = 279 +/- 4 GPa, nu = 0.246 at 293 K): Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7, openly readable in Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 6 Tables 6-1 & 6-2 pp. 140-144, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "shear_modulus_gpa": "112.0 GPa (derived via G = E/(2*(1+nu)) = 112.0 GPa from measured polycrystal E = 279 +/- 4 GPa, nu = 0.246 at 293 K): Bruls et al., J. Eur. Ceram. Soc. 21(3), 263-268 (2001), DOI:10.1016/S0955-2219(00)00210-7, openly readable in Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 6 Tables 6-1 & 6-2 pp. 140-144, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "thermal_expansion_ppm": "4.4 ppm/K (measured isotropic average on powder by neutron diffraction at 300 K): Bruls et al., J. Phys. Chem. Solids 61(8), 1285-1292 (2000), DOI:10.1016/S0022-3697(99)00416-3, openly readable in Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 4 pp. 110 & 112, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "thermal_cond_w_mk": "23 W/(m*K) (measured on hot-pressed polycrystal at 300 K): Bruls et al., J. Eur. Ceram. Soc. 18, 823-830 (1998), DOI:10.1016/S0955-2219(97)00205-7, as reported in Bruls, PhD Thesis, TU Eindhoven (2000), Ch. 8 pp. 191 & 195, DOI:10.6100/IR535906, https://pure.tue.nl/ws/portalfiles/portal/1306902/200013035.pdf",
                "band_gap_ev": "5.6 +/- 0.2 eV (measured on powder by soft X-ray spectroscopy at room temperature): UNSOURCED in paywalled de Boer et al., Phys. Status Solidi RRL 9(4), 250-254 (2015), DOI:10.1002/pssr.201510043",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="ZrW2O8 (Alpha-Zirconium Tungstate)",
            structure_class="Open Framework Oxide",
            composition={"Zr": 1, "W": 2, "O": 8},
            density_g_cm3=5.07,
            bulk_modulus_gpa=74.5,
            shear_modulus_gpa=33.8,
            thermal_expansion_ppm=-8.7,
            thermal_cond_w_mk=0.51,
            band_gap_ev=2.84,
            source="Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; Kennedy & White, Solid State Commun. 134(4), 271-276 (2005), DOI:10.1016/j.ssc.2005.01.031; Kennedy, PhD Thesis, Dalhousie Univ. (2005), http://hdl.handle.net/10222/54702; Ouyang, Xu & Ching, Phys. Rev. B 65, 113110 (2002), DOI:10.1103/PhysRevB.65.113110",
            property_sources={
                "composition": "Cubic P2_13 alpha-ZrW2O8 (O8W2Zr): Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90; Materials Project mp-18778, https://optimade.materialsproject.org/v1/structures/mp-18778",
                "density_g_cm3": "5.07 g/cm^3 (theoretical crystal density at 300 K): UNSOURCED exact 5.07 (Mary 1996 does not print 5.07); cf. theoretical crystal density 5.08 g/cm^3 read in Kennedy, PhD Thesis, Dalhousie Univ. (2005), Table 3.8 p. 138, http://hdl.handle.net/10222/54702, and Wu et al., NIFS-PROC-86 (2011), p. 98, https://www.nifs.ac.jp/report/NIFS-PROC-86.pdf",
                "bulk_modulus_gpa": "74.5 GPa (measured, single crystal at 300 K, resonant ultrasound spectroscopy paired with G): Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502, as cited in Wu et al., NIFS-PROC-86 (2011), p. 106, https://www.nifs.ac.jp/report/NIFS-PROC-86.pdf",
                "shear_modulus_gpa": "33.8 GPa (measured, single crystal at 300 K, resonant ultrasound spectroscopy paired with B = 74.5 GPa): UNSOURCED exact 33.8 in paywalled Drymiotis et al., Phys. Rev. Lett. 93, 025502 (2004), DOI:10.1103/PhysRevLett.93.025502; cf. 33.9 GPa cited in Wu et al., NIFS-PROC-86 (2011), p. 106, https://www.nifs.ac.jp/report/NIFS-PROC-86.pdf",
                "thermal_expansion_ppm": "-8.7 ppm/K (measured isotropic mean linear thermal expansion on powder by neutron diffraction over 0.3 to 693 K in the alpha phase; overall NTE persists from 0.3 to 1050 K): Mary et al., Science 272(5258), 90-92 (1996), DOI:10.1126/science.272.5258.90, as reported in Kennedy, PhD Thesis, Dalhousie Univ. (2005), Sec. 2.2 p. 94, http://hdl.handle.net/10222/54702",
                "thermal_cond_w_mk": "0.51 +/- 0.10 W/(m*K) (measured on ~75.4%-dense sintered polycrystal pellets at 300 K; porosity-corrected estimate for fully dense ZrW2O8 is 0.91 W/(m*K) at 300 K): Kennedy & White, Solid State Commun. 134(4), 271-276 (2005), DOI:10.1016/j.ssc.2005.01.031, openly readable in Kennedy, PhD Thesis, Dalhousie Univ. (2005), Table 3.8 p. 138, Table 3.9 p. 143 & p. 144, http://hdl.handle.net/10222/54702",
                "band_gap_ev": "2.84 eV (calculated, DFT OLCAO-LDA indirect band gap of cubic alpha-ZrW2O8 crystal): Ouyang, Xu & Ching, Phys. Rev. B 65, 113110 (2002), abstract, DOI:10.1103/PhysRevB.65.113110, https://api.crossref.org/works/10.1103/PhysRevB.65.113110",
            },
            placeholder=False,
        ),
        MaterialEntry(
            name="SiO2 (Alpha-Cristobalite)",
            structure_class="Tetrahedral Silica Polymorph",
            composition={"Si": 1, "O": 2},
            density_g_cm3=2.33,
            bulk_modulus_gpa=16.4,
            shear_modulus_gpa=39.1,
            thermal_expansion_ppm=24.0,
            thermal_cond_w_mk=6.15,
            band_gap_ev=8.57,
            source="Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650, as tabulated at http://www.numericana.com/answer/elastic.htm; Pluth, Smith & Faber, J. Appl. Phys. 57, 1045-1049 (1985), DOI:10.1063/1.334545; Kunugi et al., J. Am. Ceram. Soc. 55, 580 (1972), DOI:10.1111/j.1151-2916.1972.tb13442.x, as tabulated in Mizokami, Togo & Tanaka, Phys. Rev. B 97, 224306 (2018), DOI:10.1103/PhysRevB.97.224306; Milton et al., Materials 16, 1382 (2023), DOI:10.3390/ma16041382",
            property_sources={
                "composition": "Tetragonal P4_12_12 alpha-cristobalite SiO2: Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), abstract, DOI:10.1126/science.257.5070.650",
                "density_g_cm3": "2.33 g/cm^3 (measured single-crystal X-ray density at room temperature ~293 K, 2335 g/L): Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650, as tabulated in Michon, Numericana, http://www.numericana.com/answer/elastic.htm",
                "bulk_modulus_gpa": "16.4 GPa (measured single-crystal Brillouin scattering Voigt-Reuss-Hill isotropic aggregate at room temperature ~293 K, paired with G = 39.1 GPa): Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650, as tabulated in Michon, Numericana, http://www.numericana.com/answer/elastic.htm",
                "shear_modulus_gpa": "39.1 GPa (measured single-crystal Brillouin scattering Voigt-Reuss-Hill isotropic aggregate at room temperature ~293 K, paired with B = 16.4 GPa -> Poisson ratio -0.16): Yeganeh-Haeri, Weidner & Parise, Science 257(5070), 650-652 (1992), DOI:10.1126/science.257.5070.650, as tabulated in Michon, Numericana, http://www.numericana.com/answer/elastic.htm",
                "thermal_expansion_ppm": "24.0 ppm/K (measured powder neutron diffraction mean linear thermal expansion over 296-473 K, strongly anisotropic): UNSOURCED in paywalled Pluth, Smith & Faber, J. Appl. Phys. 57, 1045-1049 (1985), DOI:10.1063/1.334545",
                "thermal_cond_w_mk": "6.15 W/(m*K) (measured on powder at 300 K; reported values range from ~3 to 7 W/(m*K)): Kunugi et al., J. Am. Ceram. Soc. 55, 580 (1972), DOI:10.1111/j.1151-2916.1972.tb13442.x, as tabulated in Mizokami, Togo & Tanaka, Phys. Rev. B 97, 224306 (2018), Table I p. 2, DOI:10.1103/PhysRevB.97.224306, https://repository.kulib.kyoto-u.ac.jp/server/api/core/bitstreams/48a0e84e-adb4-4754-9aa2-7bd3603f257c/content",
                "band_gap_ev": "8.57 eV (calculated, hybrid DFT PBE0-TC-LRC on alpha-cristobalite crystal): Milton et al., Materials 16, 1382 (2023), Table 1 & Sec. 4.1, DOI:10.3390/ma16041382, https://pmc.ncbi.nlm.nih.gov/articles/PMC9960460/",
            },
            placeholder=False,
        ),
    ]


MAX_PLACEHOLDER_COUNT: int = 20


def generate_placeholder_candidates(
    count: int = 4,
    rng: Optional[random.Random] = None,
    start_index: int = 1,
) -> List[MaterialEntry]:
    """
    Generates synthetic, unverified placeholder rows using a deterministic
    `random.Random` instance for UI/pipeline testing.
    Every returned entry has `placeholder=True` and carries no literature citations.
    Raises ValueError if count < 0 or count > MAX_PLACEHOLDER_COUNT (20).
    """
    if count < 0 or count > MAX_PLACEHOLDER_COUNT:
        raise ValueError(
            f"Placeholder count must be between 0 and {MAX_PLACEHOLDER_COUNT}, got {count}."
        )
    if count == 0:
        return []
    if rng is None:
        rng = random.Random(0)

    cations = ["Fe", "Al", "Mg", "Si", "Ca", "Na", "Mn", "Zn"]
    anions = ["B", "C", "N", "O", "P", "S"]
    results: List[MaterialEntry] = []

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
            MaterialEntry(
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
        help="Number of synthetic placeholder rows to generate in a separate unranked table (0 to 20, default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic placeholder generation (default: 42).",
    )
    args = parser.parse_args(argv)
    if args.placeholders < 0 or args.placeholders > MAX_PLACEHOLDER_COUNT:
        parser.error(f"--placeholders must be between 0 and {MAX_PLACEHOLDER_COUNT}.")

    catalog = get_catalog()
    heuristic_order = [
        "Dielectric thermal conductor",
        "Elastic screen: Hv estimate >= 15, B/G >= 1.35, density <= 5",
        "Negative / low thermal expansion",
        "Auxetic (negative Poisson ratio)",
    ]

    passing_by_heuristic: Dict[str, List[MaterialEntry]] = {h: [] for h in heuristic_order}
    no_cutoff_rows: List[MaterialEntry] = []
    for entry in catalog:
        label, score, _, _ = entry.evaluate_heuristics()
        if score is not None and score > 0.0 and label in passing_by_heuristic:
            passing_by_heuristic[label].append(entry)
        else:
            no_cutoff_rows.append(entry)

    print("=" * 120)
    print("  LITERATURE-SOURCED MATERIALS CATALOG: DERIVED PROPERTIES & HEURISTIC CUT-OFFS")
    print("=" * 120)
    print(
        f"{'Material':<44} | {'Cost/kg':<9} | {'Poisson':<7} | {'Est. Hv':<7} | "
        f"{'Top Heuristic Classification':<36} | {'Score'}"
    )
    print("-" * 120)

    print("[Rows passing a documented heuristic cut-off (grouped by heuristic)]")
    for h_name in heuristic_order:
        group = sorted(
            passing_by_heuristic[h_name],
            key=lambda item: item.evaluate_heuristics()[1] or 0.0,
            reverse=True,
        )
        for c in group:
            label, score, areas, _ = c.evaluate_heuristics()
            hv_str = f"{c.vickers_hardness_gpa:>6.1f}G" if c.vickers_hardness_gpa is not None else "    n/a"
            score_str = f"{score:>5.1f}" if score is not None else "  n/a"
            print(
                f"{c.name[:44]:<44} | ${c.raw_cost_per_kg:>7.2f} | "
                f"{c.poisson_ratio:>+7.3f} | {hv_str:<7} | {label:<36} | {score_str}"
            )
            if areas:
                print(f"     +- Possible application areas: {', '.join(areas)}")
            print(f"     +- Source: {c.source}")

    print("-" * 120)
    print("[no cut-off met]")
    for c in no_cutoff_rows:
        label, score, _, _ = c.evaluate_heuristics()
        hv_str = f"{c.vickers_hardness_gpa:>6.1f}G" if c.vickers_hardness_gpa is not None else "    n/a"
        score_str = f"{score:>5.1f}" if score is not None else "  n/a"
        print(
            f"{c.name[:44]:<44} | ${c.raw_cost_per_kg:>7.2f} | "
            f"{c.poisson_ratio:>+7.3f} | {hv_str:<7} | {label:<36} | {score_str}"
        )
        print(f"     +- Source: {c.source}")
    print("=" * 120)

    if args.placeholders > 0:
        rng = random.Random(args.seed)
        placeholders = generate_placeholder_candidates(count=args.placeholders, rng=rng)
        print("\n" + "=" * 118)
        print("  SYNTHETIC PLACEHOLDER CANDIDATES (UNVERIFIED RANDOM NUMBERS - UNRANKED)")
        print("=" * 118)
        for p in placeholders:
            label, score, _, _ = p.evaluate_heuristics()
            score_str = f"{score:.1f}" if score is not None else "n/a"
            print(
                f"[PLACEHOLDER] {p.name:<38} | formula={p.formula:<12} | "
                f"cost=${p.raw_cost_per_kg:>6.2f}/kg | heuristic={label} ({score_str})"
            )
        print("=" * 118)


if __name__ == "__main__":
    main()
