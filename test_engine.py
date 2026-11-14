#!/usr/bin/env python3
"""
Unit test suite for engine.py and app.py using Python's standard-library unittest.
"""

import io
import json
import random
import re
from typing import Tuple
import unittest
import urllib.error

import app
import engine
from tools import check_dois


class _MockSocket:
    """In-memory socket wrapper for testing BaseHTTPRequestHandler without TCP permissions."""

    def __init__(self, raw_request: bytes) -> None:
        self._rfile = io.BytesIO(raw_request)
        self._wfile = io.BytesIO()

    def makefile(self, mode: str, *args, **kwargs):
        if "r" in mode:
            return self._rfile
        return self._wfile

    def sendall(self, data: bytes) -> None:
        self._wfile.write(data)

    def close(self) -> None:
        pass

    @property
    def response_bytes(self) -> bytes:
        return self._wfile.getvalue()


def invoke_http(method: str, path: str) -> Tuple[int, str, str]:
    """Executes a single HTTP request against CatalogHandler in-memory."""
    raw_req = f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode("utf-8")
    sock = _MockSocket(raw_req)
    app.CatalogHandler(sock, ("127.0.0.1", 0), None)
    raw_resp = sock.response_bytes.decode("utf-8", errors="replace")
    header_block, _, body = raw_resp.partition("\r\n\r\n")
    status_line = header_block.splitlines()[0]
    status_code = int(status_line.split()[1])
    return status_code, header_block, body


class TestEngineAndApp(unittest.TestCase):
    def test_cost_per_kg_for_known_formula(self) -> None:
        """Verifies raw_cost_per_kg against hand-computed molar mass weighted average."""
        aln = engine.MaterialEntry(
            name="AlN Test",
            structure_class="Wurtzite",
            composition={"Al": 1.0, "N": 1.0},
            density_g_cm3=3.26,
            bulk_modulus_gpa=202.0,
            shear_modulus_gpa=131.0,
            thermal_expansion_ppm=4.5,
            thermal_cond_w_mk=285.0,
            band_gap_ev=6.2,
        )
        m_al = engine.ATOMIC_MASS["Al"]
        m_n = engine.ATOMIC_MASS["N"]
        expected = (
            m_al * engine.ELEMENT_COST_USD_PER_KG["Al"]
            + m_n * engine.ELEMENT_COST_USD_PER_KG["N"]
        ) / (m_al + m_n)
        self.assertAlmostEqual(aln.raw_cost_per_kg, expected, places=6)

    def test_ti_6al_4v_weight_fractions(self) -> None:
        """Verifies that Ti-6Al-4V atomic composition reproduces 90 / 6 / 4 wt%."""
        catalog = {c.name: c for c in engine.get_catalog()}
        ti64 = catalog["Ti-6Al-4V (Grade 5 Titanium Alloy Baseline)"]
        wf = ti64.weight_fractions()
        self.assertAlmostEqual(wf["Ti"], 0.90, delta=0.001)
        self.assertAlmostEqual(wf["Al"], 0.06, delta=0.001)
        self.assertAlmostEqual(wf["V"], 0.04, delta=0.001)
        self.assertAlmostEqual(sum(wf.values()), 1.0, places=6)

    def test_youngs_modulus_and_poisson_ratio_identities(self) -> None:
        """
        Verifies standard isotropic elasticity identities:
          B = E / (3 * (1 - 2*nu))
          G = E / (2 * (1 + nu))
        across all catalog entries.
        """
        for entry in engine.get_catalog():
            e_mod = entry.youngs_modulus_gpa
            nu = entry.poisson_ratio
            b_recovered = e_mod / (3.0 * (1.0 - 2.0 * nu))
            g_recovered = e_mod / (2.0 * (1.0 + nu))
            self.assertAlmostEqual(b_recovered, entry.bulk_modulus_gpa, places=5)
            self.assertAlmostEqual(g_recovered, entry.shear_modulus_gpa, places=5)

    def test_every_catalog_element_in_both_element_tables(self) -> None:
        """
        Ensures element tables have identical keys, cover all catalog entries,
        and every property_sources entry contains a DOI, URL, database ID, or UNSOURCED tag.
        """
        self.assertEqual(
            set(engine.ELEMENT_COST_USD_PER_KG.keys()),
            set(engine.ATOMIC_MASS.keys()),
        )
        self.assertEqual(
            set(engine.ELEMENT_COST_USD_PER_KG.keys()),
            set(engine.ELEMENT_COST_SOURCES.keys()),
        )
        citation_pattern = re.compile(
            r"(10\.\d{4,9}/\S+|https?://\S+|mp-\d+|UNSOURCED)"
        )
        for el, cost_src in engine.ELEMENT_COST_SOURCES.items():
            self.assertRegex(
                cost_src,
                citation_pattern,
                f"Element {el} cost source lacks DOI, URL, database id, or UNSOURCED tag",
            )
        for entry in engine.get_catalog():
            self.assertFalse(entry.placeholder)
            self.assertTrue(entry.source)
            self.assertRegex(entry.source, citation_pattern)
            for prop in (
                "composition",
                "density_g_cm3",
                "bulk_modulus_gpa",
                "shear_modulus_gpa",
                "thermal_expansion_ppm",
                "thermal_cond_w_mk",
                "band_gap_ev",
            ):
                self.assertIn(prop, entry.property_sources)
                src_text = entry.property_sources[prop]
                self.assertTrue(src_text)
                self.assertRegex(
                    src_text,
                    citation_pattern,
                    f"{entry.name}.{prop} source lacks DOI, URL, database id, or UNSOURCED tag: {src_text}",
                )
            for el in entry.composition:
                self.assertIn(el, engine.ELEMENT_COST_USD_PER_KG)
                self.assertIn(el, engine.ATOMIC_MASS)

    def test_seeded_generator_reproducible_and_flagged_as_placeholder(self) -> None:
        """Verifies deterministic output, placeholder=True, formula consistency, and empty applications."""
        batch_a = engine.generate_placeholder_candidates(count=12, rng=random.Random(12345))
        batch_b = engine.generate_placeholder_candidates(count=12, rng=random.Random(12345))
        self.assertEqual(len(batch_a), 12)
        self.assertEqual([c.name for c in batch_a], [c.name for c in batch_b])
        self.assertEqual(
            [c.raw_cost_per_kg for c in batch_a],
            [c.raw_cost_per_kg for c in batch_b],
        )
        for c in batch_a:
            self.assertTrue(c.placeholder)
            self.assertIn(c.formula, c.name)
            label, score, areas, _ = c.evaluate_heuristics()
            self.assertEqual(label, "not evaluated (placeholder)")
            self.assertIsNone(score)
            self.assertEqual(areas, [])

        for bad_count in (-1, 21):
            with self.assertRaises(ValueError):
                engine.generate_placeholder_candidates(count=bad_count)
            with self.assertRaises(SystemExit):
                engine.main(["--placeholders", str(bad_count)])

        # Verify CLI output is pure ASCII (does not crash on ASCII-encoded stdout)
        buf = io.StringIO()
        import contextlib

        with contextlib.redirect_stdout(buf):
            engine.main(["--placeholders", "2", "--seed", "42"])
        cli_out = buf.getvalue()
        cli_out.encode("ascii")

    def test_no_baseline_is_labelled_an_anomaly(self) -> None:
        """Verifies that Ti-6Al-4V and B4C baselines return 'no anomaly' and score 0.0."""
        catalog = {c.name: c for c in engine.get_catalog()}
        for baseline_name in (
            "Ti-6Al-4V (Grade 5 Titanium Alloy Baseline)",
            "B4C (Boron Carbide Ceramic Baseline)",
        ):
            entry = catalog[baseline_name]
            label, score, areas, _ = entry.evaluate_heuristics()
            self.assertEqual(label, "no anomaly")
            self.assertEqual(score, 0.0)
            self.assertEqual(areas, [])

        # Also verify true auxetic definition (negative Poisson ratio only)
        cristobalite = catalog["SiO2 (Alpha-Cristobalite)"]
        self.assertLess(cristobalite.poisson_ratio, 0.0)
        self.assertGreaterEqual(cristobalite.heuristic_auxetic, 50.0)

    def test_cristobalite_chen_hardness_is_none(self) -> None:
        """Verifies that vickers_hardness_gpa returns None when G/B > 1 or nu < 0 (cristobalite)."""
        catalog = {c.name: c for c in engine.get_catalog()}
        cristobalite = catalog["SiO2 (Alpha-Cristobalite)"]
        self.assertGreater(cristobalite.shear_modulus_gpa, cristobalite.bulk_modulus_gpa)
        self.assertLess(cristobalite.poisson_ratio, 0.0)
        self.assertIsNone(cristobalite.vickers_hardness_gpa)
        self.assertEqual(cristobalite.heuristic_elastic_screen, 0.0)

        status, _, body = invoke_http("GET", "/api/compounds")
        self.assertEqual(status, 200)
        items = {row["name"]: row for row in json.loads(body)}
        self.assertIsNone(items["SiO2 (Alpha-Cristobalite)"]["hardness_gpa"])

    def test_post_init_input_validation(self) -> None:
        """Verifies that MaterialEntry.__post_init__ rejects non-positive moduli/density, empty composition, and unknown elements."""
        base_kwargs = dict(
            name="Valid",
            structure_class="Ceramic",
            composition={"Al": 1.0, "N": 1.0},
            density_g_cm3=3.26,
            bulk_modulus_gpa=211.0,
            shear_modulus_gpa=133.0,
            thermal_expansion_ppm=4.5,
            thermal_cond_w_mk=285.0,
            band_gap_ev=6.2,
        )
        for bad_override in (
            {"bulk_modulus_gpa": 0.0},
            {"bulk_modulus_gpa": -50.0},
            {"shear_modulus_gpa": 0.0},
            {"shear_modulus_gpa": -10.0},
            {"density_g_cm3": 0.0},
            {"density_g_cm3": -1.0},
            {"composition": {}},
            {"composition": {"Xe": 1.0}},
            {"composition": {"Al": 0.0, "N": 1.0}},
        ):
            kwargs = {**base_kwargs, **bad_override}
            with self.assertRaises(ValueError):
                engine.MaterialEntry(**kwargs)

    def test_no_max_mab_or_alloy_gets_hardness_label(self) -> None:
        """Verifies that MAX phases, MAB phases, and metallic alloys return 0.0 on heuristic 2."""
        for entry in engine.get_catalog():
            cls_lower = entry.structure_class.lower()
            if any(token in cls_lower for token in ("max", "mab", "alloy", "metal")):
                self.assertEqual(entry.heuristic_elastic_screen, 0.0)
                label, score, areas, _ = entry.evaluate_heuristics()
                self.assertNotIn("Hv estimate", label)
                self.assertEqual(score, 0.0)
                self.assertEqual(areas, [])

        # Even if a synthetic MAX/MAB entry has B/G >= 1.35, Hv >= 15, and density <= 5,
        # heuristic 2 must return 0.0.
        synthetic_max = engine.MaterialEntry(
            name="Synthetic MAX Test",
            structure_class="MAX-Phase Nanolaminate",
            composition={"Ti": 3.0, "Si": 1.0, "C": 2.0},
            density_g_cm3=4.2,
            bulk_modulus_gpa=220.0,
            shear_modulus_gpa=150.0,
            thermal_expansion_ppm=8.0,
            thermal_cond_w_mk=25.0,
            band_gap_ev=0.0,
        )
        self.assertEqual(synthetic_max.heuristic_elastic_screen, 0.0)

    def test_http_endpoints_cap_and_404s(self) -> None:
        """Tests GET /, GET /api/compounds, POST cap at MAX_PLACEHOLDER_ROWS, and 404s."""
        app.reset_state()
        try:
            # 1. GET / -> 200 HTML
            status, _, html_body = invoke_http("GET", "/")
            self.assertEqual(status, 200)
            self.assertIn("Materials Catalog", html_body)

            # 2. GET /api/compounds -> 200 JSON with 9 sourced rows
            status, _, body = invoke_http("GET", "/api/compounds")
            self.assertEqual(status, 200)
            items = json.loads(body)
            base_count = len(engine.get_catalog())
            self.assertEqual(len(items), base_count)
            self.assertTrue(all(not item["placeholder"] for item in items))

            # 3. Unknown GET -> 404
            status, _, _ = invoke_http("GET", "/nonexistent")
            self.assertEqual(status, 404)

            # 4. Unknown POST (including legacy /api/evolve) -> 404
            status, _, _ = invoke_http("POST", "/api/unknown")
            self.assertEqual(status, 404)
            status, _, _ = invoke_http("POST", "/api/evolve")
            self.assertEqual(status, 404)

            # 5. POST /api/placeholders capped at MAX_PLACEHOLDER_ROWS (20)
            data = []
            for _ in range(8):
                status, _, body = invoke_http("POST", "/api/placeholders")
                self.assertEqual(status, 200)
                data = json.loads(body)

            placeholders = [row for row in data if row["placeholder"]]
            sourced = [row for row in data if not row["placeholder"]]
            self.assertEqual(len(sourced), base_count)
            self.assertEqual(len(placeholders), app.MAX_PLACEHOLDER_ROWS)
            self.assertTrue(all(row["possible_applications"] == [] for row in placeholders))
            self.assertTrue(
                all(
                    row["heuristic_label"] == "not evaluated (placeholder)"
                    and row["heuristic_score"] is None
                    for row in placeholders
                )
            )
        finally:
            app.reset_state()

    def test_catalog_dois_syntax_and_resolution(self) -> None:
        """
        Verifies that catalog citations contain valid DOIs and that each unique DOI
        resolves in Crossref or DataCite (skipped gracefully when offline).
        """
        records = check_dois.extract_catalog_dois()
        self.assertGreaterEqual(len(records), 20)
        unique_dois = sorted({doi for _, _, doi in records})
        for doi in unique_dois:
            self.assertRegex(doi, r"^10\.\d{4,9}/\S+$")

        for doi in unique_dois:
            try:
                meta = check_dois.resolve_doi_metadata(doi, timeout=10.0)
                self.assertIn(meta["provider"], ("Crossref", "DataCite"))
                self.assertTrue(meta["title"])
            except urllib.error.URLError as exc:
                self.skipTest(f"Skipping online DOI resolution check while offline: {exc}")


if __name__ == "__main__":
    unittest.main()
