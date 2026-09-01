from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class DiagnosticSetTest(unittest.TestCase):
    def test_generated_holdout_is_disjoint_balanced_and_catalog_valid(self) -> None:
        path = ROOT / "data" / "diagnostic_set_100.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        public = [
            json.loads(line)
            for line in (ROOT / "data" / "public_set.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        with (ROOT / "data" / "catalog.jsonl").open(encoding="utf-8") as handle:
            catalog_ids = {
                str(json.loads(line)["parent_asin"])
                for line in handle
                if line.strip()
            }
        targets = [str(row["ground_truth"]["parent_asin"]) for row in rows]
        public_targets = {
            str(row["ground_truth"]["parent_asin"]) for row in public
        }

        self.assertEqual(len(rows), 100)
        self.assertEqual(len(set(targets)), 100)
        self.assertFalse(set(targets) & public_targets)
        self.assertTrue(set(targets) <= catalog_ids)
        self.assertEqual(
            Counter(row["scenario_type"] for row in rows),
            Counter({"buying": 40, "browsing": 40, "intent_override": 15, "boundary": 5}),
        )
        self.assertEqual(
            Counter(row["category_bucket"] for row in rows),
            Counter({"clothing": 40, "shoes": 35, "jewelry": 25}),
        )
        self.assertEqual(
            Counter(
                row["diagnostic_design"]["evidence_ambiguity"] for row in rows
            ),
            Counter({"unique": 53, "narrow": 30, "ambiguous": 17}),
        )


if __name__ == "__main__":
    unittest.main()
