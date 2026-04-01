from __future__ import annotations

import unittest
from pathlib import Path

from src.modeling import run_pipeline


class RecommendationModelTensorFlowTestCase(unittest.TestCase):
    def test_pipeline_contract(self) -> None:
        project_dir = Path(__file__).resolve().parents[1]
        summary = run_pipeline(project_dir)
        self.assertEqual(summary["interaction_count"], 18)
        self.assertEqual(summary["user_count"], 6)
        self.assertEqual(summary["item_count"], 6)
        self.assertGreater(summary["train_interaction_count"], 10)
        self.assertGreater(summary["test_interaction_count"], 3)
        self.assertLess(summary["rmse"], 2.5)


if __name__ == "__main__":
    unittest.main()
