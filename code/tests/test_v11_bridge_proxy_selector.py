import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def write_bridge_json(path: Path, active_rmse: float, f1: float, link_rmse: float) -> None:
    payload = {
        "policy_checkpoint": f"checkpoints/{path.stem}.pt",
        "policy_threshold": 0.98,
        "test": {
            "active_rate": {"active_rmse": active_rmse, "active_mae": active_rmse / 2.0, "active_count": 10},
            "activity": {"f1": f1, "precision": f1, "recall": f1, "tp": 1.0, "fp": 2.0, "fn": 3.0},
            "link_rate": {"rmse": link_rmse, "mae": link_rmse / 2.0},
            "positive_rate_active": {"active_rmse": active_rmse + 1.0},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class V11BridgeProxySelectorTest(unittest.TestCase):
    def test_select_bridge_proxy_candidates_filters_and_sorts(self):
        from select_v11_bridge_proxy_candidates import select_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "epoch_007.json"
            lower_rmse_bad_f1 = root / "epoch_002.json"
            bad_link = root / "epoch_010.json"
            best = root / "epoch_009.json"
            write_bridge_json(good, active_rmse=245.0, f1=0.24, link_rmse=34.0)
            write_bridge_json(lower_rmse_bad_f1, active_rmse=238.0, f1=0.04, link_rmse=79.0)
            write_bridge_json(bad_link, active_rmse=240.0, f1=0.26, link_rmse=45.0)
            write_bridge_json(best, active_rmse=243.0, f1=0.25, link_rmse=33.0)

            selected = select_candidates(
                [lower_rmse_bad_f1, good, bad_link, best],
                split="test",
                min_f1=0.23,
                max_link_rmse=40.0,
                top_k=2,
            )

            self.assertEqual([item["path"] for item in selected], [str(best), str(good)])
            self.assertEqual(selected[0]["active_rate_rmse"], 243.0)
            self.assertEqual(selected[0]["activity_f1"], 0.25)

    def test_select_bridge_proxy_candidates_can_fall_back_when_no_candidate_passes_gate(self):
        from select_v11_bridge_proxy_candidates import select_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "epoch_001.json"
            second = root / "epoch_002.json"
            write_bridge_json(first, active_rmse=300.0, f1=0.01, link_rmse=90.0)
            write_bridge_json(second, active_rmse=250.0, f1=0.02, link_rmse=80.0)

            selected = select_candidates(
                [first, second],
                split="test",
                min_f1=0.23,
                max_link_rmse=40.0,
                top_k=1,
                allow_fallback=True,
            )

            self.assertEqual(selected[0]["path"], str(second))
            self.assertFalse(selected[0]["passes_gate"])


if __name__ == "__main__":
    unittest.main()
