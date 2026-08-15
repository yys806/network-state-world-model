import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pi_jwm.r5_module_confirmation import (  # noqa: E402
    build_confirmation_matrix,
    build_confirmation_model,
    build_confirmation_run_specs,
    write_confirmation_bundle,
)


class R5ModuleConfirmationTest(unittest.TestCase):
    def test_matrix_changes_one_module_from_graph_rssm_control(self) -> None:
        matrix = build_confirmation_matrix()

        self.assertEqual(tuple(matrix), ("B", "F", "G", "H", "J"))
        self.assertEqual(matrix["B"].components["dynamics"], "graph_rssm_v1")
        self.assertEqual(matrix["F"].components["coupling"], "no_cross_graph_coupling_v1")
        self.assertEqual(
            matrix["G"].components["coupling"],
            "relation_constrained_cross_attention_v1",
        )
        self.assertEqual(
            matrix["H"].components["graph_encoder"],
            "edge_conditioned_relation_mpnn_v1",
        )
        self.assertEqual(
            matrix["J"].components["dynamics"],
            "legacy_directed_dynamic_residual_v2_adapted_v1",
        )
        self.assertEqual(matrix["J"].role, "architecture_control")
        self.assertEqual(matrix["J"].configuration["hidden_dim"], 16)
        for candidate_id in ("F", "G", "H"):
            changed = {
                key
                for key, value in matrix[candidate_id].components.items()
                if value != matrix["B"].components[key]
            }
            self.assertEqual(len(changed), 1)

    def test_run_specs_reuse_existing_b_and_train_only_nine_new_runs(self) -> None:
        specs = build_confirmation_run_specs()

        self.assertEqual(len(specs), 15)
        self.assertEqual(sum(spec.reuse_existing for spec in specs), 3)
        self.assertEqual(sum(not spec.reuse_existing for spec in specs), 12)
        self.assertTrue(all(spec.reuse_existing for spec in specs if spec.combination_id == "B"))

    def test_new_confirmation_models_are_executable_graph_rssm_compositions(self) -> None:
        for combination_id in ("F", "G", "H", "J"):
            model = build_confirmation_model(combination_id)
            self.assertEqual(model.combination_id, combination_id)
            self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)
        self.assertEqual(
            build_confirmation_model("J").config.dynamics,
            "legacy_directed_dynamic_residual_v2_adapted_v1",
        )

    def test_writer_emits_auditable_matrix_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "confirmation"

            write_confirmation_bundle(output, existing_r5_manifest_sha256="abc")

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"matrix.json", "run_specs.json", "summary.json", "README.md", "manifest.json"},
            )
            import json

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reused_run_count"], 3)
            self.assertEqual(summary["new_gpu_run_count"], 12)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_entry_count"], 4)
            self.assertEqual(manifest["existing_r5_manifest_sha256"], "abc")


if __name__ == "__main__":
    unittest.main()
