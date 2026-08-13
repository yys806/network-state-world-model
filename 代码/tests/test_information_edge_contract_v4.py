from __future__ import annotations

import unittest

import numpy as np

from pi_jwm.information_edge_contract_v4 import (
    CONTRACT_VERSION,
    MissingReason,
    build_field_registry,
    build_legacy_slot_mapping,
    assignment_coo_to_dense,
    validate_assignment_coo,
    validate_edge_applicability,
    validate_field_values,
    validate_link_outcome,
    validate_masked_field,
    validate_prev_field_timing,
    validate_rb_outcome,
)


class InformationEdgeContractV4RegistryTests(unittest.TestCase):
    def test_registry_has_five_core_fields_and_separate_namespaces(self):
        rows = build_field_registry()
        by_name = {row["name"]: row for row in rows}

        self.assertEqual("PIJWM-DG-Contract-v4", CONTRACT_VERSION)
        self.assertEqual(len(rows), len(by_name))
        self.assertEqual(5, sum(row["tier"] == "E1" for row in rows))
        self.assertEqual("pre_link", by_name["pre_link.prev_served_data"]["namespace"])
        self.assertEqual(
            "AirFogSim data-unit",
            by_name["pre_link.prev_served_data"]["unit"],
        )
        self.assertEqual(
            ["wireless"],
            by_name["pre_link.channel_attenuation_mean_db"][
                "applicable_edge_classes"
            ],
        )
        self.assertIn("action.assignment_coo", by_name)
        self.assertIn("outcome_rb_optional.outage", by_name)
        self.assertEqual(
            "fixed_config", by_name["config.noise_power_dbm"]["provenance_level"]
        )
        self.assertEqual("unavailable", by_name["excluded.mcs"]["provenance_level"])
        self.assertEqual(
            {"direct", "derived", "fixed_config", "unavailable"},
            {row["provenance_level"] for row in rows},
        )
        self.assertEqual(0.0, by_name["outcome_link.served_data"]["valid_min"])
        self.assertIn(
            "sum",
            by_name["outcome_link.effective_rate_per_s"]["dependency_formula"],
        )

    def test_legacy_mapping_has_exactly_eighteen_unique_slots(self):
        rows = build_legacy_slot_mapping()
        self.assertEqual(list(range(18)), [row["legacy_index"] for row in rows])
        self.assertEqual(18, len({row["legacy_slot"] for row in rows}))
        mapping = {row["legacy_slot"]: row for row in rows}

        self.assertEqual(
            "delete_continuous_feature",
            mapping["pre.interface_available"]["decision"],
        )
        self.assertEqual("unavailable", mapping["action.mcs"]["decision"])
        self.assertEqual(
            "regenerate_from_assigned_rb",
            mapping["outcome.rate_sum"]["decision"],
        )
        self.assertEqual(
            "legacy_semantics_conflicted",
            mapping["outcome.rate_sum"]["source_status"],
        )


class InformationEdgeContractV4MaskTests(unittest.TestCase):
    def test_mask_and_missing_reason_dtypes_are_not_silently_coerced(self):
        with self.assertRaisesRegex(ValueError, "valid_mask must have bool dtype"):
            validate_masked_field(
                values=np.asarray([0.0], np.float32),
                valid_mask=np.asarray([1], np.uint8),
                missing_reason=np.asarray([MissingReason.NONE.value], np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "missing_reason must have integer dtype"):
            validate_masked_field(
                values=np.asarray([0.0], np.float32),
                valid_mask=np.asarray([True]),
                missing_reason=np.asarray([0.5], np.float32),
            )

    def test_valid_zero_is_not_missing(self):
        validate_masked_field(
            values=np.asarray([0.0], np.float32),
            valid_mask=np.asarray([True]),
            missing_reason=np.asarray([MissingReason.NONE.value], np.uint8),
        )

    def test_invalid_value_requires_zero_fill(self):
        with self.assertRaisesRegex(ValueError, "invalid element must use zero fill"):
            validate_masked_field(
                values=np.asarray([3.0], np.float32),
                valid_mask=np.asarray([False]),
                missing_reason=np.asarray(
                    [MissingReason.NOT_COLLECTED.value], np.uint8
                ),
            )

    def test_valid_and_invalid_reasons_are_not_interchangeable(self):
        with self.assertRaisesRegex(ValueError, "valid element must use"):
            validate_masked_field(
                values=np.asarray([0.0], np.float32),
                valid_mask=np.asarray([True]),
                missing_reason=np.asarray(
                    [MissingReason.NOT_COLLECTED.value], np.uint8
                ),
            )
        with self.assertRaisesRegex(ValueError, "invalid element requires"):
            validate_masked_field(
                values=np.asarray([0.0], np.float32),
                valid_mask=np.asarray([False]),
                missing_reason=np.asarray([MissingReason.NONE.value], np.uint8),
            )

    def test_first_frame_prev_field_uses_no_history(self):
        validate_masked_field(
            values=np.asarray([0.0], np.float32),
            valid_mask=np.asarray([False]),
            missing_reason=np.asarray([MissingReason.NO_HISTORY.value], np.uint8),
        )

    def test_previous_outcome_timing_requires_no_history_on_first_frame(self):
        validate_prev_field_timing(
            valid_mask=np.asarray([[False], [True]]),
            missing_reason=np.asarray(
                [[MissingReason.NO_HISTORY.value], [MissingReason.NONE.value]],
                np.uint8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "first frame"):
            validate_prev_field_timing(
                valid_mask=np.asarray([[True], [True]]),
                missing_reason=np.asarray(
                    [[MissingReason.NONE.value], [MissingReason.NONE.value]],
                    np.uint8,
                ),
            )

    def test_nonnegative_field_range_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "below valid_min"):
            validate_field_values(
                field_name="outcome_link.served_data",
                values=np.asarray([-1.0], np.float32),
                valid_mask=np.asarray([True]),
                missing_reason=np.asarray([MissingReason.NONE.value], np.uint8),
            )

    def test_link_outcome_conservation_is_enforced(self):
        validate_link_outcome(
            effective_rate_per_s=np.asarray([5.0]),
            served_data=np.asarray([1.0]),
            slot_seconds=0.2,
            remaining_before=np.asarray([2.0]),
            assigned_rate_by_rb=np.asarray([[2.0, 3.0]]),
        )
        with self.assertRaisesRegex(ValueError, "assigned-RB sum"):
            validate_link_outcome(
                effective_rate_per_s=np.asarray([6.0]),
                served_data=np.asarray([1.0]),
                slot_seconds=0.2,
                remaining_before=np.asarray([2.0]),
                assigned_rate_by_rb=np.asarray([[2.0, 3.0]]),
            )
        with self.assertRaisesRegex(ValueError, "rate times slot"):
            validate_link_outcome(
                effective_rate_per_s=np.asarray([5.0]),
                served_data=np.asarray([1.1]),
                slot_seconds=0.2,
                remaining_before=np.asarray([2.0]),
                assigned_rate_by_rb=np.asarray([[2.0, 3.0]]),
            )

    def test_rb_outage_and_noise_floor_are_enforced(self):
        validate_rb_outcome(
            rate_per_s=np.asarray([0.0, 2.0]),
            outage=np.asarray([True, False]),
            interference_plus_noise_mw=np.asarray([1.0, 1.5]),
            noise_power_mw=1.0,
        )
        with self.assertRaisesRegex(ValueError, "outage RB must have zero rate"):
            validate_rb_outcome(
                rate_per_s=np.asarray([1.0]),
                outage=np.asarray([True]),
                interference_plus_noise_mw=np.asarray([1.0]),
                noise_power_mw=1.0,
            )
        with self.assertRaisesRegex(ValueError, "below noise power"):
            validate_rb_outcome(
                rate_per_s=np.asarray([0.0]),
                outage=np.asarray([False]),
                interference_plus_noise_mw=np.asarray([0.9]),
                noise_power_mw=1.0,
            )

    def test_wireless_only_field_on_wired_edge_is_not_applicable(self):
        validate_edge_applicability(
            field_name="pre_link.channel_attenuation_mean_db",
            edge_class="wired",
            valid_mask=np.asarray([False]),
            missing_reason=np.asarray(
                [MissingReason.NOT_APPLICABLE.value], np.uint8
            ),
        )

    def test_applicable_missing_field_cannot_hide_as_not_applicable(self):
        with self.assertRaisesRegex(ValueError, "cannot hide missing data"):
            validate_edge_applicability(
                field_name="pre_link.channel_attenuation_mean_db",
                edge_class="wireless",
                valid_mask=np.asarray([False]),
                missing_reason=np.asarray(
                    [MissingReason.NOT_APPLICABLE.value], np.uint8
                ),
            )


class InformationEdgeContractV4ActionTests(unittest.TestCase):
    def test_empty_assignment_is_a_valid_all_zero_action(self):
        validate_assignment_coo(
            np.empty((0, 4), dtype=np.int64),
            capacities=(2, 3, 4, 5),
        )

    def test_duplicate_assignment_is_rejected(self):
        coo = np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]], np.int64)
        with self.assertRaisesRegex(ValueError, "duplicate assignment"):
            validate_assignment_coo(coo, capacities=(2, 3, 4, 5))

    def test_out_of_range_resource_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "resource index out of range"):
            validate_assignment_coo(
                np.asarray([[0, 0, 0, 5]], np.int64),
                capacities=(1, 1, 1, 5),
            )

    def test_assignment_must_be_integer_record_by_four(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_assignment_coo(
                np.asarray([[0.0, 0.0, 0.0, 0.0]], np.float32),
                capacities=(1, 1, 1, 1),
            )

    def test_assignment_coo_round_trips_to_dense_binary_tensor(self):
        coo = np.asarray([[0, 1, 2, 3], [1, 0, 1, 2]], np.int64)
        dense = assignment_coo_to_dense(coo, capacities=(2, 2, 3, 4))
        self.assertEqual((2, 2, 3, 4), dense.shape)
        self.assertEqual(2, int(dense.sum()))
        self.assertTrue(dense[0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
