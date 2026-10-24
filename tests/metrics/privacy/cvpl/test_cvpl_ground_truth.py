"""Ground-truth handling: complete, partial, invalid, absent."""

import pandas as pd
import pytest

from .helpers import GT, anchors, measure, release_frame, source_frame


def test_duplicate_source_identifiers_make_the_ground_truth_invalid_not_zero():
    src = source_frame([("A", 1.0, "s0"), ("A", 1.1, "s0")])
    rel = release_frame([("A", 1.0, "s0")] + anchors())
    r = measure(src, rel)
    assert r["ground_truth"]["status"] == "invalid"
    assert r["cvpl_lr"] == 1.0                      # exposure is still measured
    st = r["cvpl_id_uniform_gt_status"]
    assert r["cvpl_id_uniform_gt"] is None
    assert st["status"] == "not_measured" and st["reason"] == "source_subject_identifiers_not_unique"


def test_released_rows_without_identifier_make_complete_scope_partial():
    src = source_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")])
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.1, None)] + anchors())
    r = measure(src, rel)
    assert r["ground_truth"]["status"] == "partial"
    assert r["ground_truth"]["reason"] == "released_rows_without_identifier"
    assert r["cvpl_id_uniform_gt"] is None
    # an anonymous released row might belong to anyone: nothing is known about any T_i
    st = r["cvpl_id_uniform_gt_status"]
    assert st["known_subset_denominator"] == 0 and st["known_subset_value"] is None
    assert st["known_subset_reason"] == "release_membership_unknown"
    assert r["gt_coverage"] == 0.0


def test_declared_partial_scope_never_reports_a_population_measure():
    src = source_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")])
    rel = release_frame([("A", 1.0, "s0"), ("A", 1.1, "s1")] + anchors())
    r = measure(src, rel, ground_truth={**GT, "scope": "partial"})
    assert r["ground_truth"]["status"] == "partial"
    assert r["cvpl_id_uniform_gt"] is None
    assert r["cvpl_top1_success_gt_status"]["known_subset_value"] is not None
    assert r["precision_at_tau_status"]["status"] == "not_measured"


def test_a_ground_truth_without_any_source_identifier_is_absent():
    src = source_frame([("A", 1.0, None), ("A", 1.1, None)])
    rel = release_frame([("A", 1.0, "s0")] + anchors())
    r = measure(src, rel)
    assert r["ground_truth"]["status"] == "absent"
    assert r["cvpl_id_uniform_gt_status"]["status"] == "not_measured"


def test_no_declared_link_means_precision_is_undefined_not_zero():
    src = source_frame([("A", -500.0, "s0")])
    rel = release_frame([("A", 1.0, "t0")] + anchors())
    r = measure(src, rel)
    # the only candidate is on the other side of the joint mean: no surviving candidate
    assert r["cvpl_lr"] == 0.0
    assert r["cvpl_id_uniform_gt"] == 0.0
    assert r["precision_at_tau"] is None
    assert r["precision_at_tau_status"]["reason"] == "no_declared_links"


def test_ground_truth_ids_may_be_numbers():
    src = pd.DataFrame({"g": ["A", "A"], "x": [1.0, 1.1], "sid": [1, 2]})
    rel = pd.DataFrame({"g": ["A", "A", "Z", "Z", "Z", "Z"], "x": [1.0, 1.1, -50, -51, -52, -53],
                        "rid": [1, 2, 11, 12, 13, 14]})
    r = measure(src, rel)
    assert r["cvpl_id_uniform_gt"] == pytest.approx(0.5)
