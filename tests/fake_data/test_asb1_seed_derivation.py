"""ASB-1: the fake-data seed uses the WHOLE key (HMAC-SHA256 -> 64 bit); no key means a random
per-run key, never a fixed public function of the original value."""

import hashlib
import json
import logging

import pandas as pd
import pytest

from pamola_core.fake_data import FakeEmailOperation, FakeNameOperation
from pamola_core.fake_data.commons import prgn
from pamola_core.fake_data.commons.prgn import PRNGenerator, derive_seed
from pamola_core.utils.ops.op_data_source import DataSource

NAMES = ["Alice Smith", "Bob Jones", "Carol White", "Dave Brown", "Erin Black", "Frank Green"]


class _Reporter:
    def add_operation(self, *a, **k):
        pass

    def add_artifact(self, *a, **k):
        pass


def _replace_names(op):
    return [op.process_value(v, language="en", field_name="name", context_salt="name-generation",
                             gender=None, record_id=None) for v in NAMES]


def _replace_emails(op):
    return [op.process_value(f"{n.split()[0].lower()}@example.com", field_name="email",
                             context_salt="email-generation", record_id=None) for n in NAMES]


def test_asb1_the_whole_key_takes_part_not_its_last_four_bytes():
    # Codex: "aaaa-2023" and "zzzz-2023" gave identical e-mail output (only the last 4 bytes counted)
    assert PRNGenerator("aaaa-2023").global_seed != PRNGenerator("zzzz-2023").global_seed
    a = FakeEmailOperation(field_name="email", key="aaaa-2023")
    z = FakeEmailOperation(field_name="email", key="zzzz-2023")
    assert _replace_emails(a) != _replace_emails(z)
    assert _replace_emails(a) == _replace_emails(FakeEmailOperation(field_name="email", key="aaaa-2023"))


def test_asb1_the_two_name_keys_that_collided_in_344_tries_no_longer_collide():
    # Codex: key-225 and key-344 had the same last 4 hex digits of the derived hash -> same output
    a = FakeNameOperation(field_name="name", key="key-225")
    b = FakeNameOperation(field_name="name", key="key-344")
    assert _replace_names(a) != _replace_names(b)


def test_asb1_seed_space_is_64_bit_and_collision_free_for_distinct_keys():
    seeds = {derive_seed(f"key-{i}".encode()) for i in range(20000)}
    assert len(seeds) == 20000  # no collision among 20k keys (2**32 would collide by then)
    assert max(seeds) > 2 ** 40  # not a 32-bit space
    assert PRNGenerator("x").global_seed == PRNGenerator(b"x").global_seed
    # the seed is HMAC-SHA256(key, domain || context), first 8 bytes
    import hmac

    expect = int.from_bytes(hmac.new(b"k", b"pamola.fake_data.seed.v1\x00ctx", hashlib.sha256).digest()[:8], "big")
    assert derive_seed(b"k", "ctx") == expect
    assert prgn.generate_seed_from_key("k", "ctx") == expect


def test_asb1_hmac_message_is_unambiguous():
    g = PRNGenerator("k")
    assert g.generate_with_seed("c", salt="ab") != g.generate_with_seed("bc", salt="a")


def test_asb1_without_a_key_the_replacement_is_not_a_fixed_public_function():
    runs = [_replace_names(FakeNameOperation(field_name="name")) for _ in range(3)]
    assert not (runs[0] == runs[1] == runs[2])
    emails = [_replace_emails(FakeEmailOperation(field_name="email")) for _ in range(3)]
    assert not (emails[0] == emails[1] == emails[2])
    assert FakeNameOperation(field_name="name").keyed is False
    assert FakeNameOperation(field_name="name", key="k").keyed is True


def test_asb1_keyless_warning_is_logged_once_and_never_contains_a_key(monkeypatch, caplog):
    monkeypatch.setattr(prgn, "_keyless_logged", False)
    with caplog.at_level(logging.DEBUG):
        FakeNameOperation(field_name="name")
        FakeNameOperation(field_name="name")
        FakeEmailOperation(field_name="email")
    msgs = [r.getMessage() for r in caplog.records if "no key was given" in r.getMessage()]
    assert len(msgs) == 1 and "explicit key" in msgs[0]


@pytest.mark.parametrize("key,keyed", [(None, False), ("project-key", True)])
def test_asb1_metrics_record_keyed_and_never_the_key(tmp_path, key, keyed):
    df = pd.DataFrame({"name": NAMES})
    op = FakeNameOperation(field_name="name", key=key, generate_visualization=False)
    out = op.execute(DataSource(dataframes={"main": df}), tmp_path / "t", _Reporter())
    assert str(out.status).endswith("SUCCESS"), out.error_message
    metrics_files = [p for p in (tmp_path / "t").rglob("*_metrics_*.json")]
    assert metrics_files
    data = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    assert data["metrics"]["keyed"] is keyed
    for p in (tmp_path / "t").rglob("*"):
        if p.is_file():
            text = p.read_bytes().decode("utf-8", errors="ignore")
            assert "project-key" not in text
    assert out.metrics.get("keyed") is keyed


def test_asb1_effective_key_is_part_of_the_cache_identity():
    a = FakeNameOperation(field_name="name", key="one")
    b = FakeNameOperation(field_name="name", key="two")
    assert a._get_base_parameters()["key_digest"] != b._get_base_parameters()["key_digest"]
