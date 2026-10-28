import io
import json
from types import SimpleNamespace

import pytest

from scripts.so4gp_compat_probe import classify_output, main, run_probe_suite


class FakeGRAANK:
    def __init__(self, data_source, min_sup=0.5, eq=False):
        self.data_source = data_source
        self.min_sup = min_sup
        self.eq = eq

    def discover(self):
        return json.dumps(
            {
                "Algorithm": "GRAANK",
                "Patterns": [["precipitation+, ndvi+", "sup=0.8", "density=0.8"]],
            }
        )


class FakeTGrad:
    def __init__(self, data_source, min_sup=0.5, target_col=2, min_rep=0.5, eq=False):
        self.data_source = data_source
        self.target_col = target_col
        self.min_rep = min_rep

    def discover_tgp(self, parallel=False):
        assert parallel is False
        return json.dumps(
            {
                "Algorithm": "TGrad",
                "Patterns": [["ndvi+, (precipitation+) +7.0 days", "sup=0.7"]],
            }
        )


class FakeAntGRAANK:
    def __init__(self, data_source, min_sup=0.5, eq=False):
        self.data_source = data_source

    def discover(self):
        return json.dumps(
            {
                "Algorithm": "ACO-GRAANK",
                "Best Patterns": [[["precipitation+", "ndvi+"], 0.7]],
                "Iterations": 3,
            }
        )


class FakeGeneticGRAANK(FakeAntGRAANK):
    pass


class FakeParticleGRAANK(FakeAntGRAANK):
    pass


class FakeGRAANKBestPatterns(FakeGRAANK):
    def discover(self):
        return json.dumps(
            {
                "Algorithm": "GRAANK",
                "Best Patterns": [[["precipitation+", "ndvi+"], 0.8]],
            }
        )


class FakeFlakyAntGRAANK(FakeAntGRAANK):
    calls = 0

    def discover(self):
        type(self).calls += 1
        if type(self).calls == 1:
            return super().discover()
        raise RuntimeError("intermittent optimizer failure")


def test_classify_output_handles_graank_patterns_json():
    summary = classify_output(
        json.dumps(
            {
                "Algorithm": "GRAANK",
                "Patterns": [["precipitation+, ndvi+", "sup=0.8"]],
            }
        )
    )

    assert summary["shape"] == "patterns_json"
    assert summary["n_patterns"] == 1
    assert summary["algorithm"] == "GRAANK"


def test_classify_output_handles_optimizer_best_patterns_json():
    summary = classify_output(
        {
            "Algorithm": "ACO-GRAANK",
            "Best Patterns": [[["precipitation+", "ndvi+"], 0.7]],
        }
    )

    assert summary["shape"] == "best_patterns_json"
    assert summary["n_patterns"] == 1


def test_classify_output_handles_plain_text():
    summary = classify_output("not json")

    assert summary["shape"] == "text"
    assert summary["preview"] == "not json"


def test_run_probe_suite_with_fake_algorithms():
    fake_algorithms = SimpleNamespace(
        GRAANK=FakeGRAANK,
        TGrad=FakeTGrad,
        AntGRAANK=FakeAntGRAANK,
        GeneticGRAANK=FakeGeneticGRAANK,
        ParticleGRAANK=FakeParticleGRAANK,
    )

    result = run_probe_suite(algorithms_module=fake_algorithms, repeats=2)

    assert result["required_passed"] is True
    by_name = {item["name"]: item for item in result["algorithms"]}
    assert by_name["graank"]["status"] == "ok"
    assert by_name["graank"]["contract_ok"] is True
    assert by_name["tgrad"]["status"] == "ok"
    assert by_name["aco"]["ok_runs"] == 2
    assert by_name["aco"]["output_shapes"] == ["best_patterns_json"]


def test_run_probe_suite_marks_missing_required_as_failure():
    result = run_probe_suite(
        algorithms_module=SimpleNamespace(GRAANK=FakeGRAANK),
        include_optimizers=False,
    )

    assert result["required_passed"] is False
    by_name = {item["name"]: item for item in result["algorithms"]}
    assert by_name["tgrad"]["status"] == "missing"


def test_required_probe_fails_when_output_shape_is_not_patterns_json():
    result = run_probe_suite(
        algorithms_module=SimpleNamespace(GRAANK=FakeGRAANKBestPatterns, TGrad=FakeTGrad),
        include_optimizers=False,
    )

    assert result["required_passed"] is False
    by_name = {item["name"]: item for item in result["algorithms"]}
    assert by_name["graank"]["status"] == "contract_mismatch"
    assert by_name["graank"]["contract_ok"] is False
    assert by_name["graank"]["output_shapes"] == ["best_patterns_json"]


def test_partial_optimizer_failures_are_degraded():
    FakeFlakyAntGRAANK.calls = 0
    result = run_probe_suite(
        algorithms_module=SimpleNamespace(
            GRAANK=FakeGRAANK,
            TGrad=FakeTGrad,
            AntGRAANK=FakeFlakyAntGRAANK,
        ),
        repeats=2,
    )

    by_name = {item["name"]: item for item in result["algorithms"]}
    assert by_name["aco"]["status"] == "degraded"
    assert by_name["aco"]["ok_runs"] == 1
    assert by_name["aco"]["total_runs"] == 2
    assert result["optimizer_status"]["aco"] == "degraded"


def test_main_writes_json_and_fails_on_missing_required():
    output = io.StringIO()
    status = main(
        ["--fail-on-required", "--skip-optimizers"],
        algorithms_module=SimpleNamespace(GRAANK=FakeGRAANK),
        stdout=output,
    )

    assert status == 1
    parsed = json.loads(output.getvalue())
    assert parsed["required_passed"] is False


def test_main_fails_on_required_contract_mismatch():
    output = io.StringIO()
    status = main(
        ["--fail-on-required", "--skip-optimizers"],
        algorithms_module=SimpleNamespace(GRAANK=FakeGRAANKBestPatterns, TGrad=FakeTGrad),
        stdout=output,
    )

    assert status == 1
    parsed = json.loads(output.getvalue())
    assert parsed["required_passed"] is False
    by_name = {item["name"]: item for item in parsed["algorithms"]}
    assert by_name["graank"]["status"] == "contract_mismatch"


def test_public_facades_use_explicit_cpu_and_bounded_deterministic_discovery():
    calls = []

    class PublicGRAANK:
        def __init__(self, data_source, min_sup, eq, device):
            assert device == "cpu"
            calls.append(("graank", min_sup, eq, device))

        def discover(self, search_type, target_col, save_results):
            calls.append((search_type, target_col, save_results))
            return FakeGRAANK(None).discover()

    class PublicTGRAANK:
        def __init__(self, data_source, min_sup, eq, min_rep, device):
            assert device == "cpu"
            calls.append(("tgraank", min_sup, min_rep, eq, device))

        def discover(
            self, target_col, transformations, search_algorithm, max_iteration, save_results
        ):
            calls.append(
                (target_col, transformations, search_algorithm, max_iteration, save_results)
            )
            return FakeTGrad(None).discover_tgp()

    result = run_probe_suite(
        algorithms_module=SimpleNamespace(GRAANK=PublicGRAANK, TGRAANK=PublicTGRAANK),
        include_optimizers=False,
    )

    assert result["required_passed"] is True
    assert calls == [
        ("graank", 0.4, False, "cpu"),
        ("apriori", None, False),
        ("tgraank", 0.4, 0.5, False, "cpu"),
        (2, "all", "apriori", 3, False),
    ]
    assert result["algorithms"][1]["class_name"] == "TGRAANK"
    assert result["algorithms"][1]["runs"][0]["class_name"] == "TGRAANK"


def test_native_error_with_patterns_is_not_a_successful_probe():
    class PublicTGRAANK:
        def __init__(self, **kwargs):
            pass

        def discover(self, **kwargs):
            return json.dumps({"Error": "invalid target", "Patterns": []})

    result = run_probe_suite(
        algorithms_module=SimpleNamespace(GRAANK=FakeGRAANK, TGRAANK=PublicTGRAANK),
        include_optimizers=False,
    )

    assert result["required_passed"] is False
    temporal = result["algorithms"][1]
    assert temporal["status"] == "error"
    assert "invalid target" in temporal["runs"][0]["error"]


def test_real_cpu_probe_uses_public_temporal_facade(tmp_path, monkeypatch):
    pytest.importorskip("so4gp")
    pytest.importorskip("pandas")
    monkeypatch.chdir(tmp_path)
    result = run_probe_suite(include_optimizers=False)
    assert result["required_passed"] is True
    assert result["packages"]["so4gp"] == "1.1.2"
    by_name = {item["name"]: item for item in result["algorithms"]}
    assert by_name["graank"]["class_name"] == "GRAANK"
    assert by_name["tgrad"]["class_name"] == "TGRAANK"
    assert by_name["tgrad"]["runs"][0]["output"]["n_patterns"] > 0
    assert not list(tmp_path.iterdir())
