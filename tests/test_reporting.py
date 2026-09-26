"""Terminal reporting preserves percentage units and unresolved outcomes."""

from copy import deepcopy

from dlm_iclr.cli import main
from dlm_iclr.runtime.reporting import format_results


def report():
    return {
        "requests": 10,
        "structures": "outputs/mp20/samples/final.jsonl",
        "direct": {
            "counts": {"requests": 10},
            "metrics": {"struct_valid": 100.0, "comp_valid": 0.0,
                        "cov_precision": 80.0, "wdist_density": 0.125},
        },
        "sun": {"metrics": {
            "SUN": {"percent_bounds": [10.0, 10.0], "pending": 0},
            "MSUN": {"percent_bounds": [40.0, 40.0], "pending": 0},
            "VUN": {"percent_bounds": [70.0, 70.0], "pending": 0},
        }},
    }


def test_completed_rates_use_percentages_but_distances_do_not():
    result = format_results(report(), dataset="MP-20")
    assert "MP-20 | 10 requests" in result
    assert "100.00%" in result and "0.00%" in result
    assert "SUN" in result and "10.00%" in result
    assert "MSUN" in result and "40.00%" in result
    assert "VUN" in result and "70.00%" in result
    distance = next(line for line in result.splitlines() if "Density W1" in line)
    assert "0.12" in distance and "%" not in distance
    assert "Coverage recall" not in result


def test_unresolved_metrics_never_turn_into_point_estimates_or_zero():
    value = deepcopy(report())
    value["direct"]["metrics"].update(struct_valid=None, cov_precision=None, wdist_density=None)
    value["direct"]["unknown_counts"] = {"struct_valid": 1}
    value["direct"]["count_bounds"] = {"struct_valid": [8, 9]}
    value["direct"]["coverage_bounds_percent"] = {"cov_precision": [70, 80]}
    value["sun"]["metrics"]["SUN"] = {"percent_bounds": [10, 30], "pending": 2}
    result = format_results(value, dataset="MP-20")
    assert "[80.00, 90.00]% (1 unresolved)" in result
    assert "[70.00, 80.00]% (unresolved)" in result
    assert "[10.00, 30.00]% (2 unresolved)" in result
    assert "unavailable" in next(line for line in result.splitlines() if "Density W1" in line)


def test_full_run_cli_prints_final_results(monkeypatch, capsys):
    from dlm_iclr.runtime import pipeline

    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: {"final": report()})
    assert main(["run", "--plans", "plans.jsonl"]) == 0
    output = capsys.readouterr().out
    assert "CrystalDLM results" in output
    assert "Physical evaluation" in output and "MSUN" in output
    assert '"metrics"' not in output
