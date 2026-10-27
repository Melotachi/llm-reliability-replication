"""Tests for Student 2 failure-analysis helper."""

from __future__ import annotations

from scripts.analyze_failures import format_markdown, summarize_failures


def test_summarize_failures_counts_initial_categories_and_repairs():
    results = {
        "run_id": "unit_run",
        "per_problem": [
            {
                "task_id": "HumanEval/0",
                "samples": [
                    {"passed": True, "failure_category": ""},
                    {
                        "passed": False,
                        "failure_category": "assertion_error",
                        "repaired_passed": True,
                        "repaired_failure_category": "",
                    },
                    {
                        "passed": False,
                        "failure_category": "type_error",
                        "repaired_passed": False,
                        "repaired_failure_category": "assertion_error",
                    },
                ],
            }
        ],
    }

    summary = summarize_failures(results)

    assert summary["total_samples"] == 3
    assert summary["initially_passing"] == 1
    assert summary["initially_failing"] == 2
    assert summary["repair_attempts"] == 2
    assert summary["successful_repairs"] == 1
    assert summary["initial_failure_counts"] == {
        "assertion_error": 1,
        "type_error": 1,
    }
    assert summary["repaired_failure_counts"] == {"assertion_error": 1}
    assert summary["repair_transitions"] == {
        "assertion_error->pass": 1,
        "type_error->assertion_error": 1,
    }


def test_format_markdown_includes_core_sections():
    summary = summarize_failures({"run_id": "empty", "per_problem": []})

    markdown = format_markdown(summary)

    assert "# Failure Analysis: empty" in markdown
    assert "## Initial Failure Types" in markdown
    assert "- None" in markdown
