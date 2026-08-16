import ast
from unittest.mock import patch

from shinka.database.complexity import analyze_code_metrics


PYTHON_PROGRAM = """\
def classify(values):
    total = 0
    for value in values:
        if value > 0:
            total += value
    return total
"""

GO_PROGRAM = """\
package main

func score(xs []int) int {
    total := 0
    for _, x := range xs {
        if x > 0 {
            total += x
        }
    }
    return total
}
"""


def test_python_complexity_metrics_contract():
    assert analyze_code_metrics(PYTHON_PROGRAM) == {
        "cyclomatic_complexity": 3,
        "average_cyclomatic_complexity": 3.0,
        "halstead_volume": 13.931568569324174,
        "halstead_difficulty": 1.3333333333333333,
        "halstead_effort": 18.575424759098897,
        "lines_of_code": 6,
        "logical_lines_of_code": 6,
        "comments": 0,
        "maintainability_index": 108.67212134673596,
        "max_nesting_depth": 3,
        "complexity_score": 0.364,
    }


def test_invalid_python_preserves_cpp_fallback_metrics():
    assert analyze_code_metrics("def broken(:\n    pass\n") == {
        "cyclomatic_complexity": 1,
        "average_cyclomatic_complexity": 1,
        "halstead_volume": 1,
        "halstead_difficulty": 1.0,
        "halstead_effort": 1,
        "lines_of_code": 3,
        "logical_lines_of_code": 2,
        "comments": 0,
        "maintainability_index": 145.09360748831728,
        "max_nesting_depth": 0,
        "complexity_score": 0.1,
    }


def test_python_complexity_parses_ast_once():
    with patch("shinka.database.complexity.ast.parse", wraps=ast.parse) as parse:
        analyze_code_metrics(PYTHON_PROGRAM)

    assert parse.call_count == 1


def test_go_uses_regex_complexity_analysis():
    assert analyze_code_metrics(GO_PROGRAM, language="go") == {
        "cyclomatic_complexity": 3,
        "average_cyclomatic_complexity": 3,
        "halstead_volume": 15.84962500721156,
        "halstead_difficulty": 1.0,
        "halstead_effort": 15.84962500721156,
        "lines_of_code": 12,
        "logical_lines_of_code": 10,
        "comments": 0,
        "maintainability_index": 91.50444811614277,
        "max_nesting_depth": 3,
        "complexity_score": 0.38,
    }
