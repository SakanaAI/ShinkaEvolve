from shinka.database.complexity import analyze_code_metrics


def test_go_uses_regex_complexity_analysis():
    metrics = analyze_code_metrics(
        """
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
""",
        language="go",
    )

    assert metrics["cyclomatic_complexity"] > 1
