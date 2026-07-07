package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// EVOLVE-BLOCK-START
func collatzSteps(n int64) int {
	steps := 0
	value := n
	for value != 1 {
		if value%2 == 0 {
			value /= 2
		} else {
			value = 3*value + 1
		}
		steps++
	}
	return steps
}

func solveStoppingTimes(queries []int64) []int {
	answers := make([]int, len(queries))
	for i, q := range queries {
		answers[i] = collatzSteps(q)
	}
	return answers
}

// EVOLVE-BLOCK-END

func readQueries(inputPath string) ([]int64, error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var queries []int64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		value, err := strconv.ParseInt(line, 10, 64)
		if err != nil {
			return nil, err
		}
		if value < 1 {
			return nil, fmt.Errorf("query must be >= 1: %d", value)
		}
		queries = append(queries, value)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return queries, nil
}

func writeAnswers(outputPath string, answers []int) error {
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, value := range answers {
		if i > 0 {
			if _, err := writer.WriteString("\n"); err != nil {
				return err
			}
		}
		if _, err := writer.WriteString(strconv.Itoa(value)); err != nil {
			return err
		}
	}
	return writer.Flush()
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: go run initial.go <input_path> <output_path>")
		os.Exit(1)
	}

	inputPath := os.Args[1]
	outputPath := os.Args[2]
	queries, err := readQueries(inputPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	answers := solveStoppingTimes(queries)
	if len(answers) != len(queries) {
		fmt.Fprintf(
			os.Stderr,
			"Length mismatch: got %d answers for %d queries\n",
			len(answers),
			len(queries),
		)
		os.Exit(1)
	}

	if err := writeAnswers(outputPath, answers); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
