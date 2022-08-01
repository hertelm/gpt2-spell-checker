import sys
import random


def read_file(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def write_file(lines, path):
    with open(path, "w") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def split(array, split_point):
    return array[:split_point], array[split_point:]


if __name__ == "__main__":
    random.seed(42)
    benchmark_dir = "benchmarks/bea60k"
    if "--spaces" in sys.argv:
        benchmark_dir += ".spaces"
    benchmark_dir += "/"
    correct_lines = read_file(benchmark_dir + "correct.txt")
    corrupt_lines = read_file(benchmark_dir + "corrupt.txt")
    n = len(correct_lines)
    n_dev = 10000
    correct_dev, correct_test = split(correct_lines, n_dev)
    corrupt_dev, corrupt_test = split(corrupt_lines, n_dev)
    write_file(correct_dev, benchmark_dir + "development/correct.txt")
    write_file(corrupt_dev, benchmark_dir + "development/corrupt.txt")
    indices = list(range(len(correct_test)))
    random.shuffle(indices)
    correct_test = [correct_test[i] for i in indices]
    corrupt_test = [corrupt_test[i] for i in indices]
    write_file(correct_test, benchmark_dir + "test/correct.txt")
    write_file(corrupt_test, benchmark_dir + "test/corrupt.txt")
