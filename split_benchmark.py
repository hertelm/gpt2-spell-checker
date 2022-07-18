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
    correct_lines = read_file("benchmarks/bea60k/correct.txt")
    corrupt_lines = read_file("benchmarks/bea60k/corrupt.txt")
    n = len(correct_lines)
    n_dev = 10000
    correct_dev, correct_test = split(correct_lines, n_dev)
    corrupt_dev, corrupt_test = split(corrupt_lines, n_dev)
    write_file(correct_dev, "benchmarks/bea60k/development/correct.txt")
    write_file(corrupt_dev, "benchmarks/bea60k/development/corrupt.txt")
    indices = list(range(len(correct_test)))
    random.shuffle(indices)
    correct_test = [correct_test[i] for i in indices]
    corrupt_test = [corrupt_test[i] for i in indices]
    write_file(correct_test, "benchmarks/bea60k/test/correct.txt")
    write_file(corrupt_test, "benchmarks/bea60k/test/corrupt.txt")
