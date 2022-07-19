
SHORT_WORDS = {"I", "a", "A"}


def contains_alpha(word):
    for char in word:
        if char.isalpha():
            return True
    return False


def filter(word):
    if not contains_alpha(word):
        return True
    if len(word) == 1 and word not in SHORT_WORDS:
        return True
    return False


if __name__ == "__main__":
    for line in open("data/word_frequencies_100k.txt"):
        line = line[:-1]
        word, frequency = line.split()
        if not filter(word):
            print(word, frequency)
