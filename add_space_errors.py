import random
import sys

from src.gpt2_spell_checker import tokenize


def can_merge(token1: str, token2: str) -> bool:
    return token1.isalpha() and token2.isalpha()


def split_token(token: str) -> str:
    if token.isalpha() and len(token) > 1:
        split_position = random.choice(list(range(1, len(token))))
        return token[:split_position] + " " + token[split_position:]
    return token


def insert_space_errors(text: str) -> str:
    tokens = tokenize(text)
    processed_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == " ":
            processed_tokens.append(tokens[i])
            i += 1
            continue
        random_number = random.random()
        if random_number < ERROR_PROBABILITY / 2:
            split = split_token(tokens[i])
            processed_tokens.append(split)
        elif random_number < ERROR_PROBABILITY:
            if i + 2 < len(tokens) and tokens[i + 1] == " " and can_merge(tokens[i], tokens[i + 2]):
                merged = tokens[i] + tokens[i + 2]
                processed_tokens.append(merged)
                i += 2
            else:
                processed_tokens.append(tokens[i])
        else:
            processed_tokens.append(tokens[i])
        i += 1
    return "".join(processed_tokens)


if __name__ == "__main__":
    random.seed(1)
    ERROR_PROBABILITY = 0.1

    file = sys.argv[1] if len(sys.argv) > 1 else "data/bea60k.repaired/corrupt.txt"
    
    with open(file) as f:
        lines = f.read().splitlines()
    for line in lines:
        result = insert_space_errors(line)
        print(result)
