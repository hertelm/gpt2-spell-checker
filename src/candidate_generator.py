from typing import Set, Dict, Tuple, List

from itertools import combinations
from tqdm import tqdm
from nltk import edit_distance


HARDCODED_WORDS = [
    "I'm",
    "can't",
    "Can't",
    "don't",
    "Don't",
    "won't",
    "Won't",
    "it's",
    "It's",
    "he's",
    "He's",
    "she's",
    "She's",
    "we'll",
    "We'll",
    "you'll",
    "You'll",
    "he'd",
    "He'd",
    "she'd",
    "She'd",
    "we'd",
    "We'd",
    "they'd",
    "They'd"
]


def get_stump(word: str, remove_positions: Set[int]) -> str:
    chars = []
    for pos, char in enumerate(word):
        if pos not in remove_positions:
            chars.append(char)
    return "".join(chars)


def get_stumps(word: str, remove_n_chars: int) -> Set[str]:
    stumps = set()
    positions = list(range(len(word)))
    for remove_positions in combinations(positions, remove_n_chars):
        remove_positions = set(remove_positions)
        stump = get_stump(word, remove_positions)
        stumps.add(stump)
    return stumps


class CandidateGenerator:
    def __init__(self,
                 n_words: int,
                 max_ed: int,
                 min_len_per_ed: Dict[int, int]):
        self.n_words = n_words
        self.words = set()
        self._read_words()
        self._add_hardcoded_words()
        self.max_ed = max_ed
        self.min_len_per_ed = min_len_per_ed
        self.stump_index = {}
        self._create_stump_index()

    def _read_words(self):
        for line in open("data/word_frequencies.txt"):
            word = line.split()[0]
            is_word = True
            for char in word:
                if not char.isalpha() and char != "-":
                    is_word = False
                    break
            if not is_word:
                continue
            self.words.add(word)
            if len(self.words) == self.n_words:
                break

    def _add_hardcoded_words(self):
        for word in HARDCODED_WORDS:
            self.words.add(word)

    def _create_stump_index(self):
        print("creating word stump index...")
        for word in tqdm(self.words):
            if word not in self.stump_index:
                self.stump_index[word] = set()
            self.stump_index[word].add(word)
            for ed in range(1, self.max_ed + 1):
                if len(word) < self.min_len_per_ed[ed]:
                    continue
                stumps = get_stumps(word, ed)
                for stump in stumps:
                    if stump not in self.stump_index:
                        self.stump_index[stump] = set()
                    self.stump_index[stump].add(word)

    def _query_stump_index(self, word) -> Set[str]:
        candidates = set()
        candidates.add(word)
        if word in self.stump_index:
            candidates.update(self.stump_index[word])
        for remove_n_chars in range(1, min(len(word), self.max_ed) + 1):
            stumps = get_stumps(word, remove_n_chars)
            for stump in stumps:
                if stump in self.stump_index:
                    candidates.update(self.stump_index[stump])
        return candidates

    def _filter_candidates(self, word: str, candidates: Set[str], assume_first_char_correct: bool) \
            -> List[Tuple[str, int]]:
        filtered_candidates = []
        for candidate in candidates:
            if word == candidate:
                continue
            if assume_first_char_correct and word[0].lower() != candidate[0].lower():
                continue
            ed = edit_distance(word, candidate, transpositions=True)
            if ed <= self.max_ed and len(candidate) >= self.min_len_per_ed[ed]:
                filtered_candidates.append((candidate, ed))
        return filtered_candidates

    def get_candidates(self, word, assume_first_char_correct=False) -> List[Tuple[str, int]]:
        candidates = self._query_stump_index(word)
        candidates = self._filter_candidates(word, candidates, assume_first_char_correct)
        return candidates

    def is_word(self, word):
        return word in self.words
