import os.path
import pickle
from typing import Set, Dict, Tuple, List, Optional

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
                 min_len_per_ed: Dict[int, int],
                 allow_space_edits: bool,
                 max_ed_splits: int):
        self.n_words = n_words
        self.words = set()
        self._read_words()
        self._add_hardcoded_words()
        self.max_ed = max_ed
        self.min_len_per_ed = min_len_per_ed
        self.stump_index = {}
        self._load_or_create_stump_index()
        self.allow_space_edits = allow_space_edits
        self.max_ed_splits = max_ed_splits

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

    def _load_or_create_stump_index(self):
        path = "data/word_stump_index.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.stump_index = pickle.load(f)
        else:
            self._create_stump_index()
            with open(path, "wb") as f:
                pickle.dump(self.stump_index, f)

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

    def _filter_candidates(self,
                           word: str,
                           candidates: Set[str],
                           max_ed: int,
                           assume_first_char_correct: bool) \
            -> List[Tuple[str, int, bool]]:
        filtered_candidates = []
        for candidate in candidates:
            if word == candidate:
                continue
            if assume_first_char_correct and word[0].lower() != candidate[0].lower():
                continue
            ed = edit_distance(word, candidate, transpositions=True)
            if ed <= max_ed and len(candidate) >= self.min_len_per_ed[ed]:
                filtered_candidates.append((candidate, ed, False))
        return filtered_candidates

    def _get_candidate_words(self, word: str, max_ed: int, assume_first_char_correct=False) \
            -> List[Tuple[str, int, bool]]:
        candidates = self._query_stump_index(word)
        candidates = self._filter_candidates(word, candidates, max_ed, assume_first_char_correct)
        return candidates

    def _get_split_candidates(self, token: str, max_ed: int) -> List[Tuple[str, int, bool]]:
        split_candidates = []
        for i in range(1, len(token)):
            left = token[:i]
            left_candidates = []
            if self.is_word(left):
                left_candidates.append((left, 0, False))
            left_candidates.extend(self._get_candidate_words(left, max_ed=max_ed - 1))
            right = token[i:]
            right_candidates = []
            if self.is_word(right):
                right_candidates.append((right, 0, False))
            right_candidates.extend(self._get_candidate_words(right, max_ed=max_ed - 1))
            for c_left, ed1, _ in left_candidates:
                for c_right, ed2, _ in right_candidates:
                    if ed1 + ed2 + 1 <= max_ed:
                        split_candidates.append((c_left + " " + c_right, ed1 + ed2 + 1, False))
        return split_candidates

    def _get_merge_candidates(self, token1: str, token2: str) -> List[Tuple[str, int, bool]]:
        merged = token1 + token2
        merge_candidates = []
        if self.is_word(merged):
            merge_candidates.append((merged, 1, True))
        candidates = self._get_candidate_words(merged, max_ed=self.max_ed - 1)
        for candidate, ed, _ in candidates:
            merge_candidates.append((candidate, ed + 1, True))
        return merge_candidates

    def _get_candidates(self, token: str, next_token: Optional[str]) -> List[Tuple[str, int, bool]]:
        candidates = self._get_candidate_words(token, max_ed=self.max_ed)
        if self.allow_space_edits:
            candidates.extend(self._get_split_candidates(token, max_ed=self.max_ed_splits))
            if next_token is not None and next_token.isalpha():
                candidates.extend(self._get_merge_candidates(token, next_token))
        return candidates

    def get_candidates(self, token: str, next_token: Optional[str]) -> List[Tuple[str, int, bool]]:
        candidates = self._get_candidates(token, next_token)
        if token[0].isupper() and len(token) == 1 or token[1:].islower():
            lower_candidates = self._get_candidates(token.lower(), next_token)
            lower_cands_filtered = []
            for candidate, ed, delay in lower_candidates:
                upper_candidate = candidate[0].upper() + candidate[1:]
                if self.is_word(upper_candidate):
                    continue
                if token == upper_candidate:
                    continue
                lower_cands_filtered.append((upper_candidate, ed, delay))
            candidates = list(set(candidates).union(set(lower_cands_filtered)))
        return candidates

    def is_word(self, word):
        return word in self.words
