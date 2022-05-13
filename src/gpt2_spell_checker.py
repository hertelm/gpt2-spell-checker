from typing import Dict

import torch.cuda
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
import numpy as np
import time

from src.candidate_generator import CandidateGenerator


def load_model(model_name: str) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model


def load_tokenizer(tokenizer_name) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def tokenize(text):
    return re.findall(r"\w[-'\w]*|.", text)


class GPT2SpellChecker:
    def __init__(self,
                 model: str,
                 tokenizer: str,
                 n_words: int,
                 max_ed: int,
                 min_len_per_ed: Dict[int, int],
                 beam_width: int,
                 penalties: Dict[int, float],
                 correct_real_words: bool,
                 real_word_penalty: float,
                 first_char_penalty: float,
                 prune_candidates: bool,
                 prune_beams: bool,
                 pruning_delta: float):
        self.model = load_model(model)
        self.tokenizer = load_tokenizer(tokenizer)
        self.candidate_generator = CandidateGenerator(n_words=n_words,
                                                      max_ed=max_ed,
                                                      min_len_per_ed=min_len_per_ed)
        self.beam_width = beam_width
        self.penalties = penalties
        self.correct_real_words = correct_real_words
        self.real_word_penalty = real_word_penalty
        self.first_char_penalty = first_char_penalty
        self.prune_candidates = prune_candidates
        self.prune_beams = prune_beams
        self.pruning_delta = pruning_delta
        self.device = "cpu"
        self._check_cuda()
        self.total_candidate_runtime = 0

    def _check_cuda(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(self.device)

    def _initial_beam(self):
        input = self.tokenizer(self.tokenizer.bos_token, return_tensors="pt")
        input = input.input_ids.to(self.device)
        with torch.no_grad():
            res = self.model(input)
        beam = {
            "score": 0,
            "probs": torch.softmax(res.logits[0][0], dim=0),
            "past_key_values": res.past_key_values,
            "text": ""
        }
        return beam

    def _update_beams(self, beams):
        for beam in beams:
            inputs = self.tokenizer(beam["candidate"], return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                output = self.model(inputs, past_key_values=beam["past_key_values"])
            beam["probs"] = torch.softmax(output.logits[0][0], dim=0)
            beam["past_key_values"] = output.past_key_values
        return beams

    def _encode_candidates(self, candidates):
        encodings = []
        for candidate in candidates:
            encoding = self.tokenizer(candidate)["input_ids"]
            encodings.append(encoding)
        return encodings

    def _get_beam_search_candidates(self, token, is_space):
        candidates = [(token, 0)]
        if (not self.candidate_generator.is_word(token) or self.correct_real_words) and token[0].isalpha():
            candidates.extend(self.candidate_generator.get_candidates(token))
        if is_space:
            candidates = [(" " + candidate, ed) for candidate, ed in candidates]
        return candidates

    def _get_best_single_token_candidate_score(self, beams, token, candidates, encoded_candidates):
        best = np.inf
        for beam in beams:
            log_probs = torch.log(beam["probs"])
            for i in range(len(candidates)):
                encoded = encoded_candidates[i]
                if len(encoded) == 1:
                    log_prob = log_probs[encoded[0]]
                    ed = candidates[i][1]
                    score = self._get_score(beam["score"], log_prob, token, candidates[i][0], ed)
                    if score < best:
                        best = score
        return best

    def _get_score(self, old_score, log_prob, token, candidate, ed):
        score = old_score - log_prob
        score += self.penalties[ed]
        if ed > 0:
            if self.candidate_generator.is_word(token):
                score += self.real_word_penalty
            first_char = candidate[0] if candidate[0] != " " else candidate[1]
            if token[0] != first_char:
                score += self.first_char_penalty
        return score

    def _beam_search_step(self, beams, token, is_space, verbose):
        n_model_calls = 0
        cand_start_time = time.time()
        candidates = self._get_beam_search_candidates(token, is_space)
        cand_runtime = time.time() - cand_start_time
        self.total_candidate_runtime += cand_runtime
        if verbose:
            print(f"{len(candidates)} candidates ({cand_runtime:.4f} seconds)")
        encoded_candidates = self._encode_candidates([candidate[0] for candidate in candidates])
        new_beams = []
        best_score_in_step = self._get_best_single_token_candidate_score(beams, token, candidates, encoded_candidates)
        if verbose:
            print("best single score:", best_score_in_step)
        for beam in beams:
            for c_i, (candidate, ed) in enumerate(candidates):
                encoded = encoded_candidates[c_i]
                prob = beam["probs"][encoded[0]]
                log_prob = torch.log(prob)
                if len(encoded) > 1:
                    if self.prune_candidates:
                        single_score = self._get_score(beam["score"], log_prob, token, candidate, ed)
                        if single_score > best_score_in_step + self.pruning_delta:
                            continue
                    encoded_tensor = self.tokenizer(candidate, return_tensors="pt")["input_ids"].to(self.device)
                    with torch.no_grad():
                        output = self.model(encoded_tensor[:, :-1], past_key_values=beam["past_key_values"])
                    probs = torch.softmax(output["logits"][0], dim=1)
                    for pos, label in enumerate(encoded_tensor[0][1:]):
                        log_prob += torch.log(probs[pos, label])
                    n_model_calls += 1
                score = self._get_score(beam["score"], log_prob, token, candidate, ed)
                best_score_in_step = min(score, best_score_in_step)
                new_beam = {
                    "score": score,
                    "text": beam["text"] + candidate,
                    "past_key_values": beam["past_key_values"],
                    "candidate": candidate
                }
                new_beams.append(new_beam)
            new_beams = sorted(new_beams, key=lambda beam: beam["score"])
            new_beams = new_beams[:self.beam_width]
        if self.prune_beams:
            new_beams = [b for b in new_beams if b["score"] < new_beams[0]["score"] + self.pruning_delta]
        beams = self._update_beams(new_beams)
        n_model_calls += len(beams)
        if verbose:
            for beam in beams:
                beam_score = beam["score"].item()
                beam_text = beam["text"]
                print(f"{beam_score:.4f} {beam_text}")
            print(n_model_calls, "model calls")
        return beams

    def correct(self, text, verbose=False):
        start_time = time.time()
        self.total_candidate_runtime = 0

        tokens = tokenize(text)

        beam = self._initial_beam()
        beams = [beam]

        is_space = False
        for step, token in enumerate(tokens):
            is_real_word = self.candidate_generator.is_word(token)
            if verbose:
                print(f"== step {step + 1} ==")
                print(f"token: {token} (real word: {is_real_word})")
            step_start_time = time.time()
            if token == " ":
                is_space = True
            else:
                beams = self._beam_search_step(beams, token, is_space, verbose)
                is_space = False
            step_runtime = time.time() - step_start_time
            if verbose:
                print(f"{step_runtime:.4f} seconds")

        if verbose:
            runtime = time.time() - start_time
            print("== result ==")
            print(beams[0]["text"])
            print(f"{runtime:.4f} seconds ({self.total_candidate_runtime:.4f} seconds for candidate generation)")

        return beams[0]["text"]