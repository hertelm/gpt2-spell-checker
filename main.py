import argparse

import yaml
import time
from tqdm import tqdm

from src.gpt2_spell_checker import GPT2SpellChecker


def read_config(config_file: str):
    with open(config_file) as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def main(args):
    config = read_config(args.config_file)
    print(config)

    spell_checker = GPT2SpellChecker(model=config["model"],
                                     tokenizer=config["tokenizer"],
                                     n_words=config["n_words"],
                                     max_ed=config["maximum_edit_distance"],
                                     min_len_per_ed=config["minimum_length_per_ed"],
                                     beam_width=config["beam_width"],
                                     penalties=config["penalties"],
                                     correct_real_words=config["correct_real_words"],
                                     real_word_penalty=config["real_word_penalty"],
                                     first_char_penalty=config["first_char_penalty"],
                                     correct_spaces=config["correct_spaces"],
                                     max_ed_splits=config["maximum_edit_distance_splits"],
                                     prune_candidates=config["prune_candidates"],
                                     prune_beams=config["prune_beams"],
                                     pruning_delta=config["pruning_delta"])

    if config["input_file"] == "None":
        while True:
            query = input("> ")
            result = spell_checker.correct(query, verbose=config["verbose"])
            print(result)

    else:
        with open(config["input_file"]) as in_file:
            sequences = in_file.read().splitlines()
        sequences = sequences[args.start:(args.end if args.end is None else args.end + 1)]
        if config["output_file"] != "None":
            out_file_name = config["output_file"]
            if args.start is not None:
                out_file_name += f".{args.start}-{args.end}"
            out_file = open(out_file_name, "w")
        else:
            out_file = None
        runtime = 0
        for sequence in tqdm(sequences):
            is_upper = sequence.isupper()
            if is_upper:
                sequence = sequence.lower()
            start_time = time.time()
            result = spell_checker.correct(sequence, verbose=config["verbose"])
            runtime += time.time() - start_time
            if is_upper:
                result = result.upper()
            if out_file is not None:
                out_file.write(result)
                out_file.write("\n")
        if out_file is not None:
            out_file.write(str(runtime))
            out_file.write("\n")
            out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="config.yml", nargs="?")
    parser.add_argument("--start", type=int, default=None, required=False)
    parser.add_argument("--end", type=int, default=None, required=False)
    args = parser.parse_args()
    main(args)
