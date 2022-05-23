import argparse

import yaml

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

    while True:
        query = input("> ")
        result = spell_checker.correct(query, verbose=config["verbose"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="config.yml")
    args = parser.parse_args()
    main(args)
