# GPT2-spell-checker

This is the code for the GPT2-spell-checker, a spell checker based on the famous language model and a simple error model.

## Installation

```commandline
git clone https://github.com/hertelm/gpt2-spell-checker.git
cd gpt2-spell-checker
python3 -m virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

Run the spell checker interactively:

`python3 main.py`

Run the spell checker on a file:

`python3 main.py -f <input_file> -o <output_file>`

## Configuration

Many parameters, such as the GPT-2 model used, whether to prune beams, the pruning delta, whether to correct whitespaces, the penalty parameters of the error model, and many more can be modified in the file `config.yml`.

## Publication

When you use the GPT-2-spell-checker in your work, please consider citing our upcoming publication:
*Matthias Hertel and Hannah Bast: "GPT-2-spell-checker: a tool for language-model-based spelling correction and evaluation" (2022, under review)*
