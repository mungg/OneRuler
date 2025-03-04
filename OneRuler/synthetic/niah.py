# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Create a dataset jsonl file for needle in a haystack.

python niah.py \
    --save_dir=./ \
    --save_name=niah_single \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --max_seq_length=4096 \
    --tokens_to_generate=128 \
    --num_samples=10 \
    --template="Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"
"""
import os
import re
import json
import uuid
import argparse
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 
from tokenizer import select_tokenizer
import stanza
import ast


parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--data_dir", type=Path, default='../data', help='data folder of books, prompt and vocabs')

parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--tokenizer_path", type=str, required=True, help='path to the tokenizer model')
parser.add_argument("--tokenizer_type",  type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument("--lang", type=str, default='en', help='language')
parser.add_argument("--inst_lang", type=str, default='en', help='language for instruction')
parser.add_argument("--xling", action="store_true", help="Enable xling mode")
parser.add_argument("--word_by_index", action="store_true", help="The key of the needle is retrieved sequentially from the vocab list by index")

# Complexity Configurations
parser.add_argument("--num_needle_k", type=int, default=1)
parser.add_argument("--num_needle_v", type=int, default=1)
parser.add_argument("--num_needle_q", type=int, default=1)
parser.add_argument("--relevant_needle", type=int, default=1)
parser.add_argument("--type_haystack", type=str, default='essay', help='[Options] repeat, essay, needle.')
parser.add_argument("--type_needle_k", type=str, default='words', help='[Options] numbers, words, uuids.')
parser.add_argument("--type_needle_v", type=str, default='numbers', help='[Options] numbers, words, uuids.')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
args.num_needle_k = max(args.num_needle_k, args.num_needle_q)
assert args.num_needle_q == 1 or args.num_needle_q == 2

curr_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = args.data_dir

module = importlib.import_module(f"constants")
lang = args.lang.lower()
inst_lang = args.inst_lang.lower() if args.xling else lang
     
is_stanza = True

print(f'stanza directory ------------> {os.environ["STANZA_RESOURCES_DIR"]}')
try: 
    stanza.download(lang, model_dir=os.environ["STANZA_RESOURCES_DIR"]) 
except Exception as e:
    print(f"Error: {e}") 
    is_stanza = False

# Load Tokenizer
TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

# Words
noun_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), data_folder, "vocab/100_noun_list_translated.tsv"), sep="\t")

if args.xling:
    ## Exclude ambiguous words for cross-lingual
    skip_xling_indices = [21, 28, 42, 64, 65]
    skip_en_word = noun_df['en'].iloc[skip_xling_indices]
    noun_df = noun_df.drop(skip_xling_indices)

    print(f"[xling] index {skip_xling_indices} of vocab list removed") 

nouns = noun_df[args.lang].dropna().tolist() if args.lang in noun_df.columns else print(f"noun_list doesn't have {args.lang}")
words = [noun.strip() for noun in nouns]


# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))

# Define template 
template_dict = json.load(open(os.path.join(curr_folder, data_folder, f"prompt/{inst_lang}/niah.txt"), "r", encoding="utf-8"))
task_template = template_dict['task']
context_dict = json.load(open(os.path.join(curr_folder, data_folder, f"prompt//{lang}/niah.txt"), "r", encoding="utf-8"))
needle = context_dict[f"needle_{args.type_needle_v}"]
if args.num_needle_q == 1:
    query_type = 'single'
elif args.num_needle_q == 2:
    query_type = 'multi'
else:
    print("query num {} is not handled")
    exit()
if lang in ["zh", "ja"]:
    question_template = template_dict[f"question_{query_type}_{args.type_needle_v}"] + template_dict['please_list']+ template_dict[f'if_no_{args.type_needle_v}']
    answer_template = template_dict['answer_prefix'] + template_dict[f"answer_{args.type_needle_v}"]
else:
    question_template = template_dict[f"question_{query_type}_{args.type_needle_v}"] + ' ' + template_dict['please_list'] + ' ' + template_dict[f'if_no_{args.type_needle_v}']
    answer_template = template_dict['answer_prefix'] + ' ' + template_dict[f"answer_{args.type_needle_v}"]
template = task_template + question_template + answer_template 


# Define Haystack Format 
if args.type_haystack == 'essay':
    essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/PaulGrahamEssays.json")
    essay = json.load(open(essay))['text']
    haystack = re.sub(r'\s+', " ", essay).split(" ")
elif args.type_haystack == 'book':
    book_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_folder, f"books/{args.lang}")
    book =''
    for book_name in os.listdir(book_path): 
        if book_name.endswith(".txt"):
            content = open(os.path.join(book_path, book_name), 'r', encoding="utf-8").read()
            if lang == 'en':
                fixed_text =content
                book += fixed_text
            else:
                fixed_text =content
                book += fixed_text
    
    book = re.sub(r'\s+', " ", book)
    if is_stanza:
        multi_tokenizer = stanza.Pipeline(lang=lang, processors='tokenize')
        tokenized_book = multi_tokenizer(book)
        haystack = [sentence.text.strip() for sentence in tokenized_book.sentences]
    else:
        tokenized_book = re.split(r'(?<=[?.!])\s+', book.strip())
        haystack = [sentence.strip() for sentence in tokenized_book]
    
    print(f"len of haystack is {len(haystack)}")

else:
    raise NotImplementedError(f'{args.type_haystack} is not implemented.')


def generate_random_number(num_digits=7):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def generate_random_word():
    word = random.choice(words)
    return word

def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def generate_random(type_needle: str):
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    else:
        raise NotImplementedError(f'{args.type_needle} is not implemented.')
def expand_haystack(haystack, num_haystack):
    if len(haystack) >= num_haystack:
        return haystack[:num_haystack]
    
    expanded_haystack = haystack.copy()
    while len(expanded_haystack) < num_haystack:
        expanded_haystack.extend(haystack)
    
    return expanded_haystack[:num_haystack]

def add_period(sentence: str, lang: str) -> str:
    """
    Adds the correct period to a given sentence based on the language.

    - Strips trailing spaces
    - Appends the correct punctuation mark based on the language
    """
    sentence = sentence.rstrip()  # Remove trailing spaces

    if lang in ["zh", "ja"]:  # Full-width period for Chinese and Japanese
        return sentence + "。"
    elif lang in ["fa"]:  # Arabic-style period for Persian
        return sentence + "۔"
    elif lang in [
        "en", "ko", "pl", "vi", "ta", "hu", "fr", "no", "uk", "ru", "de", "es", "sv", "fi", "cs", "sr", "pt", "it", "sw", "nl", "st", "hi", "da"
    ]:
        return sentence + "."
    else:
        raise ValueError(f"Unsupported language code: {lang}")

def find_optimal_sentences_multi_targets(sentences, target, tokenize_func=None):
    sentence_tokens = [len(tokenize_func.text_to_tokens(sent)) for sent in sentences]
    
    cumsum = [0]
    for tokens in sentence_tokens:
        cumsum.append(cumsum[-1] + tokens)
    total_available = cumsum[-1]
    
    left, right = 0, len(sentences) * (target // total_available + 1)
    best_n = 0
    min_diff = float('inf')
    
    while left <= right:
        mid = (left + right) // 2
        full_repeats = mid // len(sentences)
        remaining = mid % len(sentences)
        
        total = (full_repeats * total_available)
        if remaining > 0:
            total += cumsum[remaining]
            
        # Key change: only update best_n if we're under target
        if total <= target and (target - total) < min_diff:
            min_diff = target - total
            best_n = mid
            
        if total < target:
            left = mid + 1
        else:
            right = mid - 1
    
    full_repeats = best_n // len(sentences)
    remaining = best_n % len(sentences)
    
    result_indices = []
    for _ in range(full_repeats):
        result_indices.extend(range(len(sentences)))
    if remaining > 0:
        result_indices.extend(range(remaining))
        
    return result_indices


def generate_input_output(num_haystack, index=None):
    keys, values, needles = [], [], []
    if args.num_needle_k == 1 and index is not None:
        keys.append(words[index])
        value = []
        for _ in range(args.num_needle_v):
            value.append(generate_random(args.type_needle_v))
            needles.append(add_period(needle.format(
                key=keys[-1], 
                value=value[-1],
            ), lang=lang))
        values.append(value)
    else:
        # keys.append(generate_random(args.type_needle_k))
        keys = random.sample(words, args.num_needle_k)  ### for making sure that there are no duplications
        for i_needle in range(args.num_needle_k):
            value = []
            for _ in range(args.num_needle_v):
                value.append(generate_random(args.type_needle_v))
                needles.append(add_period(needle.format(
                    key=keys[i_needle], 
                    value=value[-1],
                ), lang=lang))
            values.append(value)
        
    random.Random(args.random_seed).shuffle(needles)
    
    # Context
    if  args.type_haystack in {'essay', 'book'}:     
        document_sents = expand_haystack(haystack, num_haystack)
        insertion_positions = [0] + \
                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                              [len(document_sents)]
        document_sents_list = []
        for i in range(1,len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            if lang in ["zh", "ja"]:
                join_str = ""
            else:
                join_str =" "
            document_sents_list.append(join_str.join(document_sents[last_pos:next_pos]))
            if i-1 < len(needles):
                document_sents_list.append(needles[i-1])
        context = join_str.join(document_sents_list)

    ## Query and Answer
    indices = random.sample(range(args.num_needle_k), args.num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    if not args.relevant_needle:
        while(True):
            word = generate_random(args.type_needle_k)
            if word not in keys:
                queries = [word]
                break
        answers = [template_dict['none']]

    if args.xling:
        for idx, query in enumerate(queries):
            translated_noun = noun_df[noun_df[lang].str.strip() == query][args.inst_lang]
            queries[idx] = translated_noun.values[0].strip()
    if args.num_needle_q == 2:
        input_text = template.format(
            context=context,
            query1=queries[0],
            query2=queries[1],
        )
    else:
        input_text = template.format(
            context=context,
            query1=queries[0],
        )
    return input_text, answers

def generate_samples(num_samples: int):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    num_haystack = len(find_optimal_sentences_multi_targets(haystack, args.max_seq_length, TOKENIZER))
    print('Num haystack:', num_haystack)
    
    for index in tqdm(range(num_samples)):
        used_haystack = num_haystack
        while(True):
            try:
                word_index = index if args.word_by_index else None
                input_text, answer  = generate_input_output(used_haystack, index=word_index)
                length = len(TOKENIZER.text_to_tokens(input_text)) + tokens_to_generate
                break
            except Exception as e:
                print(f"error {e}")

        if args.remove_newline_tab:
            input_text = input_text.replace('\n', ' ').replace('\t', ' ').strip()

        formatted_output = {
            'index': index,
            "input": input_text,
            "outputs": answer,
            "length": length,
        }
        write_jsons.append(formatted_output)

    return write_jsons


def main():
    save_file = args.save_dir / f'{args.save_name}' / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    write_jsons = generate_samples(
        num_samples=args.num_samples
    )

    ensure_ascii = False
    with open(save_file, "w", encoding="utf-8") as outfile:
        for tgt in write_jsons:
            json.dump(tgt, outfile, ensure_ascii=ensure_ascii)
            outfile.write('\n')

if __name__ == "__main__":
    main()