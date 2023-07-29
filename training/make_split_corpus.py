"""
https://github.com/cl-tohoku/bert-japanese/blob/v2.0/make_corpus_wiki.py
を参考に作成されました
"""

import json
import gzip
import re
import os
import sys

MIN_LENGTH = 3
PUNCTUATIONAS = ['。', '？', '?', '!', '!']
NUM_FILES = 32


def process(input_file_name: str, output_dirname: str):
    os.makedirs(output_dirname, exist_ok=True)
    output_files = [open(os.path.join(output_dirname, f'corpus_{i}.txt'), mode='wt') for i in range(NUM_FILES)]

    current_id = ''
    idx = 0
    with gzip.open(input_file_name, 'rt') as input_file:
        for row in input_file:
            data = json.loads(row)
            if 'index' in data and '_id' in data['index']:
                current_id = data['index']['_id']
                continue
            if 'text' not in data:
                continue

            text = data['text']
            title = data.get('title')
            text = preprocess_text(text, title=title)

            sentences = []
            for sentence in split_text(text):
                if len(sentence) < MIN_LENGTH:
                    continue
                if contains_equations(sentence):
                    continue
                sentences.append(sentence)

            if len(sentences) > 0:
                print('\n'.join(sentences), file=output_files[ idx % NUM_FILES ])
                print('', file=output_files[ idx % NUM_FILES ])
                idx += 1

    for f in output_files:
        f.close()


def split_text(text: str) -> list:
    sentences = []
    prev_is_punctuation = False
    chars = []
    for c in text:
        if c in PUNCTUATIONAS:
            chars.append(c)
            prev_is_punctuation = True
        elif prev_is_punctuation:
            sentence = ''.join(chars).strip()
            if sentence:
                sentences.append(sentence)
                chars = [c]
            prev_is_punctuation = False
        else:
            chars.append(c)

    sentence = ''.join(chars).strip()
    if sentence:
        sentences.append(sentence)
    return sentences



def preprocess_text(text: str, title: str=None) -> str:
    # remove invisible characters
    text = "".join(c for c in text if c.isprintable())

    # remove templates
    text = re.sub(r"\[\d+?\]", "", text)
    text = re.sub(r"\[要.+?\]", "", text)
    text = re.sub(r"\{\{+[^{}]+?\}\}+", "", text)

    # remove navigation
    if title is not None:
        text = re.sub(r"^.+? \> " + re.escape(title), "", text)

    # remove footnotes
    text = re.sub(r" \^ .+", "", text)
    # remove annotations
    text = re.sub(r"\[(要出典|リンク切れ|.+?\?)\]", "", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_equations(text: str) -> bool:
    # filter out text containing equations
    return "\displaystyle" in text


if __name__ == '__main__':
    input_filename, output_dir = sys.argv[1], sys.argv[2]
    process(input_filename, output_dir)
