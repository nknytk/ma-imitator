import fugashi
from sudachipy import Dictionary, SplitMode


class FugashiTokenizer:
    def __init__(self):
        self.tagger = fugashi.Tagger()

    def tokenize(self, text: str) -> list:
        tokens = []
        for word in self.tagger(text):
            tokens.append((word.surface, word.pos.split(',')[0]))
        return tokens


class SudachiTokenizer:
    def __init__(self, split_mode='B'):
        self.tokenizer = Dictionary(dict='full').create()
        self.split_mode = SplitMode.B if split_mode == 'B' else SplitMode.C

    def tokenize(self, text: str) -> list:
        tokens = []
        for word in self.tokenizer.tokenize(text):
            if len(word.surface()) > 0:
                tokens.append((word.surface(), word.part_of_speech()[0]))
        return tokens
