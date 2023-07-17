import re
import json

PAD_TOKEN_ID =0
UNKNOWN_KANJI_ID = 1
UNKNOWN_NOKANJI_ID = 2


def normalize(text: str) -> str:
    return re.sub('\s+', ' ', text).strip()


def pad(values: list, pad_value: int, max_length: int):
    if len(values) > max_length:
        del(values[max_length])
    else:
        pads = [pad_value] * (max_length - len(values))
        values.extend(pads)


class Preprocessor:
    def __init__(self, config_file_path: str):
        with open(config_file_path) as fp:
            config = json.load(fp)
        self.max_length = config['max_position_embeddings']
        self.id_to_pos = ['[PAD]', '前文字と同じ単語内'] + config['parts_of_speech']
        self.pos_to_id = {pos: i for i, pos in enumerate(self.id_to_pos)}
        self.chr_ids = {c: i + 3 for i, c in enumerate(config['chars'])}

    def _to_chr_ids(self, text: str):
        _chr_ids = []
        for c in text:
            if c in self.chr_ids:
                _chr_ids.append(self.chr_ids[c])
                continue
            _c = ord(c)
            if int('4E00', 16) <= _c <= int('9FFF', 16) or int('3400', 16) <= _c <= int('4DBF', 16):
                _chr_ids.append(UNKNOWN_KANJI_ID)
            else:
                _chr_ids.append(UNKNOWN_NOKANJI_ID)
        return _chr_ids

    def encode_estimation_input(self, text: str, include_metadata: bool=False):
        text = text[:self.max_length]
        token_ids = self._to_chr_ids(text)
        token_type_ids = [0] * len(token_ids)
        encoded = {'input_ids': token_ids}
        if include_metadata:
            encoded['attention_mask'] = [1] * len(token_ids)
            encoded['token_type_ids'] = [0] * len(token_ids)
        return encoded

    def encode_training_input(self, fugashi_tagger, text: str, include_metadata: bool=False, do_padding: bool=False):
        text = text[:self.max_length]
        _input = self.encode_estimation_input(text, include_metadata)

        idx = 0
        pos_ids = []
        for word in fugashi_tagger(text):
            _idx = text.find(word.surface, idx)
            for _i in range(idx, _idx):
                pos_ids.append(self.pos_to_id['[PAD]'])
            pos_ids.append(self.pos_to_id[word.pos.split(',')[0]])
            for _i in range(1, len(word.surface)):
                pos_ids.append(self.pos_to_id['前文字と同じ単語内'])
            idx = _idx + len(word.surface)
        _input['labels'] = pos_ids

        if do_padding:
            pad(_input['input_ids'], 0, self.max_length)
            pad(_input['labels'], 0, self.max_length)
            if include_metadata:
                pad(_input['attention_mask'], 0, self.max_length)
                _input['token_type_ids'] = self.token_type_ids

        return _input

    def decode(self, text, pos_ids):
        tokens = []
        word = []
        pos = ''
        for char, pos_id in zip(text, pos_ids):
            if pos_id == self.pos_to_id['[PAD]']:
                if pos:
                    tokens.append((''.join(word), pos))
                    pos = ''
                    word = []
                tokens.append((' ', '[PAD]'))
            elif pos_id == self.pos_to_id['前文字と同じ単語内']:
                word.append(char)
            else:
                if pos:
                    tokens.append((''.join(word), pos))
                word = [char]
                pos = self.id_to_pos[pos_id]
        if pos:
            tokens.append((''.join(word), pos))
        return tokens
