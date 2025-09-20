import json
import unicodedata as ud
from typing import List, Iterable

class Tokenizer:
    def __init__(self, classes_path: str):
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.tokens: List[str] = json.load(f)
        if not isinstance(self.tokens, list) or not all(isinstance(x, str) for x in self.tokens):
            raise ValueError("classes_vedic.json must be a flat list of strings")

        self.classes = self.tokens
        self.blank_idx = self.tokens.index('<pad_blank>') if '<pad_blank>' in self.tokens else 0
        self.blank_index = self.blank_idx
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.char2idx = self.token_to_id
        self.idx2char = self.id_to_token

    def normalize(self, s: str) -> str:
        return ud.normalize('NFC', s)

    def encode(self, s: str) -> List[int]:
        s = self.normalize(s)
        return [self.token_to_id.get(ch, self.blank_idx) for ch in s]

    def decode_greedy(self, ids: Iterable[int]) -> str:
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i == self.blank_idx:
                prev = i
                continue
            if i != prev:
                out.append(self.id_to_token.get(i, ''))
            prev = i
        return self.normalize(''.join(out))

    def text_to_ids(self, s: str) -> List[int]:
        return self.encode(s)

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return ''.join(self.id_to_token.get(int(i), '') for i in ids)
