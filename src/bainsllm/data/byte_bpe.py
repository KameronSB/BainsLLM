import json
import re
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path


PIECES = re.compile(r"\w+|[^\w\s]+|\s+")
EOD = 256
FIRST_MERGE = 257

def merge_pair(ids, pair, new_id):
    """Replace each non-overlapping occurrence of pair, from left to right."""
    result = []
    i = 0

    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    return result

def count_pairs(ids):
    return Counter(pairwise(ids))

def count_pieces(texts):
    frequencies = Counter()

    for text in texts:
        frequencies.update(
            match.group() for match in PIECES.finditer(text)
        )

    return frequencies

class ByteBPETokenizer:
    eos_id = EOD

    def __init__(self, merges=()):
        self.merges = tuple(tuple(pair) for pair in merges)
        self.merge_ids = {}
        self.token_bytes = [bytes([i]) for i in range(256)] + [b""]

        for left, right in self.merges:
            new_id = len(self.token_bytes)

            if not (0 <= left < new_id and 0 <= right < new_id):
                raise ValueError("A merge refers to an undefined token")

            if EOD in (left, right) or (left, right) in self.merge_ids:
                raise ValueError("Invalid merge rule")

            self.merge_ids[left, right] = new_id
            self.token_bytes.append(
                self.token_bytes[left] + self.token_bytes[right]
            )

    @property
    def vocab_size(self):
        return len(self.token_bytes)

    @classmethod
    def train(cls, texts, vocab_size=2000):
        if vocab_size < FIRST_MERGE:
            raise ValueError("Need 256 bytes and one end-of-document ID")

        frequencies = count_pieces(texts)
        if not frequencies:
            raise ValueError("Training text is empty")

        pieces = [list(piece.encode("utf-8")) for piece in frequencies]
        weights = list(frequencies.values())

        totals = Counter()
        locations = defaultdict(set)

        for i, ids in enumerate(pieces):
            for pair, count in count_pairs(ids).items():
                totals[pair] += weights[i] * count
                locations[pair].add(i)

        merges = []

        for new_id in range(FIRST_MERGE, vocab_size):
            highest = max(totals.values(), default=0)
            if highest < 2:
                break

            winner = min(
                pair for pair, count in totals.items()
                if count == highest
            )
            merges.append(winner)

            for i in list(locations[winner]):
                old = pieces[i]
                new = merge_pair(old, winner, new_id)

                before = count_pairs(old)
                after = count_pairs(new)

                for pair in before.keys() | after.keys():
                    totals[pair] += weights[i] * (after[pair] - before[pair])

                    if after[pair]:
                        locations[pair].add(i)
                    else:
                        locations[pair].discard(i)

                    if totals[pair] == 0:
                        del totals[pair]
                        del locations[pair]

                pieces[i] = new

        return cls(merges)

    def _encode_piece(self, piece):
        ids = list(piece.encode("utf-8"))

        while len(ids) > 1:
            candidates = (
                self.merge_ids[pair]
                for pair in pairwise(ids)
                if pair in self.merge_ids
            )
            new_id = min(candidates, default=None)

            if new_id is None:
                break

            pair = self.merges[new_id - FIRST_MERGE]
            ids = merge_pair(ids, pair, new_id)

        return ids

    def encode(self, text):
        ids = []
        cache = {}

        for match in PIECES.finditer(text):
            piece = match.group()

            if piece not in cache:
                cache[piece] = self._encode_piece(piece)

            ids.extend(cache[piece])

        return ids

    def decode(self, ids, *, errors="strict"):
        raw = bytearray()

        for token_id in ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"Unknown token ID: {token_id}")

            raw.extend(self.token_bytes[token_id])

        return raw.decode("utf-8", errors=errors)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "bains-byte-bpe-v1",
            "merges": self.merges,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        if payload["format"] != "bains-byte-bpe-v1":
            raise ValueError("Unrecognised tokenizer format")

        return cls(payload["merges"])