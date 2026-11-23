"""Byte-level BPE training and tokenization."""

import os
import pickle
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

import regex as re

# GPT-2 style pre-tokenization pattern: splits raw text into word pieces so that
# pair counting can be parallelized across CPU processes.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# Number of chunks the input file is split into for parallel pre-tokenization.
NUM_CHUNKS = 32


def _initialize_vocabulary(special_tokens: list[str]) -> list[bytes]:
    """Start with the 256 single-byte tokens, then append the special tokens."""
    vocab = [bytes([i]) for i in range(256)]
    for special in special_tokens:
        vocab.append(special.encode("utf-8"))
    return vocab


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Split the file into parts that can be counted independently.

    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes.
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced.
    # Chunks start on the previous index and do not include the last index.
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4 KB at a time.

    # Only adjust the internal boundaries.
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at the boundary guess.
        while True:
            mini_chunk = file.read(mini_chunk_size)

            # At EOF, this boundary should be at the end of the file.
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Ensure all boundaries are unique; may be fewer than desired_num_chunks.
    return sorted(set(chunk_boundaries))


def _process_special_tokens(chunk: str, special_tokens: list[str], special_tokens_remain: bool) -> list[str]:
    """Split a chunk on special tokens, optionally preserving them."""
    if not special_tokens:
        return [chunk]
    escaped_tokens = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    pattern_str = "(" + "|".join(escaped_tokens) + ")" if special_tokens_remain else "|".join(escaped_tokens)
    pattern = re.compile(pattern_str)
    return pattern.split(chunk)


def _pre_decode_tokenization(chunked_tokens: str) -> list[tuple[bytes, ...]]:
    """Convert a special-token-split chunk into a list of byte tuples for decoding."""
    result = []
    for match in re.finditer(PAT, chunked_tokens):
        pre_tokenized_word = match.group()
        encoded_bytes = pre_tokenized_word.encode("utf-8")
        result.append(tuple(bytes([b]) for b in encoded_bytes))
    return result


def _pre_tokenization_helper(chunked_tokens: str) -> dict[tuple[bytes, ...], int]:
    """Pre-tokenize a chunk with the GPT-2 style PAT and count the pieces.

    Uses ``finditer`` instead of ``findall`` to avoid materializing the full
    list of pre-tokenized words in memory.
    """
    counts: dict[tuple[bytes, ...], int] = {}
    for match in re.finditer(PAT, chunked_tokens):
        pre_tokenized_word = match.group()
        encoded_bytes = pre_tokenized_word.encode("utf-8")
        key = tuple(bytes([b]) for b in encoded_bytes)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_byte_chunk(chunk_bytes: bytes, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """Count pre-tokenized pieces in one raw byte chunk.

    Shared by the serial and the multiprocessing pre-tokenization paths so both
    produce exactly the same counts.
    """
    chunk = chunk_bytes.decode("utf-8", errors="ignore")

    # Normalize newlines.
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    chunk_counts: Counter = Counter()

    # Remove special tokens, then count pre-tokenized pieces in each segment.
    for chunked_tokens in _process_special_tokens(chunk, special_tokens, False):
        chunk_counts.update(_pre_tokenization_helper(chunked_tokens))

    return dict(chunk_counts)


def _process_chunk_worker(
    file_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """Pre-tokenize one byte range of the file; runs in a worker process."""
    # Each process opens the file itself and reads only its own slice.
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
    return _count_byte_chunk(chunk_bytes, special_tokens)


# Corpora smaller than this are processed serially. On Windows the cost of
# spawning worker processes dominates, so a process pool would make small-file
# tokenizer training several times slower than a single pass.
_PARALLEL_MIN_BYTES = 8 * 1024 * 1024


def _pre_tokenization(
    file_path: str | os.PathLike,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """Count pre-tokenized pieces across the file, in parallel for large inputs."""
    file_size = os.path.getsize(file_path)
    # The main process only locates the chunk boundaries.
    with open(file_path, "rb") as f:
        split_special_token = b"<|endoftext|>" if len(special_tokens) == 0 else special_tokens[0].encode("utf-8")
        boundaries = _find_chunk_boundaries(f, desired_num_chunks, split_special_token)

    # Serial fast path for small corpora: avoids process-pool startup overhead.
    if file_size < _PARALLEL_MIN_BYTES:
        merged: Counter = Counter()
        with open(file_path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                merged.update(_count_byte_chunk(f.read(end - start), special_tokens))
        return dict(merged)

    tasks = [
        (file_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    workers = min(len(tasks), os.cpu_count() or 1, desired_num_chunks)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_chunk_worker, *task) for task in tasks]

        merged_counts: Counter = Counter()
        for future in futures:
            merged_counts.update(future.result())
    return dict(merged_counts)


def bpe_train_naive(
    counts: dict[tuple[bytes, ...], int],
    vocab: list[bytes],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Naive (O(n) rescan per merge) BPE training, kept for reference/testing."""
    merges = []
    while len(vocab) < vocab_size:
        frequency: dict[tuple[bytes, bytes], int] = {}
        for key, value in counts.items():
            for i in range(len(key) - 1):
                pre = key[i]
                cur = key[i + 1]
                frequency[(pre, cur)] = frequency.get((pre, cur), 0) + value

        if len(frequency) == 0:
            break

        max_freq = max(frequency.values())
        max_key_list = [k for k, v in frequency.items() if v == max_freq]
        max_freq_key = sorted(max_key_list)[-1]

        if max_freq_key is None:
            break

        merged_key = b"".join(max_freq_key)
        vocab.append(merged_key)
        merges.append(max_freq_key)

        new_counts: dict[tuple[bytes, ...], int] = {}
        for key, value in counts.items():
            if len(key) < 2:
                new_counts[key] = value
                continue

            new_key = []
            i = 0
            while i < len(key):
                if i < len(key) - 1 and (key[i], key[i + 1]) == max_freq_key:
                    new_key.append(key[i] + key[i + 1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1
            new_counts[tuple(new_key)] = new_counts.get(tuple(new_key), 0) + value
        counts = new_counts
    vocab = dict(enumerate(vocab))
    return vocab, merges


def bpe_train(
    counts: dict[tuple[bytes, ...], int],
    vocab: list[bytes],
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE merges with incremental pair-frequency updates.

    This avoids the O(n) rescan per merge of the naive version, which is
    necessary to train on large corpora such as OpenWebText.
    """
    start_time = time.perf_counter()
    merges = []

    # Initialize frequency and pair -> words mapping.
    frequency: dict[tuple[bytes, bytes], int] = {}
    pair_to_words: dict[tuple[bytes, bytes], set] = {}
    for key, value in counts.items():
        for i in range(len(key) - 1):
            pre = key[i]
            cur = key[i + 1]
            frequency[(pre, cur)] = frequency.get((pre, cur), 0) + value
            pair_to_words.setdefault((pre, cur), set()).add(key)

    while len(vocab) < vocab_size:
        if not frequency:
            break

        max_freq_key = max(frequency, key=lambda k: (frequency[k], k))
        merged_key = b"".join(max_freq_key)

        vocab.append(merged_key)
        merges.append(max_freq_key)

        elapsed_time = time.perf_counter() - start_time
        if len(vocab) % 1000 == 0:
            print(f"current vocab length: {len(vocab)}")
            print(f"current training time: {elapsed_time:.3f}s\n")

        word_to_update_set = pair_to_words[max_freq_key].copy()
        for word_to_update in word_to_update_set:
            word_counts = counts[word_to_update]

            # Remove the old word's pairs from frequency and pair_to_words.
            for i in range(len(word_to_update) - 1):
                pre = word_to_update[i]
                cur = word_to_update[i + 1]
                frequency[(pre, cur)] -= word_counts
                pair_to_words[(pre, cur)].discard(word_to_update)

                if frequency[(pre, cur)] == 0:
                    del frequency[(pre, cur)]
                    del pair_to_words[(pre, cur)]

            # Construct the merged word.
            new_word = []
            i = 0
            while i < len(word_to_update):
                if i == len(word_to_update) - 1:
                    new_word.append(word_to_update[i])
                    break
                pre = word_to_update[i]
                cur = word_to_update[i + 1]
                if (pre, cur) == max_freq_key:
                    new_word.append(merged_key)
                    i += 2
                else:
                    new_word.append(pre)
                    i += 1
            new_word = tuple(new_word)

            # Register the new word's pairs.
            for i in range(len(new_word) - 1):
                pre = new_word[i]
                cur = new_word[i + 1]
                frequency[(pre, cur)] = frequency.get((pre, cur), 0) + word_counts
                pair_to_words.setdefault((pre, cur), set()).add(new_word)

            # Update the counts.
            del counts[word_to_update]
            counts[new_word] = counts.get(new_word, 0) + word_counts

    vocab = dict(enumerate(vocab))
    return vocab, merges


def run_bpe_train(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Run the full BPE training pipeline and report a timing profile."""
    total_start = time.perf_counter()
    init_start = time.perf_counter()

    print("Start initializing vocabulary:\n")
    vocab = _initialize_vocabulary(special_tokens)
    print("Initializing vocabulary Ends.\n")
    init_time = time.perf_counter() - init_start

    pretok_total_start = time.perf_counter()

    print("Start pre-tokenization:\n")
    counts = _pre_tokenization(input_path, NUM_CHUNKS, special_tokens)
    print("Pre-tokenization Ends.\n")
    pretok_total_time = time.perf_counter() - pretok_total_start

    bpe_start = time.perf_counter()

    print("Start bpe train:\n")
    vocab, merges = bpe_train(counts, vocab, vocab_size)
    print("Bpe train ends.\n")

    bpe_time = time.perf_counter() - bpe_start

    total_time = time.perf_counter() - total_start

    print("\n====== BPE TRAIN PROFILE ======")
    print(f"Vocabulary initialization: {init_time:.3f} s")
    print(f"Total pre-tokenization:    {pretok_total_time:.3f} s")
    print(f"BPE training:              {bpe_time:.3f} s")
    print(f"Total run_bpe_train:       {total_time:.3f} s")
    print("===============================\n")

    return vocab, merges


class Tokenizer:
    """A byte-level BPE tokenizer backed by a vocabulary and an ordered merge list."""

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        # Merges must be applied in rank order.
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text) -> list[int]:
        # Split the raw text on special tokens, preserving them (unlike training).
        split_chunk = _process_special_tokens(text, self.special_tokens, True)
        ids = []
        for chunk in split_chunk:
            if self.special_tokens and chunk in self.special_tokens:
                special_token_id = self.token_to_id[chunk.encode("utf-8")]
                ids.append(special_token_id)
            else:
                for token in _pre_decode_tokenization(chunk):
                    ids += self._encode_token(token)
        return ids

    def _encode_token(self, token) -> list[int]:
        while True:
            pairs = [(token[i], token[i + 1]) for i in range(len(token) - 1)]
            valid_pairs = [pair for pair in pairs if pair in self.merge_rank]
            if len(valid_pairs) == 0:
                break
            merge_pair = min(valid_pairs, key=lambda pair: self.merge_rank[pair])
            new_token = []
            i = 0
            while i < len(token):
                if i == len(token) - 1:
                    new_token.append(token[i])
                    break
                pair = (token[i], token[i + 1])
                if pair == merge_pair:
                    new_token.append(token[i] + token[i + 1])
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            token = tuple(new_token)
        return [self.token_to_id[bytes(t)] for t in token]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decode_bytes = b""
        for id_ in ids:
            decode_bytes += self.vocab[id_]
        return decode_bytes.decode("utf-8", errors="replace")
