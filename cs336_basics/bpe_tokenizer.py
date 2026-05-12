import os
import re
from typing import BinaryIO
from collections import Counter

import regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPETokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.merges = []
        self.vocab = {}

    def pretokenize_chunk(self, chunk: str, special_tokens: list[str]) -> Counter:

        counter = Counter()

        if special_tokens:
            special_pattern = "|".join(re.escape(token) for token in special_tokens)
            parts = re.split(special_pattern, chunk)
        else:
            parts = [chunk]

        for part in parts:
            for match in regex.finditer(self.PAT, part):
                token = match.group(0)

                token_bytes = token.encode("utf-8")
                token_tuple = tuple(bytes([b]) for b in token_bytes)

                if token_tuple:
                    counter[token_tuple] += 1

        return counter


    def get_pair_counts(self, freq_key_b):
        count_dict = {}

        for tokens, freq in freq_key_b.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                count_dict[pair] = count_dict.get(pair, 0) + freq

        return count_dict

    def merge_token(self, tokens, pair):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)

    def merge_all(self, freq_key_b: Counter, best_pair: tuple[bytes, bytes]) -> Counter:

        new_freq_key_b = Counter()

        for tokens, freq in freq_key_b.items():
            new_tokens = self.merge_token(tokens, best_pair)
            new_freq_key_b[new_tokens] += freq
        return new_freq_key_b

    def train(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], num_processes: int = 4):
        '''
            input_path: str Path to a text file with BPE tokenizer training data.

            vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

            special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat
                them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
        '''

        # 1. Build initial vocab: 256 byte tokens

        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # 2. Add special tokens to vocab

        for tok in special_tokens:
            vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        # 3. Decide how many merges to do
        num_merges = vocab_size - len(vocab)

        if num_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is too small. "
                f"Need at least 256 + len(special_tokens) = {len(vocab)}."
            )

        # 4. Read chunks and collect global pre-token frequencies

        global_freq_key_b = Counter()
        with open(input_path, "rb") as f:
            if special_tokens:
                split_token = special_tokens[0].encode("utf-8")
                boundaries = find_chunk_boundaries(f, num_processes, split_token)
            else:
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                boundaries = [0, file_size]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk = chunk_bytes.decode("utf-8", errors="ignore")
                local_counter = self.pretokenize_chunk(chunk, special_tokens)
                global_freq_key_b.update(local_counter)
        # 5. Train BPE merges globally, not per chunk
        merges = []
        freq_key_b = global_freq_key_b
        for _ in range(num_merges):
            count_dict = self.get_pair_counts(freq_key_b)
            if not count_dict:
                break
            max_val = max(count_dict.values())
            # 找所有并列最大 pair
            max_pairs = [
                pair for pair, count in count_dict.items()
                if count == max_val
            ]

            # tie-break: lexicographically greater pair
            best_pair = max(max_pairs)
            merges.append(best_pair)

            # Add merged token to vocab

            vocab[next_id] = best_pair[0] + best_pair[1]
            next_id += 1

            # Apply merge to all pre-token sequences

            freq_key_b = self.merge_all(freq_key_b, best_pair)

        self.vocab = vocab
        self.merges = merges

        return vocab, merges


if __name__ == '__main__':
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    data_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    tokenizer = BPETokenizer()

    final_tokens = tokenizer.train(data_path, 256, ["<|endoftext|>"])

    print(tokenizer.merges)

    print(final_tokens)
    # print(pre_tokens)
    # print(freq)
    # print(freq_key_b)