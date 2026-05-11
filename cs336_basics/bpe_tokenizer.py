import os
from typing import BinaryIO
from collections import Counter

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
    def __init__(self):
        self.merges = []

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

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str], num_merges=None):
        '''
            input_path: str Path to a text file with BPE tokenizer training data.

            vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).

            special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat
                them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
        '''

        # Read files
        with open(input_path, "rb") as f:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

        return
        tokens = text.split(" ")
        freq = dict(Counter(tokens))

        freq_key_b = {tuple(k): v for k, v in freq.items()}

        self.merges = []

        while True:
            count_dict = self.get_pair_counts(freq_key_b)

            if not count_dict:
                break

            max_val = max(count_dict.values())

            if max_val <= 1:
                break

            max_pairs = [
                pair for pair, count in count_dict.items()
                if count == max_val
            ]

            # 并列时选择字典序更大的 pair
            best_pair = max(max_pairs)

            self.merges.append(best_pair)

            new_freq_key_b = {}

            for tokens, freq in freq_key_b.items():
                new_tokens = self.merge_token(tokens, best_pair)
                new_freq_key_b[new_tokens] = new_freq_key_b.get(new_tokens, 0) + freq

            freq_key_b = new_freq_key_b

            if num_merges is not None and len(self.merges) >= num_merges:
                break

        return freq_key_b

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