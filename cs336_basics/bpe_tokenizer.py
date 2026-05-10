
from collections import Counter


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

    def train(self, text, num_merges=None):
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


if __name__ == '__main__':
    text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

    tokenizer = BPETokenizer()

    final_tokens = tokenizer.train(text)

    print(tokenizer.merges)

    print(final_tokens)
    # print(pre_tokens)
    # print(freq)
    # print(freq_key_b)