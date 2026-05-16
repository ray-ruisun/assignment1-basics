import os
import re
import json
from typing import BinaryIO
from collections import Counter
from collections.abc import Iterable, Iterator

import regex


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    根据 special token 找到安全的 chunk 边界。

    这个函数用于把大文件切成多个 chunk，方便后续并行做 pre-tokenization。

    为什么不能随便切？
    ----------------
    假设文件内容是：

        Doc1<|endoftext|>Doc2<|endoftext|>Doc3

    如果随便在中间切，可能得到：

        chunk1: Doc1<|endof
        chunk2: text|>Doc2...

    这样 special token 被切坏，后面就没法正确识别 document boundary。

    更好的做法是让 chunk 边界落在 special token 的起始位置，例如：

        chunk1: Doc1
        chunk2: <|endoftext|>Doc2
        chunk3: <|endoftext|>Doc3

    注意：
    ----
    这个函数返回的是 byte offset，不是字符串 index。

    返回例子：
    --------
    [0, 1024, 4096, 9000]

    表示文件被切成：

        [0, 1024)
        [1024, 4096)
        [4096, 9000)
    """

    assert isinstance(split_special_token, bytes), "split_special_token 必须是 bytes"

    # 1. 读取文件总大小，单位是 byte。
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if desired_num_chunks <= 1:
        return [0, file_size]

    # 2. 先粗略平均切分。
    #
    # 例如 file_size = 10000, desired_num_chunks = 4
    # chunk_size = 2500
    #
    # 初始边界：
    # [0, 2500, 5000, 7500, 10000]
    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]

    # 最后一个边界必须是文件末尾。
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    # 3. 调整中间边界。
    #
    # 目标：
    #   每个中间边界向后移动，直到找到 split_special_token。
    #
    # 例如初始边界在 2500，但最近的 <|endoftext|> 在 2710，
    # 那么边界从 2500 移到 2710。
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 到达文件结尾还没找到 special token，
            # 就把这个边界放到文件末尾。
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break

            initial_position += mini_chunk_size

    # 去重并排序。
    # 有时候多个边界会移动到同一个位置。
    return sorted(set(chunk_boundaries))


class BPETrainer:
    """
    负责训练 byte-level BPE tokenizer。

    训练输出：
    --------
    vocab:
        dict[int, bytes]

        例子：
            {
                0: b"\\x00",
                ...
                97: b"a",
                98: b"b",
                ...
                255: b"\\xff",
                256: b"<|endoftext|>",
                257: b"st",
                258: b"est",
            }

    merges:
        list[tuple[bytes, bytes]]

        例子：
            [
                (b"s", b"t"),
                (b"e", b"st"),
                (b"o", b"w"),
            ]

        含义：
            第 1 次 merge: b"s" + b"t" -> b"st"
            第 2 次 merge: b"e" + b"st" -> b"est"
            第 3 次 merge: b"o" + b"w" -> b"ow"
    """

    # GPT-2 风格 pre-tokenization regex。
    #
    # 它大致会把文本切成：
    #   1. 英文缩写后缀：'s, 't, 're, 've, 'm, 'll, 'd
    #   2. 可选前导空格 + 字母串
    #   3. 可选前导空格 + 数字串
    #   4. 可选前导空格 + 标点符号
    #   5. 空白
    #
    # 例子：
    #   "Hello, world!"
    #
    # 可能切成：
    #   "Hello"
    #   ","
    #   " world"
    #   "!"
    #
    # 注意 " world" 前面带空格。
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.vocab: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []

    def pretokenize_chunk(
        self,
        chunk: str,
        special_tokens: list[str],
    ) -> Counter[tuple[bytes, ...]]:
        """
        对一个 chunk 做 pre-tokenization，并统计每个 pre-token 出现次数。

        输入例子：
        --------
        chunk:

            "low low<|endoftext|>newest"

        special_tokens:

            ["<|endoftext|>"]

        Step 1: 按 special token 切开
        ----------------------------
        parts:

            ["low low", "newest"]

        为什么要切开？
        ------------
        因为 special token 是硬边界，不能让 BPE merge 跨过它。

        也就是说，不允许把：

            "low" 的最后一个 byte

        和：

            "newest" 的第一个 byte

        合并到一起。

        Step 2: regex pre-tokenize
        --------------------------
        对 "low low" 可能得到：

            "low"
            " low"

        注意第二个 token 可能带前导空格。

        Step 3: 转 UTF-8 bytes
        ----------------------
        "low" -> b"low"

        Step 4: 每个 byte 作为初始 token
        --------------------------------
        b"low" -> (b"l", b"o", b"w")

        如果是中文：

            "你" -> b"\\xe4\\xbd\\xa0"
            -> (b"\\xe4", b"\\xbd", b"\\xa0")

        返回：
        ----
        Counter({
            (b"l", b"o", b"w"): 1,
            (b" ", b"l", b"o", b"w"): 1,
            ...
        })
        """

        counter: Counter[tuple[bytes, ...]] = Counter()

        if special_tokens:
            special_pattern = "|".join(
                re.escape(token)
                for token in special_tokens
            )
            parts = re.split(special_pattern, chunk)
        else:
            parts = [chunk]

        for part in parts:
            for match in regex.finditer(self.PAT, part):
                pretoken = match.group(0)

                # 字符串 -> UTF-8 bytes
                token_bytes = pretoken.encode("utf-8")

                # bytes -> tuple of single-byte bytes tokens
                #
                # 例子：
                #   b"low"
                #   -> (b"l", b"o", b"w")
                #
                # 注意：
                #   bytes([108]) -> b"l"
                #   bytes([111]) -> b"o"
                #   bytes([119]) -> b"w"
                token_tuple = tuple(
                    bytes([byte_value])
                    for byte_value in token_bytes
                )

                if token_tuple:
                    counter[token_tuple] += 1

        return counter

    @staticmethod
    def get_pair_counts(
        word_freq: Counter[tuple[bytes, ...]],
    ) -> Counter[tuple[bytes, bytes]]:
        """
        统计所有相邻 pair 的加权出现次数。

        输入例子：
        --------
        word_freq:

            {
                (b"l", b"o", b"w"): 5,
                (b"l", b"o", b"w", b"e", b"r"): 2,
            }

        含义：
        ----
            low 出现 5 次
            lower 出现 2 次

        统计过程：
        --------
        对 (b"l", b"o", b"w"): 5

            (b"l", b"o") += 5
            (b"o", b"w") += 5

        对 (b"l", b"o", b"w", b"e", b"r"): 2

            (b"l", b"o") += 2
            (b"o", b"w") += 2
            (b"w", b"e") += 2
            (b"e", b"r") += 2

        输出：
        ----
            {
                (b"l", b"o"): 7,
                (b"o", b"w"): 7,
                (b"w", b"e"): 2,
                (b"e", b"r"): 2,
            }
        """

        pair_counts: Counter[tuple[bytes, bytes]] = Counter()

        for tokens, freq in word_freq.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq

        return pair_counts

    @staticmethod
    def merge_token(
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        在一个 token 序列里，把指定 pair 的所有非重叠出现合并。

        例子 1：
        -------
        tokens:

            (b"w", b"i", b"d", b"e", b"s", b"t")

        pair:

            (b"s", b"t")

        输出：

            (b"w", b"i", b"d", b"e", b"st")

        例子 2：
        -------
        tokens:

            (b"l", b"o", b"w")

        pair:

            (b"l", b"o")

        输出：

            (b"lo", b"w")

        例子 3：重叠情况
        ---------------
        tokens:

            (b"a", b"a", b"a")

        pair:

            (b"a", b"a")

        从左到右非重叠合并：

            第 0 个 a + 第 1 个 a -> aa
            第 2 个 a 保留

        输出：

            (b"aa", b"a")

        为什么不能输出 (b"a", b"aa") 或者其他？
        --------------------------------------
        BPE merge 是从左到右处理非重叠 pair。
        """

        new_tokens: list[bytes] = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)

    def merge_all(
        self,
        word_freq: Counter[tuple[bytes, ...]],
        best_pair: tuple[bytes, bytes],
    ) -> Counter[tuple[bytes, ...]]:
        """
        把一个 merge 应用到整个 word_freq 上。

        输入：
        ----
        word_freq:

            {
                (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
                (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
            }

        best_pair:

            (b"s", b"t")

        输出：
        ----
            {
                (b"w", b"i", b"d", b"e", b"st"): 3,
                (b"n", b"e", b"w", b"e", b"st"): 6,
            }

        为什么用 Counter += freq？
        -------------------------
        因为两个不同的旧 token 序列 merge 后可能变成同一个新序列。
        这时频率要相加，不能覆盖。
        """

        new_word_freq: Counter[tuple[bytes, ...]] = Counter()

        for tokens, freq in word_freq.items():
            new_tokens = self.merge_token(tokens, best_pair)
            new_word_freq[new_tokens] += freq

        return new_word_freq

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        num_chunks: int = 4,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
        训练 byte-level BPE tokenizer。

        训练 train:

        文本文件
          ↓
        按 special token 找 chunk boundaries
          ↓
        每个 chunk:
            按 special token 切开
            regex pre-tokenize
            pre-token 转 UTF-8 bytes
            每个 byte 变成初始 token
            Counter 统计 pre-token 频率
          ↓
        合并所有 chunk Counter
          ↓
        BPE merge loop:
            统计 pair frequency
            选最高频 pair
            tie-break 选字典序最大
            merge pair
            新 token 加入 vocab
          ↓
        返回 vocab, merges

        参数：
        ----
        input_path:
            训练文件路径。

        vocab_size:
            最终 vocab 大小，包括：
                1. 256 个基础 byte token
                2. special tokens
                3. BPE merge 生成的新 token

        special_tokens:
            例如：
                ["<|endoftext|>"]

        num_chunks:
            切成多少 chunk。
            当前代码是串行处理 chunk，但结构方便之后改 multiprocessing。

        返回：
        ----
        vocab, merges
        """

        special_tokens = special_tokens or []

        # ============================================================
        # Step 1. 初始化 256 byte vocab
        # ============================================================
        #
        # byte-level tokenizer 固定保留 256 个 byte token：
        #
        #   0 -> b"\x00"
        #   1 -> b"\x01"
        #   ...
        #   97 -> b"a"
        #   ...
        #   255 -> b"\xff"
        #
        # 即使某个 byte 在训练集中没有出现，也要保留。
        # 这样未来任何 UTF-8 文本都可以被表示，不会 OOV。
        vocab: dict[int, bytes] = {
            i: bytes([i])
            for i in range(256)
        }

        next_id = 256

        # ============================================================
        # Step 2. 加入 special tokens
        # ============================================================
        #
        # 例子：
        #   special_tokens = ["<|endoftext|>"]
        #
        # 则：
        #   vocab[256] = b"<|endoftext|>"
        #   next_id = 257
        #
        # special token 加入 vocab，但不参与 merge 统计。
        for token in special_tokens:
            vocab[next_id] = token.encode("utf-8")
            next_id += 1

        # ============================================================
        # Step 3. 计算 BPE merge 次数
        # ============================================================
        #
        # 例如：
        #   vocab_size = 300
        #   special_tokens = ["<|endoftext|>"]
        #
        # 已有 vocab:
        #   256 byte tokens + 1 special token = 257
        #
        # 所以还能做：
        #   300 - 257 = 43 次 merge
        num_merges = vocab_size - len(vocab)

        if num_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} 太小，至少需要 {len(vocab)}。"
            )

        # ============================================================
        # Step 4. 读取文件并统计全局 pre-token 频率
        # ============================================================
        #
        # 重要：
        #   每个 chunk 只做 pre-tokenization 和 Counter。
        #   不能在 chunk 里单独训练 BPE。
        #
        # 因为 BPE merge 必须基于全局 pair frequency。
        global_word_freq: Counter[tuple[bytes, ...]] = Counter()

        with open(input_path, "rb") as file:
            if special_tokens:
                split_token = special_tokens[0].encode("utf-8")
                boundaries = find_chunk_boundaries(
                    file=file,
                    desired_num_chunks=num_chunks,
                    split_special_token=split_token,
                )
            else:
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                boundaries = [0, file_size]

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                file.seek(start)
                chunk_bytes = file.read(end - start)

                # 理想情况下 chunk boundary 在 special token 处，
                # 不会切坏 UTF-8。
                #
                # errors="ignore" 是保险处理。
                chunk = chunk_bytes.decode("utf-8", errors="ignore")

                local_counter = self.pretokenize_chunk(
                    chunk=chunk,
                    special_tokens=special_tokens,
                )

                global_word_freq.update(local_counter)

        # ============================================================
        # Step 5. 串行训练 BPE merges
        # ============================================================
        #
        # 为什么不能并行 merge？
        # --------------------
        # 第 2 次 merge 依赖第 1 次 merge 后的新 token 序列。
        # 第 3 次 merge 依赖第 2 次 merge 后的新 token 序列。
        #
        # 所以 merge loop 是顺序依赖的。
        merges: list[tuple[bytes, bytes]] = []
        word_freq = global_word_freq

        for _ in range(num_merges):
            pair_counts = self.get_pair_counts(word_freq)

            if not pair_counts:
                break

            max_count = max(pair_counts.values())

            max_pairs = [
                pair
                for pair, count in pair_counts.items()
                if count == max_count
            ]

            # tie-break:
            # 如果多个 pair 出现次数相同，选字典序最大的。
            #
            # 例如：
            #   (b"e", b"s") 和 (b"s", b"t") 都出现 9 次
            #
            # 因为：
            #   (b"s", b"t") > (b"e", b"s")
            #
            # 所以选：
            #   (b"s", b"t")
            best_pair = max(max_pairs)

            merges.append(best_pair)

            # 新 token 加入 vocab。
            #
            # 例如：
            #   best_pair = (b"e", b"st")
            #   new_token = b"est"
            #   vocab[next_id] = b"est"
            vocab[next_id] = best_pair[0] + best_pair[1]
            next_id += 1

            # 把 best_pair 应用到所有 pre-token 序列。
            word_freq = self.merge_all(word_freq, best_pair)

        self.vocab = vocab
        self.merges = merges

        return vocab, merges

    def save(self, vocab_path: str, merges_path: str) -> None:
        """
        保存 vocab 和 merges 到 JSON 文件。
        """
        self.save_vocab(vocab_path)
        self.save_merges(merges_path)

    def save_vocab(self, vocab_path: str) -> None:
        """
        保存 vocab。

        内存里：
            {
                97: b"a",
                256: b"st"
            }

        JSON 不支持 bytes，所以保存成：

            {
                "97": [97],
                "256": [115, 116]
            }

        解释：
            b"a"  -> [97]
            b"st" -> [115, 116]
        """

        raw_vocab = {
            str(token_id): list(token_bytes)
            for token_id, token_bytes in self.vocab.items()
        }

        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(raw_vocab, file, ensure_ascii=False, indent=2)

    def save_merges(self, merges_path: str) -> None:
        """
        保存 merges。

        内存里：
            [
                (b"s", b"t"),
                (b"e", b"st")
            ]

        JSON 里：
            [
                [[115], [116]],
                [[101], [115, 116]]
            ]
        """

        raw_merges = [
            [list(left), list(right)]
            for left, right in self.merges
        ]

        with open(merges_path, "w", encoding="utf-8") as file:
            json.dump(raw_merges, file, ensure_ascii=False, indent=2)


class BPETokenizer:
    """
    已训练好的 BPE tokenizer。

    它接收：
        vocab:
            id -> bytes

        merges:
            BPE merge list

        special_tokens:
            例如 ["<|endoftext|>"]

    提供：
        encode(text) -> list[int]
        decode(ids) -> str
    """

    PAT = BPETrainer.PAT

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # 反向 vocab：
        #
        # 原始 vocab:
        #   97 -> b"a"
        #   256 -> b"st"
        #
        # token_to_id:
        #   b"a" -> 97
        #   b"st" -> 256
        #
        # encode 时需要从 bytes token 查 id，所以必须有这个。
        self.token_to_id: dict[bytes, int] = {
            token_bytes: token_id
            for token_id, token_bytes in vocab.items()
        }

        # merge rank：
        #
        # merges:
        #   [
        #       (b"s", b"t"),
        #       (b"e", b"st"),
        #       (b"o", b"w"),
        #   ]
        #
        # merge_ranks:
        #   {
        #       (b"s", b"t"): 0,
        #       (b"e", b"st"): 1,
        #       (b"o", b"w"): 2,
        #   }
        #
        # encode 时，如果当前有多个 pair 都能 merge，
        # 要选 rank 最小的，也就是训练时最早学到的 merge。
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank
            for rank, pair in enumerate(merges)
        }

        # special token -> id
        #
        # 例子：
        #   "<|endoftext|>" -> 256
        self.special_token_to_id: dict[str, int] = {}

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                raise ValueError(
                    f"special token {token!r} 不在 vocab 中。"
                )

            self.special_token_to_id[token] = self.token_to_id[token_bytes]

    @classmethod
    def from_file(
        cls,
        vocab_path: str,
        merges_path: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """
        从 vocab.json 和 merges.json 加载 tokenizer。

        vocab.json 例子：
        ----------------
        {
          "0": [0],
          "97": [97],
          "256": [115, 116]
        }

        读取后：
        -------
        {
            0: b"\\x00",
            97: b"a",
            256: b"st"
        }

        merges.json 例子：
        -----------------
        [
          [[115], [116]],
          [[101], [115, 116]]
        ]

        读取后：
        -------
        [
            (b"s", b"t"),
            (b"e", b"st")
        ]
        """

        vocab = cls.load_vocab(vocab_path)
        merges = cls.load_merges(merges_path)

        return cls(
            vocab=vocab,
            merges=merges,
            special_tokens=special_tokens,
        )

    @staticmethod
    def load_vocab(vocab_path: str) -> dict[int, bytes]:
        """
        加载 vocab.json。
        """

        with open(vocab_path, "r", encoding="utf-8") as file:
            raw_vocab = json.load(file)

        vocab: dict[int, bytes] = {}

        for token_id_str, byte_list in raw_vocab.items():
            token_id = int(token_id_str)
            token_bytes = bytes(byte_list)
            vocab[token_id] = token_bytes

        return vocab

    @staticmethod
    def load_merges(merges_path: str) -> list[tuple[bytes, bytes]]:
        """
        加载 merges.json。
        """

        with open(merges_path, "r", encoding="utf-8") as file:
            raw_merges = json.load(file)

        merges: list[tuple[bytes, bytes]] = []

        for left_byte_list, right_byte_list in raw_merges:
            left = bytes(left_byte_list)
            right = bytes(right_byte_list)
            merges.append((left, right))

        return merges

    @staticmethod
    def merge_token(
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        和训练阶段一样：把 tokens 里的指定 pair 合并。

        例子：
        ----
        tokens:
            (b"l", b"o", b"w")

        pair:
            (b"l", b"o")

        输出：
            (b"lo", b"w")
        """

        new_tokens: list[bytes] = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)

    def apply_bpe_merges(
        self,
        token_tuple: tuple[bytes, ...],
    ) -> tuple[bytes, ...]:
        """
        对一个 pre-token 的 byte token 序列应用 BPE merges。

        输入例子：
        --------
        token_tuple:

            (b"l", b"o", b"w")

        假设训练得到的 merges 是：

            [
                (b"l", b"o"),
                (b"lo", b"w"),
            ]

        那么 merge_ranks 是：

            {
                (b"l", b"o"): 0,
                (b"lo", b"w"): 1,
            }

        Step 1:
            当前 pairs:
                (b"l", b"o")
                (b"o", b"w")

            可 merge 的 pair:
                (b"l", b"o")

            合并后：
                (b"lo", b"w")

        Step 2:
            当前 pairs:
                (b"lo", b"w")

            可 merge 的 pair:
                (b"lo", b"w")

            合并后：
                (b"low",)

        输出：
            (b"low",)

        为什么用 rank 最小的 pair？
        -------------------------
        BPE encode 必须按照训练时 merges 的优先级来做。
        越早训练出来的 merge，rank 越小，优先级越高。
        """

        tokens = token_tuple

        while True:
            if len(tokens) < 2:
                break

            pairs = [
                (tokens[i], tokens[i + 1])
                for i in range(len(tokens) - 1)
            ]

            candidate_pairs = [
                pair
                for pair in pairs
                if pair in self.merge_ranks
            ]

            if not candidate_pairs:
                break

            best_pair = min(
                candidate_pairs,
                key=lambda pair: self.merge_ranks[pair],
            )

            tokens = self.merge_token(tokens, best_pair)

        return tokens

    def encode_ordinary(self, text: str) -> list[int]:
        """
        编码普通文本，不处理 special tokens。

        输入例子：
        --------
        text:
            "low"

        Step 1: regex pre-tokenize
        --------------------------
            "low"

        Step 2: UTF-8 bytes
        -------------------
            b"low"

        Step 3: 初始 byte tokens
        ------------------------
            (b"l", b"o", b"w")

        Step 4: BPE merge
        -----------------
        假设 merges 包含：
            (b"l", b"o") -> b"lo"
            (b"lo", b"w") -> b"low"

        则：
            (b"l", b"o", b"w")
            -> (b"lo", b"w")
            -> (b"low",)

        Step 5: 查 token id
        -------------------
        假设：
            token_to_id[b"low"] = 300

        输出：
            [300]
        """

        ids: list[int] = []

        for match in regex.finditer(self.PAT, text):
            pretoken = match.group(0)

            token_bytes = pretoken.encode("utf-8")

            token_tuple = tuple(
                bytes([byte_value])
                for byte_value in token_bytes
            )

            merged_tokens = self.apply_bpe_merges(token_tuple)

            for token in merged_tokens:
                ids.append(self.token_to_id[token])

        return ids

    def encode(self, text: str) -> list[int]:
        """
        编码文本，包括 special tokens。
        编码 encode:

        输入 text
          ↓
        按 special token split，并保留 special token
          ↓
        普通文本:
            regex pre-tokenize
            UTF-8 bytes
            初始 byte tokens
            按 merge_ranks 应用 BPE merges
            bytes tokens -> ids
          ↓
        special token:
            直接 token -> id
          ↓
        返回 list[int]

        重要：
        ----
        不要用：

            text.replace("<|endoftext|>", "256")

        因为这会把 special token 变成普通字符 "2", "5", "6"，
        最后被当成普通文本编码。

        正确做法：
        --------
        先用 regex split，并且保留 special token。

        例子：
        ----
        text:
            "hello<|endoftext|>world"

        special_tokens:
            ["<|endoftext|>"]

        split 后：
            ["hello", "<|endoftext|>", "world"]

        然后：
            "hello"          -> 普通 BPE encode
            "<|endoftext|>" -> 直接 append special token id
            "world"          -> 普通 BPE encode
        """

        if not self.special_tokens:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(
            re.escape(token)
            for token in self.special_tokens
        ) + ")"

        parts = re.split(special_pattern, text)

        ids: list[int] = []

        for part in parts:
            if part == "":
                continue

            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids

    def encode_iterable(
        self,
        text_iterable: Iterable[str],
    ) -> Iterator[list[int]]:
        """
        批量 encode 多个字符串。

        为什么返回 Iterator？
        --------------------
        如果输入有 100 万行文本，直接全部 encode 到一个 list 会很占内存。

        用 yield 可以一条一条返回。

        使用例子：
        --------
        for ids in tokenizer.encode_iterable(["hello", "world"]):
            print(ids)
        """

        for text in text_iterable:
            yield self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        把 token ids 解码回字符串。

        解码 decode:

        token ids
          ↓
        ids -> bytes tokens
          ↓
        拼接所有 bytes
          ↓
        整体 UTF-8 decode
          ↓
        返回字符串

        输入例子：
        --------
        token_ids:
            [300, 256, 301]

        假设：
            vocab[300] = b"hello"
            vocab[256] = b"<|endoftext|>"
            vocab[301] = b"world"

        Step 1: ids -> bytes
        --------------------
            [b"hello", b"<|endoftext|>", b"world"]

        Step 2: 拼接 bytes
        ------------------
            b"hello<|endoftext|>world"

        Step 3: UTF-8 decode
        --------------------
            "hello<|endoftext|>world"

        注意：
        ----
        如果某些 byte 序列不是合法 UTF-8，可以用 errors="replace"。
        """

        byte_chunks: list[bytes] = []

        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Unknown token id: {token_id}")

            byte_chunks.append(self.vocab[token_id])

        text_bytes = b"".join(byte_chunks)

        return text_bytes.decode("utf-8", errors="replace")