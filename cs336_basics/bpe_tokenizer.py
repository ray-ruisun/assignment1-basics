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
    根据 special token 找到安全的 chunk 边界。

    为什么需要这个函数？
    -------------------
    在训练 BPE tokenizer 时，pre-tokenization 可能很慢，
    所以我们希望把一个大文件切成多个 chunk，然后交给多个 process 并行处理。

    但是不能随便在任意 byte 位置切分，因为：
    1. 可能把一个 UTF-8 字符切坏；
    2. 可能把一个文档切成两半；
    3. BPE 训练时不希望跨 document boundary 合并。

    所以这个函数会尽量把 chunk 边界放在 split_special_token 的开头。
    例如 split_special_token = b"<|endoftext|>"。
    这样每个 chunk 都大致是若干完整 document 的集合。

    参数：
    ----
    file:
        以二进制方式打开的文件对象，比如 open(path, "rb")。

    desired_num_chunks:
        希望切成几个 chunk。
        注意：最终 chunk 数量可能少于这个数，因为有些边界可能重合。

    split_special_token:
        用来切分文档的特殊 token，必须是 bytes。
        例如 b"<|endoftext|>"。

    返回：
    ----
    一个 byte offset 列表。
    例如：
        [0, 10240, 20480, 30000]
    表示 chunk 是：
        [0, 10240)
        [10240, 20480)
        [20480, 30000)
    """

    assert isinstance(split_special_token, bytes), "special token 必须是 bytes 类型"

    # 1. 获取整个文件大小，单位是 bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 2. 先粗略地平均切分
    #    例如 file_size = 10000, desired_num_chunks = 4
    #    chunk_size = 2500
    chunk_size = file_size // desired_num_chunks

    # 3. 初始边界：
    #    [0, 2500, 5000, 7500, 10000]
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]

    # 最后一个边界必须是文件结尾
    chunk_boundaries[-1] = file_size

    # 每次向前搜索 4096 bytes，直到找到 special token
    mini_chunk_size = 4096

    # 第一个边界 0 和最后一个边界 file_size 不用调整
    # 只调整中间边界
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]

        file.seek(initial_position)

        while True:
            mini_chunk = file.read(mini_chunk_size)

            # 如果已经到文件结尾，说明后面找不到 special token
            # 那就把这个边界放到文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在当前 mini chunk 中找 special token
            found_at = mini_chunk.find(split_special_token)

            if found_at != -1:
                # 如果找到了，就把边界移动到 special token 的起始位置
                chunk_boundaries[bi] = initial_position + found_at
                break

            # 如果当前 4096 bytes 没找到，就继续往后找
            initial_position += mini_chunk_size

    # 去重并排序
    # 可能多个边界都被移动到了同一个 special token 位置
    return sorted(set(chunk_boundaries))


class BPETokenizer:
    """
    一个用于训练 byte-level BPE tokenizer 的简化实现。

    训练目标：
    --------
    输入：
        一个文本文件
        vocab_size
        special_tokens

    输出：
        vocab:
            dict[int, bytes]
            token id -> token bytes

        merges:
            list[tuple[bytes, bytes]]
            BPE merge 规则列表，按训练顺序排列

    BPE 核心思想：
    ------------
    初始 vocab 是 256 个 byte：
        0 -> b'\\x00'
        1 -> b'\\x01'
        ...
        97 -> b'a'
        ...
        255 -> b'\\xff'

    然后反复做：
        1. 统计所有相邻 token pair 的频率
        2. 找最高频 pair
        3. 如果并列，选字典序更大的 pair
        4. 合并这个 pair
        5. 把合并结果加入 vocab
    """

    # GPT-2 风格 pre-tokenization regex
    #
    # 它会大致匹配：
    # 1. 英文缩写后缀：'s, 't, 're, 've, 'm, 'll, 'd
    # 2. 可选前导空格 + 字母串
    # 3. 可选前导空格 + 数字串
    # 4. 可选前导空格 + 标点/符号串
    # 5. 空白
    #
    # 注意：
    # Python 内置 re 不支持 \\p{L} 这种 Unicode property，
    # 所以这里要用第三方 regex 包。
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self):
        self.merges = []
        self.vocab = {}

    def pretokenize_chunk(self, chunk: str, special_tokens: list[str]) -> Counter:
        """
        对一个 chunk 做 pre-tokenization，并统计每个 pre-token 的频率。

        这一步做什么？
        -------------
        chunk 是一段文本，例如：

            "low low lower<|endoftext|>newest"

        如果 special_tokens = ["<|endoftext|>"]，
        那么先按 special token 切开：

            ["low low lower", "newest"]

        这样做的原因是：
            special token 是 hard boundary，
            不允许 BPE merge 跨过它。
            同时 special token 自己不参与 merge 统计。

        然后对每一段普通文本用 regex pre-tokenize。

        例如：
            "low low lower"

        可能得到：
            "low"
            " low"
            " lower"

        然后把每个 pre-token 编码成 UTF-8 bytes。

        例如：
            "low".encode("utf-8") = b"low"

        再把每个 byte 变成单独的 bytes token：

            b"low"
            -> bytes 值是 [108, 111, 119]
            -> token tuple 是 (b"l", b"o", b"w")

        最后统计频率：

            Counter({
                (b"l", b"o", b"w"): 1,
                (b" ", b"l", b"o", b"w"): 1,
                ...
            })

        参数：
        ----
        chunk:
            当前 chunk 的字符串内容。

        special_tokens:
            需要作为硬边界的特殊 token 列表。

        返回：
        ----
        Counter:
            key 是 tuple[bytes, ...]
            value 是这个 pre-token 出现的次数。
        """

        counter = Counter()

        # 1. 先按 special tokens 切分
        #    注意这里用 re.escape，避免 special token 里的 | 等符号被当成 regex 语法
        if special_tokens:
            special_pattern = "|".join(re.escape(token) for token in special_tokens)
            parts = re.split(special_pattern, chunk)
        else:
            parts = [chunk]

        # 2. 对每个普通文本片段做 regex pre-tokenization
        for part in parts:
            for match in regex.finditer(self.PAT, part):
                token = match.group(0)

                # 3. 转成 UTF-8 bytes
                #    例如：
                #        "你" -> b"\\xe4\\xbd\\xa0"
                token_bytes = token.encode("utf-8")

                # 4. 每一个 byte 作为一个初始 token
                #    例如：
                #        b"low" -> (b"l", b"o", b"w")
                #        b"\\xe4\\xbd\\xa0" -> (b"\\xe4", b"\\xbd", b"\\xa0")
                token_tuple = tuple(bytes([b]) for b in token_bytes)

                # 5. 统计这个 pre-token 出现次数
                if token_tuple:
                    counter[token_tuple] += 1

        return counter

    def get_pair_counts(self, freq_key_b: Counter) -> Counter:
        """
        统计所有相邻 token pair 的加权频率。

        输入是什么？
        ----------
        freq_key_b 是 pre-token 频率表。

        例如：
            {
                (b"l", b"o", b"w"): 5,
                (b"l", b"o", b"w", b"e", b"r"): 2,
            }

        这表示：
            "low" 出现 5 次
            "lower" 出现 2 次

        那么相邻 pair 统计是：

        对 (b"l", b"o", b"w"): 5
            (b"l", b"o") += 5
            (b"o", b"w") += 5

        对 (b"l", b"o", b"w", b"e", b"r"): 2
            (b"l", b"o") += 2
            (b"o", b"w") += 2
            (b"w", b"e") += 2
            (b"e", b"r") += 2

        最终：
            {
                (b"l", b"o"): 7,
                (b"o", b"w"): 7,
                (b"w", b"e"): 2,
                (b"e", b"r"): 2,
            }

        返回：
        ----
        Counter:
            key 是 pair，即 tuple[bytes, bytes]
            value 是这个 pair 的加权出现次数。
        """

        count_dict = Counter()

        for tokens, freq in freq_key_b.items():
            # tokens 是一个 tuple，例如 (b"l", b"o", b"w")
            # freq 是这个 pre-token 出现次数，例如 5
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                count_dict[pair] += freq

        return count_dict

    def merge_token(
        self,
        tokens: tuple[bytes, ...],
        pair: tuple[bytes, bytes],
    ) -> tuple[bytes, ...]:
        """
        在一个 token tuple 里面，把指定 pair 的所有非重叠出现合并。

        例子 1：
        -------
        tokens = (b"w", b"i", b"d", b"e", b"s", b"t")
        pair = (b"s", b"t")

        结果：
            (b"w", b"i", b"d", b"e", b"st")

        例子 2：
        -------
        tokens = (b"l", b"o", b"w")
        pair = (b"l", b"o")

        结果：
            (b"lo", b"w")

        例子 3：为什么 i += 2？
        ----------------------
        tokens = (b"a", b"a", b"a")
        pair = (b"a", b"a")

        从左到右合并非重叠 pair：
            第 0 和第 1 个 a 合并成 aa
            第 2 个 a 保留

        结果：
            (b"aa", b"a")

        不能同时把第 1 和第 2 个 a 再合并，因为会重叠。
        """

        new_tokens = []
        i = 0

        while i < len(tokens):
            # 如果当前位置和下一个位置刚好等于要 merge 的 pair
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                # 合并两个 bytes token
                # 例如 b"e" + b"st" = b"est"
                new_tokens.append(tokens[i] + tokens[i + 1])

                # 跳过两个 token
                i += 2
            else:
                # 否则保留当前 token
                new_tokens.append(tokens[i])
                i += 1

        return tuple(new_tokens)

    def merge_all(
        self,
        freq_key_b: Counter,
        best_pair: tuple[bytes, bytes],
    ) -> Counter:
        """
        把某个 best_pair 应用到所有 pre-token 序列上。

        输入：
        ----
        freq_key_b 例如：
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

        为什么要用 Counter 并 += freq？
        -----------------------------
        因为不同的旧 token 序列，merge 后可能变成同一个新 token 序列。
        这种情况下频率要相加，不能覆盖。
        """

        new_freq_key_b = Counter()

        for tokens, freq in freq_key_b.items():
            new_tokens = self.merge_token(tokens, best_pair)
            new_freq_key_b[new_tokens] += freq

        return new_freq_key_b

    def train(
        self,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_processes: int = 4,
    ):
        """
        训练 byte-level BPE tokenizer。

        参数：
        ----
        input_path:
            训练文本文件路径。

        vocab_size:
            最终词表大小，包括：
                256 个初始 byte tokens
                special tokens
                BPE merge 产生的新 tokens

        special_tokens:
            特殊 token 列表。
            例如：
                ["<|endoftext|>"]

            它们的作用：
                1. 加入 vocab；
                2. 在训练时作为 hard boundary；
                3. 不参与 merge statistics。

        num_processes:
            想切成多少个 chunk。
            当前这版代码还是串行处理 chunk，
            但 chunk 逻辑已经准备好，之后可以接 multiprocessing。

        返回：
        ----
        vocab:
            dict[int, bytes]

        merges:
            list[tuple[bytes, bytes]]
        """

        # ============================================================
        # Step 1. 初始化 byte-level vocabulary
        # ============================================================
        #
        # byte-level BPE 的初始词表是所有可能的 byte 值。
        #
        # 因为：
        #     1 byte = 8 bits
        #     2^8 = 256
        #
        # 所以 byte 的取值范围是：
        #     0, 1, 2, ..., 255
        #
        # vocab[97] = b"a"
        # vocab[98] = b"b"
        # vocab[228] = b"\\xe4"
        #
        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # ============================================================
        # Step 2. 把 special tokens 加入 vocabulary
        # ============================================================
        #
        # 例如：
        #     special_tokens = ["<|endoftext|>"]
        #
        # 那么：
        #     vocab[256] = b"<|endoftext|>"
        #     next_id = 257
        #
        # 注意：
        # special token 加入 vocab，但不参与 BPE merge 统计。
        #
        for tok in special_tokens:
            vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        # ============================================================
        # Step 3. 计算要做多少次 BPE merge
        # ============================================================
        #
        # 最终词表大小 vocab_size 包含三部分：
        #
        #     vocab_size = 256 byte tokens
        #                + len(special_tokens)
        #                + num_merges
        #
        # 所以：
        #
        #     num_merges = vocab_size - 当前已有 vocab 大小
        #
        # 例子：
        #     vocab_size = 300
        #     special_tokens = ["<|endoftext|>"]
        #
        #     初始 vocab = 256 + 1 = 257
        #     num_merges = 300 - 257 = 43
        #
        num_merges = vocab_size - len(vocab)

        if num_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} 太小。"
                f"至少需要 256 + len(special_tokens) = {len(vocab)}。"
            )

        # ============================================================
        # Step 4. 读取文件 chunk，并统计全局 pre-token 频率
        # ============================================================
        #
        # 这一阶段只做 pre-tokenization 和 frequency counting。
        # 不做 BPE merge。
        #
        # 为什么？
        # 因为 BPE merge 必须基于全局 pair frequency。
        # 如果每个 chunk 单独训练 merges，结果会是错的。
        #
        # global_freq_key_b 的形式：
        #
        #     Counter({
        #         (b"l", b"o", b"w"): 5,
        #         (b"l", b"o", b"w", b"e", b"r"): 2,
        #         ...
        #     })
        #
        global_freq_key_b = Counter()

        with open(input_path, "rb") as f:
            if special_tokens:
                # 使用第一个 special token 作为 chunk boundary marker
                split_token = special_tokens[0].encode("utf-8")
                boundaries = find_chunk_boundaries(f, num_processes, split_token)
            else:
                # 如果没有 special token，那就不切 chunk
                f.seek(0, os.SEEK_END)
                file_size = f.tell()
                boundaries = [0, file_size]

            # 遍历所有 chunk
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)

                # 读取当前 chunk 的 bytes
                chunk_bytes = f.read(end - start)

                # 解码为字符串
                # errors="ignore" 表示如果切分处刚好导致某些非法 UTF-8，
                # 就忽略坏 byte。
                # 理想情况下，chunk boundary 在 special token 处，不应该有问题。
                chunk = chunk_bytes.decode("utf-8", errors="ignore")

                # 对当前 chunk 做 pre-tokenization，并统计 local frequency
                local_counter = self.pretokenize_chunk(chunk, special_tokens)

                # 把当前 chunk 的统计合并到全局统计中
                global_freq_key_b.update(local_counter)

        # ============================================================
        # Step 5. 在全局 pre-token 频率上训练 BPE merges
        # ============================================================
        #
        # 注意：
        # 这里是串行的。
        #
        # 因为第 k 次 merge 的结果依赖第 k-1 次 merge 之后的 token 序列。
        # 所以不能让多个 process 各自 merge 再合并。
        #
        merges = []
        freq_key_b = global_freq_key_b

        for _ in range(num_merges):
            # 5.1 统计当前所有相邻 pair 的频率
            count_dict = self.get_pair_counts(freq_key_b)

            # 如果已经没有 pair 可以 merge，停止
            if not count_dict:
                break

            # 5.2 找最大频率
            max_val = max(count_dict.values())

            # 5.3 找出所有频率等于 max_val 的 pair
            max_pairs = [
                pair for pair, count in count_dict.items()
                if count == max_val
            ]

            # 5.4 如果有多个最高频 pair，用字典序选最大的
            #
            # 例如：
            #     (b"e", b"s") 和 (b"s", b"t") 都出现 9 次
            #
            # 因为：
            #     (b"s", b"t") > (b"e", b"s")
            #
            # 所以选择：
            #     (b"s", b"t")
            #
            best_pair = max(max_pairs)

            # 5.5 记录 merge 规则
            #
            # 例如：
            #     merges.append((b"s", b"t"))
            #
            merges.append(best_pair)

            # 5.6 把新 token 加入 vocab
            #
            # 例如：
            #     best_pair = (b"s", b"t")
            #     best_pair[0] + best_pair[1] = b"st"
            #
            #     vocab[257] = b"st"
            #
            vocab[next_id] = best_pair[0] + best_pair[1]
            next_id += 1

            # 5.7 把这个 merge 应用到所有 pre-token 序列
            #
            # 例如：
            #     (b"w", b"i", b"d", b"e", b"s", b"t")
            #     -> (b"w", b"i", b"d", b"e", b"st")
            #
            freq_key_b = self.merge_all(freq_key_b, best_pair)

        # 保存训练结果
        self.vocab = vocab
        self.merges = merges

        return vocab, merges