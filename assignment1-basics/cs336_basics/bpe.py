import os
from multiprocessing import Pool
import regex as re
from collections import Counter, defaultdict # 引入 defaultdict
import pickle
from typing import BinaryIO

# --- 以下函数保持不变 ---

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def pre_split_word(args):
    """
    Worker function to read and decode a specific chunk of the file.
    (此函数内代码有微小但重要的修正，以正确处理token)
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore").replace('\r', '')
        if special_tokens:
            pattern = "|".join(re.escape(tok) for tok in special_tokens)
            segments = re.split(pattern, chunk) if pattern else [chunk]
        else:
            segments = [chunk]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = []
        for segment in segments:
            tokens = [match.group(0) for match in re.finditer(PAT, segment)]
            # 修正: 原代码的 tuple(text.encode()) 会将多字节字符错误分割
            # 正确做法是先编码，再将字节序列转为整数元组
            tokens_list = [tuple(text.encode("utf-8")) for text in tokens]
            pre_tokens.extend(tokens_list)

        token_counts = Counter(pre_tokens)
        return dict(token_counts)

def get_byte_pairs_and_counts(token_counts):
    """统计所有相邻字节对的出现频率"""
    pair_counts = Counter()
    for token, count in token_counts.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
    return pair_counts

def find_most_common_pair(pair_counts, vocab):
    """找到频率最高的字节对，如果频率相同则按字典序更大的优先 (保持不变)"""
    if not pair_counts:
        return None, 0
    max_count = max(pair_counts.values())
    candidates = [pair for pair, count in pair_counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0], max_count
    candidates.sort(key=lambda x: (vocab.get(x[0], b''), vocab.get(x[1], b'')), reverse=True)
    return candidates[0], max_count

# --- train_bpe 函数是优化的核心 ---

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Args: (同上)
    Returns: (同上)
    """
    # 1. 初始化词汇表 (逻辑不变)
    vocab = {}
    merges = []
    next_token_id = 0
    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1

    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        # 假设特殊token不会和基础字节冲突
        vocab[next_token_id] = token_bytes
        next_token_id += 1
    
    # 2. 并行处理文件获取初始词符计数 (逻辑不变)
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        results = pool.map(pre_split_word, chunk_args)
    token_counts = Counter()
    for count_dict in results:
        token_counts.update(count_dict)
    
    # 3. --- 新增优化步骤: 构建初始字节对频率和反向索引 ---
    print("Building initial stats and inverted index...")
    pair_counts = Counter()
    # 这是核心优化数据结构: { (pair_byte1, pair_byte2): set(token1, token2, ...), ... }
    pair_to_tokens_map = defaultdict(set)
    for token, count in token_counts.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
            pair_to_tokens_map[pair].add(token)

    # 4. BPE算法主循环 (已使用反向索引优化)
    print("Starting BPE merges...")
    while len(vocab) < vocab_size:
        # 4.1 找到频率最高的字节对 (逻辑不变)
        most_common_pair, _ = find_most_common_pair(pair_counts, vocab)
        if most_common_pair is None:
            break
        
        a, b = most_common_pair
        
        # 4.2 创建新词符并记录 (逻辑不变)
        new_token_id = next_token_id
        new_token_bytes = vocab[a] + vocab[b]
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[a], vocab[b]))

        # 4.3 --- 核心优化: 只更新受影响的词符 ---
        # 旧的逻辑是遍历整个 token_counts，现在我们只遍历包含 most_common_pair 的词符
        affected_tokens = list(pair_to_tokens_map[most_common_pair])
        
        for old_token in affected_tokens:
            # 这个词符可能在之前的迭代中已经被处理并删除了
            if old_token not in token_counts:
                continue
            
            count = token_counts[old_token]

            # --- 步骤A: 从全局统计中“撤销”旧词符的贡献 ---
            for i in range(len(old_token) - 1):
                pair = (old_token[i], old_token[i+1])
                pair_counts[pair] -= count
                # 如果一个字节对的计数降为0，则从索引中移除
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    if pair in pair_to_tokens_map:
                         del pair_to_tokens_map[pair]
                # 即使计数不为0，也要从该字节对的索引中移除当前这个旧词符
                elif pair in pair_to_tokens_map:
                    pair_to_tokens_map[pair].discard(old_token)

            del token_counts[old_token]

            # --- 步骤B: 创建新词符，并将其贡献“添加”到全局统计中 ---
            i = 0
            new_token_list = []
            while i < len(old_token):
                if i < len(old_token) - 1 and old_token[i] == a and old_token[i+1] == b:
                    new_token_list.append(new_token_id)
                    i += 2
                else:
                    new_token_list.append(old_token[i])
                    i += 1
            new_token = tuple(new_token_list)
            
            token_counts[new_token] += count

            # 将新词符的字节对信息添加到全局统计和索引中
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i+1])
                pair_counts[pair] += count
                pair_to_tokens_map[pair].add(new_token)
        
        # 清理工作：已合并的字节对本身不再需要存在
        if most_common_pair in pair_counts:
            del pair_counts[most_common_pair]
        if most_common_pair in pair_to_tokens_map:
            del pair_to_tokens_map[most_common_pair]
        
        next_token_id += 1
        
            
    return vocab, merges

# --- 保存和加载模型的函数 (保持不变) ---
def save_bpe_model_binary(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str):
    """将BPE模型保存为二进制格式"""
    with open(output_path, "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "merges": merges
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bpe_model_binary(input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从二进制文件加载BPE模型"""
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["merges"]