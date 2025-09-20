# import os
# from multiprocessing import Pool
# import regex as re
# from collections import Counter, defaultdict # 引入 defaultdict
# import pickle
# from typing import BinaryIO
# import time
# import psutil


# process = psutil.Process(os.getpid())

# def log_memory_usage(step: str):
#     """记录当前内存使用情况"""
#     mem_info = process.memory_info()
#     # 转换为MB
#     rss = mem_info.rss / (1024 * 1024)  # 常驻内存
#     vms = mem_info.vms / (1024 * 1024)  # 虚拟内存
#     print(f"[{step}] 内存使用 - 常驻内存: {rss:.2f} MB, 虚拟内存: {vms:.2f} MB")
#     return rss

# def find_chunk_boundaries(
#     file: BinaryIO,
#     desired_num_chunks: int,
#     split_special_token: bytes,
# ) -> list[int]:
#     assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
#     file.seek(0, os.SEEK_END)
#     file_size = file.tell()
#     file.seek(0)
#     chunk_size = file_size // desired_num_chunks
#     chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
#     chunk_boundaries[-1] = file_size
#     mini_chunk_size = 4096
#     for bi in range(1, len(chunk_boundaries) - 1):
#         initial_position = chunk_boundaries[bi]
#         file.seek(initial_position)
#         while True:
#             mini_chunk = file.read(mini_chunk_size)
#             if mini_chunk == b"":
#                 chunk_boundaries[bi] = file_size
#                 break
#             found_at = mini_chunk.find(split_special_token)
#             if found_at != -1:
#                 chunk_boundaries[bi] = initial_position + found_at
#                 break
#             initial_position += mini_chunk_size
#     return sorted(set(chunk_boundaries))

# def pre_split_word(args):
#     """
#     Worker function to read and decode a specific chunk of the file.
#     (此函数内代码有微小但重要的修正，以正确处理token)
#     """
#     input_path, start, end, special_tokens = args
#     with open(input_path, "rb") as f:
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore").replace('\r', '')
#         if special_tokens:
#             pattern = "|".join(re.escape(tok) for tok in special_tokens)
#             segments = re.split(pattern, chunk) if pattern else [chunk]
#         else:
#             segments = [chunk]

#         PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
#         pre_tokens = []
#         for segment in segments:
#             tokens = [match.group(0) for match in re.finditer(PAT, segment)]
#             # 修正: 原代码的 tuple(text.encode()) 会将多字节字符错误分割
#             # 正确做法是先编码，再将字节序列转为整数元组
#             tokens_list = [tuple(text.encode("utf-8")) for text in tokens]
#             pre_tokens.extend(tokens_list)

#         token_counts = Counter(pre_tokens)
#         return dict(token_counts)

# def get_byte_pairs_and_counts(token_counts):
#     """统计所有相邻字节对的出现频率"""
#     pair_counts = Counter()
#     for token, count in token_counts.items():
#         for i in range(len(token) - 1):
#             pair = (token[i], token[i+1])
#             pair_counts[pair] += count
#     return pair_counts

# def find_most_common_pair(pair_counts, vocab):
#     """找到频率最高的字节对，如果频率相同则按字典序更大的优先 (保持不变)"""
#     if not pair_counts:
#         return None, 0
#     max_count = max(pair_counts.values())
#     candidates = [pair for pair, count in pair_counts.items() if count == max_count]
#     if len(candidates) == 1:
#         return candidates[0], max_count
#     candidates.sort(key=lambda x: (vocab.get(x[0], b''), vocab.get(x[1], b'')), reverse=True)
#     return candidates[0], max_count

# # --- train_bpe 函数是优化的核心 ---

# def train_bpe(
#     input_path: str | os.PathLike,
#     vocab_size: int,
#     special_tokens: list[str],
#     **kwargs,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """
#     Args: (同上)
#     Returns: (同上)
#     """
#     # 1. 初始化词汇表 (逻辑不变)
#     vocab = {}
#     merges = []
#     next_token_id = 0
#     for i in range(256):
#         vocab[next_token_id] = bytes([i])
#         next_token_id += 1

#     for token_str in special_tokens:
#         token_bytes = token_str.encode("utf-8")
#         # 假设特殊token不会和基础字节冲突
#         vocab[next_token_id] = token_bytes
#         next_token_id += 1
    
#     # 2. 并行处理文件获取初始词符计数 (逻辑不变)
#     with open(input_path, "rb") as f:
#         num_processes = 8
#         boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
#     chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
#     with Pool(processes=num_processes) as pool:
#         results = pool.map(pre_split_word, chunk_args)
#     token_counts = Counter()
#     for count_dict in results:
#         token_counts.update(count_dict)
#     print("Building initial stats and inverted index...")
#     pair_counts = Counter()
#     pair_to_tokens_map = defaultdict(set)
#     for token, count in token_counts.items():
#         for i in range(len(token) - 1):
#             pair = (token[i], token[i+1])
#             pair_counts[pair] += count
#             pair_to_tokens_map[pair].add(token)

#     # 4. BPE算法主循环 (已使用反向索引优化)
#     print("Starting BPE merges...")
#     log_memory_usage("init finish")

#     while len(vocab) < vocab_size:
#         # 4.1 找到频率最高的字节对 (逻辑不变)
#         most_common_pair, _ = find_most_common_pair(pair_counts, vocab)
#         if most_common_pair is None:
#             break
        
#         a, b = most_common_pair
        
        
#         # 4.2 创建新词符并记录 (逻辑不变)
#         new_token_id = next_token_id
#         new_token_bytes = vocab[a] + vocab[b]
#         vocab[new_token_id] = new_token_bytes
#         merges.append((vocab[a], vocab[b]))

#         # 4.3 --- 核心优化: 只更新受影响的词符 ---
#         # 旧的逻辑是遍历整个 token_counts，现在我们只遍历包含 most_common_pair 的词符
#         affected_tokens = list(pair_to_tokens_map[most_common_pair])
        
#         for old_token in affected_tokens:
#             # 这个词符可能在之前的迭代中已经被处理并删除了
#             if old_token not in token_counts:
#                 continue
            
#             count = token_counts[old_token]

#             # --- 步骤A: 从全局统计中“撤销”旧词符的贡献 ---
#             for i in range(len(old_token) - 1):
#                 pair = (old_token[i], old_token[i+1])
#                 pair_counts[pair] -= count
#                 # 如果一个字节对的计数降为0，则从索引中移除
#                 if pair_counts[pair] <= 0:
#                     del pair_counts[pair]
#                     if pair in pair_to_tokens_map:
#                          del pair_to_tokens_map[pair]
#                 # 即使计数不为0，也要从该字节对的索引中移除当前这个旧词符
#                 elif pair in pair_to_tokens_map:
#                     pair_to_tokens_map[pair].discard(old_token)

#             del token_counts[old_token]

#             # --- 步骤B: 创建新词符，并将其贡献“添加”到全局统计中 ---
#             i = 0
#             new_token_list = []
#             while i < len(old_token):
#                 if i < len(old_token) - 1 and old_token[i] == a and old_token[i+1] == b:
#                     new_token_list.append(new_token_id)
#                     i += 2
#                 else:
#                     new_token_list.append(old_token[i])
#                     i += 1
#             new_token = tuple(new_token_list)
            
#             token_counts[new_token] += count

#             # 将新词符的字节对信息添加到全局统计和索引中
#             for i in range(len(new_token) - 1):
#                 pair = (new_token[i], new_token[i+1])
#                 pair_counts[pair] += count
#                 pair_to_tokens_map[pair].add(new_token)
        
#         # 清理工作：已合并的字节对本身不再需要存在
#         if most_common_pair in pair_counts:
#             del pair_counts[most_common_pair]
#         if most_common_pair in pair_to_tokens_map:
#             del pair_to_tokens_map[most_common_pair]
        
#         next_token_id += 1
        
    
#     return vocab, merges

# # --- 保存和加载模型的函数 (保持不变) ---
# def save_bpe_model_binary(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str):
#     """将BPE模型保存为二进制格式"""
#     with open(output_path + 'vocab.pkl', "wb") as f:
#         pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(output_path + 'merges.pkl', "wb") as f:
#         pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

# def load_bpe_model_binary(input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
#     """从二进制文件加载BPE模型"""
#     with open(input_path, "rb") as f:
#         data = pickle.load(f)
#     return data["vocab"], data["merges"]
# def find_longest_vocab_key(vocab: dict):
#     """查找vocab中最长的key（tuple类型），并打印其信息"""
#     if not vocab:
#         print("词汇表为空")
#         return
    
#     # 找到最长的key及其长度
#     longest_length = 0
#     longest_key = None
#     longest_value = None
    
#     for key, value in vocab.items():
#         # 对于整数key，长度视为1；对于元组key，使用其元素数量
#         current_length = len(value)
        
#         if current_length > longest_length:
#             longest_length = current_length
#             longest_key = key
#             longest_value = value
    
#     print(f"\n最长的vocab key信息:")
#     print(f"key: {longest_key}")
#     print(f"key长度（元素数量）: {longest_length}")
#     print(f"对应的value（字节）: {longest_value}")
#     try:
#         # 尝试解码为字符串（可能包含无法解码的字节）
#         print(f"value解码为字符串: {longest_value.decode('utf-8', errors='replace')}")
#     except UnicodeDecodeError:
#         print("value无法解码为UTF-8字符串")

# def main():
#     start_time = time.time()
#     log_memory_usage("主程序开始")
#     vocab, merges = train_bpe('../data/corpus.en', 10000, ['<|endoftext|>'])
#     log_memory_usage("训练完成")
#     # print(vocab)
#     # print(merges)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(elapsed_time)
#     save_bpe_model_binary(vocab, merges, './')
#     find_longest_vocab_key(vocab)
    

# if __name__ == '__main__':
#     main()




import os
from multiprocessing import Pool
import regex as re
from collections import Counter, defaultdict
import pickle
from typing import BinaryIO
import time
import psutil

# 全局变量来追踪内存峰值
MAX_MEMORY_MB = 0.0
# 全局变量来存储每个步骤的用时
TIMING = {}
# 获取当前进程对象
process = psutil.Process(os.getpid())

def log_memory_usage(step: str) -> float:
    """记录当前内存使用情况并更新全局内存峰值"""
    global MAX_MEMORY_MB
    mem_info = process.memory_info()
    rss = mem_info.rss / (1024 * 1024)  # 常驻内存
    MAX_MEMORY_MB = max(MAX_MEMORY_MB, rss)
    return rss

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
    global TIMING
    
    # 1. 初始化词汇表
    start_time = time.time()
    vocab = {}
    merges = []
    next_token_id = 0
    for i in range(256):
        vocab[next_token_id] = bytes([i])
        next_token_id += 1
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        vocab[next_token_id] = token_bytes
        next_token_id += 1
    log_memory_usage("初始化词汇表")
    TIMING["初始化"] = time.time() - start_time

    # 2. 并行处理文件获取初始词符计数
    start_time = time.time()
    with open(input_path, "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        results = pool.map(pre_split_word, chunk_args)
    token_counts = Counter()
    for count_dict in results:
        token_counts.update(count_dict)
    log_memory_usage("并行预处理")
    TIMING["并行预处理"] = time.time() - start_time

    # 3. 建立初始统计和反向索引
    start_time = time.time()
    print("Building initial stats and inverted index...")
    pair_counts = Counter()
    pair_to_tokens_map = defaultdict(set)
    for token, count in token_counts.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
            pair_to_tokens_map[pair].add(token)
    log_memory_usage("建立初始索引")
    TIMING["建立初始索引"] = time.time() - start_time

    # 4. BPE算法主循环
    print("Starting BPE merges...")
    start_time = time.time()
    while len(vocab) < vocab_size:
        # 4.1 找到频率最高的字节对
        most_common_pair, _ = find_most_common_pair(pair_counts, vocab)
        if most_common_pair is None:
            break
        a, b = most_common_pair
        
        # 4.2 创建新词符并记录
        new_token_id = next_token_id
        new_token_bytes = vocab[a] + vocab[b]
        vocab[new_token_id] = new_token_bytes
        merges.append((vocab[a], vocab[b]))

        # 4.3 核心优化: 只更新受影响的词符
        affected_tokens = list(pair_to_tokens_map[most_common_pair])
        for old_token in affected_tokens:
            if old_token not in token_counts:
                continue
            count = token_counts[old_token]

            # 步骤A: 从全局统计中“撤销”旧词符的贡献
            for i in range(len(old_token) - 1):
                pair = (old_token[i], old_token[i+1])
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    if pair in pair_to_tokens_map:
                        del pair_to_tokens_map[pair]
                elif pair in pair_to_tokens_map:
                    pair_to_tokens_map[pair].discard(old_token)
            del token_counts[old_token]

            # 步骤B: 创建新词符，并将其贡献“添加”到全局统计中
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
            for i in range(len(new_token) - 1):
                pair = (new_token[i], new_token[i+1])
                pair_counts[pair] += count
                pair_to_tokens_map[pair].add(new_token)
        
        if most_common_pair in pair_counts:
            del pair_counts[most_common_pair]
        if most_common_pair in pair_to_tokens_map:
            del pair_to_tokens_map[most_common_pair]
        next_token_id += 1
        log_memory_usage(f"第 {len(merges)} 次合并")
    
    TIMING["BPE合并循环"] = time.time() - start_time
    
    return vocab, merges

# --- 保存和加载模型的函数 (保持不变) ---
def save_bpe_model_binary(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_path: str):
    """将BPE模型保存为二进制格式"""
    with open(output_path + 'vocab.pkl', "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_path + 'merges.pkl', "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_bpe_model_binary(input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从二进制文件加载BPE模型"""
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    return data["vocab"], data["merges"]

def find_longest_vocab_key(vocab: dict):
    """查找vocab中最长的key（tuple类型），并打印其信息"""
    if not vocab:
        print("词汇表为空")
        return
    
    longest_length = 0
    longest_key = None
    longest_value = None
    
    for key, value in vocab.items():
        current_length = len(value)
        if current_length > longest_length:
            longest_length = current_length
            longest_key = key
            longest_value = value
    
    print(f"\n最长的vocab key信息:")
    print(f"key: {longest_key}")
    print(f"key长度（元素数量）: {longest_length}")
    print(f"对应的value（字节）: {longest_value}")
    try:
        print(f"value解码为字符串: {longest_value.decode('utf-8', errors='replace')}")
    except UnicodeDecodeError:
        print("value无法解码为UTF-8字符串")

def main(dataset_path='../data/corpus.en', merge_num=20, special_tokens=['<|endoftext|>']):
    global MAX_MEMORY_MB, TIMING
    start_time = time.time()
    print("--- BPE 训练开始 ---")
    log_memory_usage("主程序开始")
    
    vocab, merges = train_bpe(dataset_path, merge_num, special_tokens)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('vocab size', len(vocab))
    print('merges size', len(merges))
    
    print("-" * 20)
    print("--- 性能统计 ---")
    for step, duration in TIMING.items():
        print(f"{step} 用时: {duration:.2f} 秒")
    print("-" * 20)
    print(f"总用时: {elapsed_time:.2f} 秒")
    print(f"内存使用峰值 (常驻内存): {MAX_MEMORY_MB:.2f} MB")
    print("-" * 20)

    save_bpe_model_binary(vocab, merges, './')
    find_longest_vocab_key(vocab)
    
if __name__ == '__main__':
    main()