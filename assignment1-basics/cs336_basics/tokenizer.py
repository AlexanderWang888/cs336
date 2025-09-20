import os
from typing import Iterable, Iterator, List, Dict, Tuple, Optional, Any
import pickle
import regex as re
import pytest

import json
import os
import resource
import sys

import psutil
import pytest


FIXTURES_PATH = "../tests/fixtures"
VOCAB_PATH = FIXTURES_PATH + "/gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH + "/gpt2_merges.txt"

class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                 special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            pattern = f"({pattern})"
            segments = re.split(pattern, text) if pattern else [text]
        else:
            segments = [text]

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        token_ids = []
        for segment in segments:
            if self.special_tokens and segment in self.special_tokens:
                # 检查特殊令牌是否在反向词汇表中
                encoded_segment = segment.encode('utf-8')
                token_id = self.reverse_vocab.get(encoded_segment)
                if token_id is None:
                    # 如果不存在，动态添加到词汇表
                    next_id = max(self.vocab.keys()) + 1
                    self.vocab[next_id] = encoded_segment
                    self.reverse_vocab[encoded_segment] = next_id
                    token_id = next_id
                token_ids.append(token_id)
            else:
                # 1. 使用正则表达式对普通文本块进行初步切分
                initial_tokens = [match.group(0) for match in re.finditer(PAT, segment)]
                
                # 2. 对切分出的每一个 token 应用 BPE 合并
                for token_str in initial_tokens:
                    token_bytes = token_str.encode('utf-8')
                    # 将 token 字节序列分解为单字节列表
                    parts = [b.to_bytes(1, 'little') for b in token_bytes]

                    # 3. 按照 merges 列表的顺序，依次应用每一个合并规则
                    for merge_pair in self.merges:
                        # 如果当前 token 已经合并成单个元素，则无需再进行后续合并
                        if len(parts) < 2:
                            break
                        
                        i = 0
                        new_parts = []
                        while i < len(parts):
                            # 检查当前位置和下一个位置的字节是否匹配合并规则
                            if (i < len(parts) - 1 and
                                    parts[i] == merge_pair[0] and
                                    parts[i+1] == merge_pair[1]):
                                # 如果匹配，则合并，并将合并后的结果添加到新列表
                                new_parts.append(merge_pair[0] + merge_pair[1])
                                # 跳过下一个字节，因为它已经被合并了
                                i += 2
                            else:
                                # 如果不匹配，则直接将当前字节添加到新列表
                                new_parts.append(parts[i])
                                i += 1
                        # 用合并后的新列表替换旧列表，为下一个合并规则做准备
                        parts = new_parts
                    
                    # 4. 将最终合并好的字节序列通过反向词汇表转换为 token ID
                    for part in parts:
                        token_id = self.reverse_vocab.get(part)
                        if token_id is not None:
                            token_ids.append(token_id)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            # 对每一个文本块调用encode方法，得到一个token ID列表
            # 然后使用 'yield from' 将列表中的每一个ID作为生成器的结果产出
            yield from self.encode(text_chunk)

    def decode(self, ids: List[int]) -> str:
        text_byte = b''
        for id in ids:
            byte = self.vocab.get(id, b'') # 使用 .get() 更安全，以防 id 不存在
            text_byte += byte
        return text_byte.decode('utf-8', errors='replace')


def test_encode_iterable():
    with open('./vocab.pkl', "rb") as f:
        vocab = pickle.load(f)
    tokenizer = Tokenizer.from_files('./vocab.pkl', './merges.pkl')

    # 一个较长的测试文本
    long_text = "Hello, world! This is a long piece of text to test the iterable encoding. It should be processed chunk by chunk without losing information."
    
    # 将文本分成几个块来模拟可迭代对象
    text_chunks = [
        "Hello, world! This is a long ",
        "piece of text to test the iterable encoding. ",
        "It should be processed chunk by chunk without losing information."
    ]

    # 使用encode_iterable进行编码
    encoded_ids_iterable = list(tokenizer.encode_iterable(text_chunks))
    
    # 直接对完整文本进行编码，作为基准
    encoded_ids_full = tokenizer.encode(long_text)
    
    # 验证两种方式的结果是否一致
    assert encoded_ids_iterable == encoded_ids_full
    assert tokenizer.decode(encoded_ids_iterable) == long_text
    
    print("test_encode_iterable 成功通过！")


def main():
    # 测试特殊令牌
    special_tokens =["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    with open('./vocab.pkl', "rb") as f:
        vocab = pickle.load(f)
    # 创建Tokenizer实例
    tokenizer = Tokenizer.from_files('./vocab.pkl', './merges.pkl', special_tokens)
    
    # 测试文本
    test_texts = [
        "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ]

    ids = tokenizer.encode(test_texts[0])
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    print(tokenized_string)
    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count("<|endoftext|>") == 1
    assert tokenized_string.count("<|endoftext|><|endoftext|>") == 1
    # Test roundtrip
    assert tokenizer.decode(ids) == test_texts[0]
    
    # 对每个测试文本进行编码并打印结果
    # for text in test_texts:
    #     print(f"测试文本: {text}")
    #     try:
    #         token_ids = tokenizer.encode(text)
    #         print(f"编码结果: {token_ids}")
    #         # 打印每个token ID对应的字节表示
    #         token_str = tokenizer.decode(token_ids)
    #         print(f"对应的字节: {token_str}")
    #         assert token_str == text
    #     except Exception as e:
    #         print(f"编码过程出错: {str(e)}")
    #     print("-" * 50)


if __name__ == "__main__":
    test_encode_iterable()
    main()