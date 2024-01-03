import time
import torch
import numpy as np
from guidance.cpp import ByteTrie
from guidance._grammar import ByteRange
from guidance._parser import EarleyCommitParser


class Timer:

    def __init__(self, name = "time", use_cuda_synchronize = False):
        self.name = name
        self.use_cuda_synchronize = use_cuda_synchronize
        self.start = None

    def __enter__(self):
        if self.use_cuda_synchronize:
            torch.cuda.synchronize()
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda_synchronize:
            torch.cuda.synchronize()
        time_used = time.time() - self.start
        print(f"{self.name}: {time_used*1000:.3f}ms")
        self.start = None


class ByteTokenizer:

    def __init__(self, tokenizer):
        self.tokens = []
        for token_id in range(len(tokenizer)):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token, str):
                token_ = tokenizer.convert_tokens_to_string(token)
                if token_ != token:
                    token = tokenizer.convert_tokens_to_string(["a", token])
                    if token[0] == "a": token = token[1: ]
                    elif token[1] == "a": token = token[2: ]
                    else: raise Exception("Can't determine tokenstring representation!")
                token = bytes(token, encoding = "utf-8")
            assert isinstance(token, bytes)
            self.tokens.append(token)
        self.trie = ByteTrie(self.tokens, np.arange(len(self.tokens)))
        assert tokenizer.eos_token_id is not None
        self.eos_token_id = tokenizer.eos_token_id


class GuidanceController:

    def __init__(self, byte_tokenizer, grammar, forbidden_token_ids):
        self.eos_token_id = byte_tokenizer.eos_token_id
        self.tokens = byte_tokenizer.tokens
        self.trie = byte_tokenizer.trie
        self.parser = EarleyCommitParser(grammar)
        self.forbidden_token_ids = forbidden_token_ids
        self.pre_decoded_token_ids = []
        self.any = ByteRange(b"\x00\xff")

    def mark_new_token(self):
        self.parser.mark_new_token()

    def consume_token(self, token):
        assert isinstance(token, bytes)
        for i in range(len(token)):
            commit_point = self.parser.consume_byte(token[i: i+1]) # TODO: 搞清楚 commit_point 到底是干啥的 ……

    def consume_token_id(self, token_id):
        if token_id != self.eos_token_id:
            self.consume_token(self.tokens[token_id])

    def pre_decode(self):
        get_new_tokens = False
        pos = self.parser.state_set_pos
        parser = self.parser
        trie = self.trie
        token_id = None
        tmp_pos = None
        while True:
            next_byte_mask = parser.next_byte_mask()
            num_valid_bytes = next_byte_mask.sum()
            if num_valid_bytes == 0:
                if token_id is not None:
                    self.pre_decoded_token_ids.append(token_id)
                    get_new_tokens = True
                break
            elif num_valid_bytes == 1:
                byte = int(next_byte_mask.nonzero()[0][0]).to_bytes(1, "little")
                if trie.has_child(byte):
                    trie = trie.child(byte)
                    commit_point = parser.consume_byte(byte)
                    if trie.value >= 0:
                        token_id = trie.value
                        tmp_pos = parser.state_set_pos
                else:
                    assert token_id is not None
                    assert tmp_pos is not None
                    trie = self.trie
                    parser.pos = tmp_pos
                    parser.mark_new_token()
                    self.pre_decoded_token_ids.append(token_id)
                    get_new_tokens = True
                    token_id = None
            else:
                for byte in trie.keys():
                    if next_byte_mask[byte[0]]:
                        break
                else:
                    if tmp_pos is not None and tmp_pos <= parser.state_set_pos:
                        assert token_id is not None
                        self.pre_decoded_token_ids.append(token_id)
                        get_new_tokens = True
                        token_id = None
                break
        self.parser.pos = pos
        return get_new_tokens

    def check_token_id(self, token_id):
        token = self.tokens[token_id]
        trie = self.trie
        pos = self.parser.state_set_pos
        new_token_id = None
        for i in range(len(token)):
            try:
                commit_point = self.parser.consume_byte(token[i: i+1])
                trie = trie.child(token[i: i+1])
                if trie.value >= 0:
                    new_token_id = trie.value
            except Exception as e:
                break
        else: assert new_token_id == token_id, f"{new_token_id}, {token_id}"
        self.parser.pos = pos # roll back parser
        return new_token_id
    
    def valid_token_ids(self, use_trie = True): # 树搜索可以继续优化
        def search_valid_token_ids_by_trie(parser, trie):
            if self.any in parser.valid_next_bytes(): # 无约束 gen 生成
                if trie.parent() is not None:         # token 后缀
                    return []                         # 无效
            valid_token_ids = []
            next_byte_mask = parser.next_byte_mask()
            for next_byte in trie.keys():
                if not next_byte_mask[next_byte[0]]: continue
                commit_point = parser.consume_byte(next_byte) # TODO: 搞清楚 commit_point 到底是干啥的 ……
                next_trie = trie.child(next_byte)
                if next_trie.value >= 0:
                    valid_token_ids.append(next_trie.value)
                valid_token_ids.extend(search_valid_token_ids_by_trie(parser, next_trie))
                parser.pos = parser.state_set_pos-1 # roll back parser
            return valid_token_ids
        if use_trie:
            return search_valid_token_ids_by_trie(self.parser, self.trie)
        return [token_id for token_id in range(len(self.tokens)) if token_id == self.check_token_id(token_id)]

    def greedy_for_unconstrained_gen(self, logits, use_fast): # 针对无约束 gen 的快速解码
        if use_fast:
            if len(self.forbidden_token_ids) > 0:
                sorted_token_ids = torch.topk(logits, len(self.forbidden_token_ids)+1)[1]
            else:
                sorted_token_ids = [torch.argmax(logits)]
        else:
            sorted_token_ids = torch.argsort(-logits)
        for i in range(len(sorted_token_ids)):
            token_id = sorted_token_ids[i].item()
            if token_id in self.forbidden_token_ids:
                continue
            new_token_id = self.check_token_id(token_id)
            if use_fast or new_token_id == token_id:
                return new_token_id
        raise Exception("Invalid Unconstrained Gen!")

    def greedy_next_token_id(self, logits, use_fast = True):
        valid_next_bytes = self.parser.valid_next_bytes()
        if len(valid_next_bytes) == 0:
            return self.eos_token_id, []
        if self.any in valid_next_bytes:
            return self.greedy_for_unconstrained_gen(logits, use_fast), None
        valid_token_ids = self.valid_token_ids(use_trie = True)
        valid_token_ids = list(filter(lambda token_id: token_id not in self.forbidden_token_ids, valid_token_ids))
        if len(valid_token_ids) > 0:
            token_id = torch.argmax(logits[valid_token_ids], dim = -1)
            token_id = valid_token_ids[token_id]
        else:
            token_id = self.eos_token_id
        return token_id, valid_token_ids

    def random_for_unconstrained_gen(self, probs, use_fast): # 针对无约束 gen 的快速解码
        if use_fast:
            sorted_token_ids = torch.multinomial(probs, len(self.forbidden_token_ids)+1)
        else:
            sorted_token_ids = torch.multinomial(probs, len(new_probs))
        for i in range(len(sorted_token_ids)):
            token_id = sorted_token_ids[i].item()
            if token_id in self.forbidden_token_ids:
                continue
            new_token_id = self.check_token_id(token_id)
            if use_fast or new_token_id == token_id:
                return new_token_id
        raise Exception("Invalid Unconstrained Gen!")
        
    def random_next_token_id(self, probs, use_fast = True):
        valid_next_bytes = self.parser.valid_next_bytes()
        if len(valid_next_bytes) == 0:
            return self.eos_token_id, []
        if self.any in valid_next_bytes:
            return self.random_for_unconstrained_gen(probs, use_fast), None
        valid_token_ids = self.valid_token_ids(use_trie = True)
        valid_token_ids = list(filter(lambda token_id: token_id not in self.forbidden_token_ids, valid_token_ids))
        if len(valid_token_ids) > 0:
            new_probs = probs[valid_token_ids]+1e-32
            new_probs /= new_probs.sum()
            token_id = torch.multinomial(new_probs, 1).item()
            token_id = valid_token_ids[token_id]
            return token_id, valid_token_ids
        else: return self.eos_token_ids, []

    def next_token_id(self, logits, sampling_params_temperature, use_fast = True):
        assert len(self.forbidden_token_ids) < len(self.tokens)
        new_logits = logits[: len(self.tokens)]
        temperature = self.parser.next_byte_temperature() # Guidance 语法中 temperature 默认为 -1
        if temperature < 1e-32:
            if sampling_params_temperature < 1e-32:
                # 要做 Greedy 生成，必须 sampling_params 中 temperature 设置为 0，且 guidance 语法中 temperature 采用默认（-1）
                return self.greedy_next_token_id(new_logits, use_fast)
        else: # 以 Guidance 的 temperature 为准计算概率
            if sampling_params_temperature > 1e-32:
                new_logits *= sampling_params_temperature # 原来的 sampling_params_temperature 在之前已经除过了，这里要乘回来
            new_logits /= temperature
        probs = torch.softmax(new_logits, dim = 0)
        return self.random_next_token_id(probs, use_fast)

