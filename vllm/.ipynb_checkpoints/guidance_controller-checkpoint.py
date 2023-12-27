import time
import torch
import numpy as np


class BTrie:

    def __init__(self, parent = None, token_id = None):
        self.parent = parent
        self.token_id = token_id
        self.children = dict()
        self.max_logit = None
        self.logit = None

    def insert(self, token, token_id):
        if token == b"":
            self.token_id = token_id
            return
        byte = token[: 1]
        if byte not in self.children:
            self.children[byte] = BTrie(self, None)
        self.children[byte].insert(token[1: ], token_id)

    def child(self, byte):
        return self.children.get(byte, None)

    def has_child(self, byte):
        return byte in self.children

    def keys(self):
        return self.children.keys()

    def size(self):
        return len(self.children)

    def kept_token_ids(self):
        token_ids = []
        if self.token_id is not None:
            token_ids += [self.token_id]
        for byte in self.children:
            child = self.children[byte]
            child_token_ids = child.kept_token_ids()
            token_ids += child_token_ids
        return token_ids

    def allocate_logits(self, logits):
        if self.token_id is not None:
            self.logit = float(logits[self.token_id])
        self.max_logit = self.logit if self.logit is not None else -float("inf")
        for byte in self.children:
            child = self.children[byte]
            child.allocate_logits(logits)
            self.max_logit = max(self.max_logit, child.max_logit)
        return 

    def sorted_keys_by_logits(self, ):
        assert self.max_logit is not None, "Please allocate logits first!"
        keys = list(self.keys())
        keys.sort(key = lambda byte: self.child(byte).max_logit, reverse = True)
        return keys


class GuidanceController:

    def __init__(self, hf_tokenizer, parser, use_original_byte_trie = True):
        self.tokens = []
        self.token_ids = []
        self.btrie = BTrie()
        for token_id in range(len(hf_tokenizer)):
            token = hf_tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(token, str):
                token = hf_tokenizer.convert_tokens_to_string(token)
                token = bytes(token, encoding = "utf-8")
            assert isinstance(token, bytes)
            self.tokens.append(token)
            self.token_ids.append(token_id)
            self.btrie.insert(token, token_id)
        if use_original_byte_trie:
            from guidance.cpp import ByteTrie
            from guidance._grammar import ByteRange
            self.trie = ByteTrie(self.tokens, self.token_ids)
            self.any = ByteRange(b"\x00\xff")
        else:
            self.trie = None
            self.any = None
        self.parser = parser
        assert hf_tokenizer.eos_token_id is not None
        self.eos_token_id = hf_tokenizer.eos_token_id
        self.pre_decoded_token_ids = []

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

    def consume_token_id(self, token_id):
        if token_id != self.eos_token_id:
            self.consume_token(self.tokens[token_id])

    def consume_token(self, token):
        assert isinstance(token, bytes)
        for i in range(len(token)):
            commit_point = self.parser.consume_byte(token[i: i+1]) # TODO: 搞清楚 commit_point 到底是干啥的 ……

    def mark_new_token(self, ):
        self.parser.mark_new_token()

    def max_logit_token_id(self, logits):
        self.btrie.allocate_logits(logits)

        def get_max_logit_token_id(parser, trie, max_logit = -float("inf")):
            token_id = None
            next_byte_mask = parser.next_byte_mask()
            for byte in trie.sorted_keys_by_logits():
                if not next_byte_mask[byte[0]]: continue
                next_trie = trie.child(byte)
                if next_trie.max_logit < max_logit: break
                if next_trie.logit is not None and next_trie.logit > max_logit:
                    max_logit = next_trie.logit
                    token_id = next_trie.token_id
                commit_point = parser.consume_byte(byte) # TODO: 搞清楚 commit_point 到底是干啥的 ……
                next_token_id, max_logit = get_max_logit_token_id(parser, next_trie, max_logit)
                parser.pos = parser.state_set_pos-1 # roll back parser
                if next_token_id is not None:
                    token_id = next_token_id
            return token_id, max_logit

        token_id = get_max_logit_token_id(self.parser, self.btrie)[0]
        return token_id if token_id is not None else self.eos_token_id

    def max_logit_token_id_with_original_byte_trie(self, logits, use_fast = True):
        assert self.trie is not None
        
        def get_valid_token_ids(parser, trie):
            if self.any in parser.valid_next_bytes():
                if trie.parent() is None:
                    if use_fast: return None
                    for token_id in torch.argsort(-logits, dim = -1):
                        token_id = token_id.item()
                        if token_id >= len(self.tokens): continue
                        token = self.tokens[token_id]
                        pos = parser.state_set_pos
                        for i in range(len(token)):
                            try:
                                commit_point = parser.consume_byte(token[i: i+1]) # TODO: 搞清楚 commit_point 到底是干啥的 ……
                            except:
                                parser.pos = pos # roll back parser
                                break
                        else:
                            parser.pos = pos # roll back parser
                            return [token_id]
                else:
                    return []
            valid_token_ids = []
            next_byte_mask = parser.next_byte_mask()
            for next_byte in trie.keys():
                if not next_byte_mask[next_byte[0]]: continue
                commit_point = parser.consume_byte(next_byte) # TODO: 搞清楚 commit_point 到底是干啥的 ……
                next_trie = trie.child(next_byte)
                if next_trie.value >= 0:
                    valid_token_ids.append(next_trie.value)
                valid_token_ids.extend(get_valid_token_ids(parser, next_trie))
                parser.pos = parser.state_set_pos-1 # roll back parser
            return valid_token_ids
        
        valid_token_ids = get_valid_token_ids(self.parser, self.trie)
        if valid_token_ids is None: # use_fast == True
            torch.cuda.synchronize()
            t = time.time()
            token_id = logits[: len(self.tokens)].argmax().item()
            torch.cuda.synchronize()
            t = time.time()-t
            print(f"argmax-time: {t*1000:.3f}ms")
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
                    token_id = new_token_id
                    break
            else: assert token_id == new_token_id, f"{token_id}, {new_token_id}"
            self.parser.pos = pos # roll back parser
        elif len(valid_token_ids) > 0:
            token_id = torch.argmax(logits[valid_token_ids], dim = -1).item()
            token_id = valid_token_ids[token_id]
        else:
            token_id = self.eos_token_id
        return token_id, valid_token_ids
