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

    def max_logit_token_id(self, parser, logits):
        assert self.parent is None
        self.allocate_logits(logits)

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

        token_id = get_max_logit_token_id(parser, self)[0]
        return token_id