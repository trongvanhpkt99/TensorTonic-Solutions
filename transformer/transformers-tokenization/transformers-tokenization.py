import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        # reset
        self.word_to_id = {}
        self.id_to_word = {}
        
        # add special tokens
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        idx = len(special_tokens)  # start from 4
        
        # add words from texts
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
                    idx += 1
        
        self.vocab_size = idx
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        tokens = []
        
        for word in text.split():
            tokens.append(
                self.word_to_id.get(word, self.word_to_id[self.unk_token])
            )
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = []
        
        for i in ids:
            word = self.id_to_word.get(i, self.unk_token)
            
            # thường bỏ special tokens khi decode
            if word in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            
            words.append(word)
        
        return " ".join(words)
