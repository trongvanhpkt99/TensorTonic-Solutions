import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # map vocab → index
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # vector kết quả
    bow = np.zeros(len(vocab), dtype=int)
    
    for token in tokens:
        idx = word_to_idx.get(token)
        if idx is not None:   # token nằm trong vocab
            bow[idx] += 1
        # else: bỏ qua (tương đương = 0)

    return bow