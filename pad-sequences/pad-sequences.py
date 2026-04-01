import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if len(seqs) == 0:
        return np.zeros((0, 0), dtype=int)
    
    # xác định max_len
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    N = len(seqs)
    
    # tạo matrix đầy pad_value
    result = np.full((N, max_len), pad_value, dtype=int)
    
    # fill dữ liệu
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]  # truncate nếu dài
        result[i, :len(seq)] = seq
    
    return result