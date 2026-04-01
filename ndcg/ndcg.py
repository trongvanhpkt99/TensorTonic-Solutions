import math
import numpy as np
def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    rel = np.array(relevance_scores, dtype=float)
    
    if len(rel) == 0:
        return 0.0
    
    k = min(k, len(rel))
    
    # lấy top-k
    rel_k = rel[:k]
    
    # positions: 1 → k
    positions = np.arange(1, k + 1)
    
    # discount: log2(i+1)
    discounts = np.log2(positions + 1)
    
    # gain: 2^rel - 1
    gains = (2 ** rel_k) - 1
    
    dcg = np.sum(gains / discounts)
    
    # IDCG: sort giảm dần
    ideal_rel = np.sort(rel)[::-1][:k]
    ideal_gains = (2 ** ideal_rel) - 1
    idcg = np.sum(ideal_gains / discounts)
    
    if idcg == 0:
        return 0.0
    
    return float(dcg / idcg)