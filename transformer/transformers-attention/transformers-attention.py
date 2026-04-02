import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here

    if Q.dim() != 3 or K.dim() != 3 or V.dim() != 3:
        raise ValueError("Q, K, V must be 3D tensors")

    if Q.size(0) != K.size(0) or Q.size(0) != V.size(0):
        raise ValueError("Batch size mismatch")

    if K.size(1) != V.size(1):
        raise ValueError("K and V must have same seq_len_k")

    if Q.size(2) != K.size(2):
        raise ValueError("Q and K must have same d_k")

    d_k = Q.size(-1)

    # ===== 1. Compute attention scores =====
    # (batch, seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # ===== 2. Scale =====
    scores = scores / math.sqrt(d_k)

    # ===== 3. Softmax =====
    attn_weights = F.softmax(scores, dim=-1)

    # ===== 4. Weighted sum =====
    # (batch, seq_q, d_v)
    output = torch.matmul(attn_weights, V)

    return output