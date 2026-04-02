import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    # ===== VALIDATION =====
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError("Q, K, V must be 3D arrays")

    B, N, d_model = Q.shape

    if K.shape != (B, K.shape[1], d_model) or V.shape != (B, V.shape[1], d_model):
        raise ValueError("K, V must have same batch and d_model as Q")

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    d_k = d_model // num_heads

    # ===== 1. Linear projection =====
    Q_proj = Q @ W_q   # (B, N, d_model)
    K_proj = K @ W_k
    V_proj = V @ W_v

    # ===== 2. Split heads =====
    def split_heads(x):
        # (B, N, h, d_k) -> (B, h, N, d_k)
        return x.reshape(B, N, num_heads, d_k).transpose(0, 2, 1, 3)

    Qh = split_heads(Q_proj)
    Kh = split_heads(K_proj)
    Vh = split_heads(V_proj)

    # ===== 3. Scaled dot-product attention =====
    # (B, h, N, N)
    scores = Qh @ Kh.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    attn_weights = softmax(scores, axis=-1)

    # (B, h, N, d_k)
    out_heads = attn_weights @ Vh

    # ===== 4. Concatenate heads =====
    # (B, h, N, d_k) -> (B, N, h, d_k)
    out = out_heads.transpose(0, 2, 1, 3).reshape(B, N, d_model)

    # ===== 5. Final linear projection =====
    output = out @ W_o

    return output