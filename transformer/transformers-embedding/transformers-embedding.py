import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
    # Initialization: chuẩn hoá giống Transformer
    nn.init.normal_(embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
    
    return embedding


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # lookup
    out = embedding(tokens)  # shape: (..., d_model)
    
    # scale theo Transformer
    return out * math.sqrt(d_model)