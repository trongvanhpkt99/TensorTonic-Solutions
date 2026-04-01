import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.array(X)
    y = np.array(y)
    
    N = len(X)
    
    # tạo index
    indices = np.arange(N)
    
    # shuffle
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)
    
    # tạo batch
    for start in range(0, N, batch_size):
        end = start + batch_size
        
        if end > N and drop_last:
            break
        
        batch_idx = indices[start:end]
        
        yield X[batch_idx], y[batch_idx]