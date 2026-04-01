import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T, dtype=float)
    pts = np.array(points, dtype=float)
    
    single = False
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
        single = True
    
    N = pts.shape[0]
    
    # thêm cột 1 → (N,4)
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts, ones])
    
    # nhân ma trận (vectorized)
    pts_transformed = pts_h @ T.T   # (N,4)
    
    # lấy lại (x,y,z)
    result = pts_transformed[:, :3]
    
    if single:
        return result[0]
    return result