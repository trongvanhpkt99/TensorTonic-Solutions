import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.

    Args:
        feature_map: 2D list (H x W)
        rois: list of [x1, y1, x2, y2]
        output_size: int

    Returns:
        List of pooled feature maps (each is output_size x output_size)
    """

    # ===== VALIDATION =====
    if not feature_map or not feature_map[0]:
        raise ValueError("feature_map must be non-empty")

    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError("output_size must be positive integer")

    H = len(feature_map)
    W = len(feature_map[0])

    results = []

    for roi in rois:
        if len(roi) != 4:
            raise ValueError(f"Invalid ROI format: {roi}")

        x1, y1, x2, y2 = roi

        # clamp ROI vào feature map
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid ROI with zero area: {roi}")

        roi_w = x2 - x1
        roi_h = y2 - y1

        pooled = []

        for i in range(output_size):
            row = []
            for j in range(output_size):

                # ===== COMPUTE BIN =====
                h_start = y1 + int((i * roi_h) // output_size)
                h_end   = y1 + int(((i + 1) * roi_h) // output_size)

                w_start = x1 + int((j * roi_w) // output_size)
                w_end   = x1 + int(((j + 1) * roi_w) // output_size)

                # ===== ENSURE AT LEAST 1 PIXEL =====
                if h_end <= h_start:
                    h_end = h_start + 1
                if w_end <= w_start:
                    w_end = w_start + 1

                # clamp lại để tránh overflow
                h_end = min(h_end, H)
                w_end = min(w_end, W)

                # ===== MAX POOL =====
                max_val = float('-inf')

                for y in range(h_start, h_end):
                    for x in range(w_start, w_end):
                        val = feature_map[y][x]
                        if val > max_val:
                            max_val = val

                row.append(max_val)

            pooled.append(row)

        results.append(pooled)

    return results