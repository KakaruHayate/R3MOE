import torch


def calc_outlier_ratio(y_true, y_pred):
    residual = (y_pred - y_true).abs()
    outlier_ratio = (residual > 2*y_true.std()).float().mean()

    return outlier_ratio


def calc_correlation(y_true, y_pred):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    
    covariance = torch.sum((y_true - mean_true) * (y_pred - mean_pred))
    
    std_true = torch.sqrt(torch.sum((y_true - mean_true) ** 2))
    std_pred = torch.sqrt(torch.sum((y_pred - mean_pred) ** 2))
    
    denominator = std_true * std_pred
    if denominator == 0:
        return 0.0
    else:
        correlation = covariance / denominator
    
    return correlation


def calc_r_squared(y_true, y_pred):
    # R-squared: r^2 = 1 - (SSE / SST)
    ss_res = torch.sum((y_pred - y_true) ** 2)
    mean_y_true = torch.mean(y_true)
    ss_total = torch.sum((y_true - mean_y_true) ** 2)
    r2 = 1 - (ss_res / ss_total) if ss_total != 0 else 0

    return r2


def calc_dtw_distance(series1, series2, normalized=True):
    """
    批量动态时间规整(DTW)距离计算
    Args:
        series1: (B, T1)
        series2: (B, T2)
        normalized: 是否按路径长度归一化
    Returns:
        dtw_dist: (B,)
    """
    device = series1.device
    B = series1.size(0)
    T1, T2 = series1.size(1), series2.size(1)
    
    # 计算距离矩阵
    series1 = series1.unsqueeze(2)  # (B, T1, 1)
    series2 = series2.unsqueeze(1)  # (B, 1, T2)
    dist_matrix = torch.abs(series1 - series2)  # (B, T1, T2)
    
    # 初始化累积距离矩阵
    acc_dist = torch.full((B, T1+1, T2+1), torch.inf, device=device)
    acc_dist[:, 0, 0] = 0.0
    
    # 动态规划计算
    for i in range(T1):
        for j in range(T2):
            min_cost = torch.min(acc_dist[:, i, j+1], 
                               torch.min(acc_dist[:, i+1, j],
                                        acc_dist[:, i, j]))
            acc_dist[:, i+1, j+1] = dist_matrix[:, i, j] + min_cost
    
    dtw_dist = acc_dist[:, -1, -1]
    
    if normalized:
        # 计算最优路径长度
        path_length = torch.zeros(B, device=device)
        i, j = T1-1, T2-1
        while i > 0 or j > 0:
            path_length += 1
            prev = torch.argmin(torch.stack([
                acc_dist[:, i, j+1],
                acc_dist[:, i+1, j],
                acc_dist[:, i, j]
            ], dim=1), dim=1)
            
            i -= (prev == 0).long()
            j -= (prev == 1).long()
            i -= (prev == 2).long()
            j -= (prev == 2).long()
        
        dtw_dist /= path_length
    
    return dtw_dist.mean()  # 返回批次平均距离
