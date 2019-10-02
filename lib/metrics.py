

def wasserstein_distance(t1, t2):
    n = min(len(t1), len(t2))
    m = max(len(t1), len(t2))
    if len(t1) == 0:
        T = max(t2)
    elif len(t2) == 0:
        T = max(t1)
    else: 
        T = max(max(t1), max(t2))
    val = sum(abs(t1[:n] - t2[:n])) + (m - n) * T - sum(t1[n:]) - sum(t2[n:])
    return val
