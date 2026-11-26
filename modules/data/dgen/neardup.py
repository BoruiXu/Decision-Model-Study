"""Near-duplicate groups (dgen neardup): two rows are near duplicates when their distance in the z-space (dgen.space)
is at most eps_ratio times the median nearest-neighbour distance of the rows; a group is a connected component of
that relation with at least two rows. Exact duplicates are always near duplicates (distance 0)."""
import numpy as np


def groups(Z, eps_ratio):
    """(groups, median nearest-neighbour distance, radius) of the rows of the z-space matrix Z: each group the row
    indices in row order, the groups ordered by their first row; ([], None, None) under two rows."""
    from sklearn.neighbors import NearestNeighbors
    n = len(Z)
    if n < 2:
        return [], None, None
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    dist, _ = nn.kneighbors(Z)
    median = float(np.median(dist[:, 1]))  # column 0 is the row itself (or an exact duplicate: distance 0 either way)
    radius = eps_ratio * median
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, near in enumerate(nn.radius_neighbors(Z, radius=radius, return_distance=False)):  # distance <= radius
        for j in near:
            a, b = find(i), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    return sorted((g for g in members.values() if len(g) > 1), key=lambda g: g[0]), median, radius
