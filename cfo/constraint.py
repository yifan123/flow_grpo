import numpy as np

def is_safe(x) -> bool:
    """
    Returns True if sample x is inside the feasible region (g(x) <= 0).
    Replace this with your actual constraint for your experiment.
    Default: triangle with vertices (0,0), (1,0), (0.5,1)
    """
    x = np.array(x, dtype=float)

    v0 = np.array([0.5, 1.0])
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])

    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])

    d1 = sign(x, v0, v1)
    d2 = sign(x, v1, v2)
    d3 = sign(x, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)