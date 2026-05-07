import numpy as np

def compute_lambda(rewards, constraint_flags, alpha, beta_cfo):
    """
    Stage 1 of CFO: compute optimal dual variable lambda_hat.

    Args:
        rewards: list or array of r(x_i) floats for each sample
        constraint_flags: list or array of bools — True means safe (g(x) <= 0)
        alpha: KL penalty weight (= config.train.beta in the repo)
        beta_cfo: violation tolerance (e.g. 0.1 means allow 10% violations)

    Returns:
        lambda_hat: float, the optimal penalty weight for safe samples
    """
    rewards = np.array(rewards, dtype=np.float64)
    flags   = np.array(constraint_flags, dtype=bool)

    exp_r = np.exp(rewards / alpha)

    A_hat = float(exp_r[~flags].sum())  # sum over VIOLATING samples
    B_hat = float(exp_r[ flags].sum())  # sum over SAFE samples

    if B_hat < 1e-10:
        # all samples violate the constraint
        return float(alpha * 10.0)

    raw = np.log(((1 - beta_cfo) * A_hat) / (beta_cfo * B_hat + 1e-10))
    return float(alpha * max(0.0, raw))
