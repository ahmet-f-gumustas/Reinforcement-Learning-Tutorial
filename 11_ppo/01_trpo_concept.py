"""
01 - TRPO Concept: Trust Region Policy Optimization

This script explains the motivation behind TRPO and the trust region
concept that PPO is built upon.

Demonstrates:
- Why large policy updates can be catastrophic
- The policy improvement guarantee
- KL divergence as a distance measure between policies
- Trust region constraint intuition
- Monotonic improvement theory
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# =============================================================================
# Visualization Helpers
# =============================================================================

def demonstrate_catastrophic_update():
    """Show how large policy updates can destroy performance."""
    print("=" * 60)
    print("THE PROBLEM: CATASTROPHIC POLICY UPDATES")
    print("=" * 60)

    print("""
    In A2C/REINFORCE, gradient ascent updates the policy:
        θ_new = θ_old + α ∇J(θ)

    Problem: The gradient is estimated from OLD data (π_old).
    If the update is too large, π_new may be very different from π_old,
    and performance can COLLAPSE.

    Example:
        Step 1: Policy learns to go RIGHT (good strategy)
        Step 2: Large gradient update → policy now goes LEFT
        Step 3: Environment gives negative reward
        Step 4: Another large update → policy is now random
        Result: Performance collapses and may never recover!

    Why is this hard to fix with learning rate tuning?
        - Too small α → very slow learning
        - Too large α → catastrophic collapse
        - Optimal α varies throughout training and per environment
    """)

    # Visualize optimization landscape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    theta = np.linspace(-3, 3, 300)

    # Simulated reward landscape (non-convex)
    reward = np.exp(-0.5 * theta**2) * np.sin(3 * theta) + 0.5 * np.exp(-0.5 * (theta - 1.5)**2)

    ax = axes[0]
    ax.plot(theta, reward, 'b-', linewidth=2, label='True Reward J(θ)')

    # A2C with large step
    start = -1.5
    grad_at_start = 0.8
    large_step = 2.5
    ax.annotate('', xy=(start + large_step, 0.1),
                xytext=(start, reward[np.argmin(np.abs(theta - start))]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.scatter([start], [reward[np.argmin(np.abs(theta - start))]], color='green',
               s=100, zorder=5, label='θ_old (good)')
    end_large = start + large_step
    ax.scatter([end_large], [reward[np.argmin(np.abs(theta - end_large))]], color='red',
               s=100, zorder=5, label='θ_new large step (bad)')
    ax.set_xlabel('θ (policy parameter)')
    ax.set_ylabel('J(θ)')
    ax.set_title('Large Update → Performance Collapse')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(theta, reward, 'b-', linewidth=2, label='True Reward J(θ)')

    # Trust region approach
    trust_radius = 0.5
    theta_circle = np.linspace(start - trust_radius, start + trust_radius, 100)
    reward_approx = reward[np.argmin(np.abs(theta - start))] + grad_at_start * (theta_circle - start)

    ax.fill_betweenx([-0.5, 1.5], start - trust_radius, start + trust_radius,
                     alpha=0.2, color='green', label='Trust Region')
    ax.plot(theta_circle, reward_approx, 'g--', linewidth=2, label='Local approx')
    end_small = start + trust_radius * 0.8
    ax.scatter([start], [reward[np.argmin(np.abs(theta - start))]], color='green',
               s=100, zorder=5, label='θ_old')
    ax.scatter([end_small], [reward[np.argmin(np.abs(theta - end_small))]], color='steelblue',
               s=100, zorder=5, label='θ_new (trust region)')
    ax.set_xlabel('θ (policy parameter)')
    ax.set_ylabel('J(θ)')
    ax.set_title('Trust Region → Safe Update')
    ax.legend()
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_ppo/trpo_concept.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/trpo_concept.png'")
    plt.close()


def demonstrate_kl_divergence():
    """Show KL divergence as a policy distance measure."""
    print("\n" + "=" * 60)
    print("KL DIVERGENCE: MEASURING POLICY DISTANCE")
    print("=" * 60)

    print("""
    KL Divergence measures how different two distributions are:

    KL(π_old || π_new) = Σ_a π_old(a|s) × log[π_old(a|s) / π_new(a|s)]

    Properties:
    - KL ≥ 0 always
    - KL = 0 if and only if π_old = π_new
    - KL is NOT symmetric: KL(p||q) ≠ KL(q||p)

    TRPO constraint:
        maximize  J(θ)
        subject to  E_s[KL(π_old(·|s) || π_new(·|s))] ≤ δ

    This ensures we don't change the policy too much per update.
    """)

    # Visualize KL divergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.linspace(-5, 5, 200)

    scenarios = [
        ("Same policies\nKL ≈ 0", 0.0, 1.0, 0.0, 1.0),
        ("Slightly different\nKL = small", 0.0, 1.0, 0.5, 1.2),
        ("Very different\nKL = large", 0.0, 1.0, 2.0, 0.5),
    ]

    for ax, (title, mu1, sigma1, mu2, sigma2) in zip(axes, scenarios):
        p = np.exp(-0.5 * ((x - mu1) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))
        q = np.exp(-0.5 * ((x - mu2) / sigma2)**2) / (sigma2 * np.sqrt(2 * np.pi))

        # Compute KL(p||q) numerically
        mask = (p > 1e-10) & (q > 1e-10)
        dx = x[1] - x[0]
        kl = np.sum(p[mask] * np.log(p[mask] / q[mask])) * dx

        ax.plot(x, p, 'b-', linewidth=2, label='π_old')
        ax.plot(x, q, 'r--', linewidth=2, label='π_new')
        ax.fill_between(x, 0, np.minimum(p, q), alpha=0.3, color='purple', label='Overlap')
        ax.set_title(f'{title}\nKL = {kl:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Action')
        ax.set_ylabel('Probability')

    plt.suptitle('KL Divergence Between Policies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('11_ppo/kl_divergence.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/kl_divergence.png'")
    plt.close()

    # Numerical example
    print("KL Divergence Example:")
    pi_old = torch.FloatTensor([0.5, 0.3, 0.2])
    pi_new1 = torch.FloatTensor([0.5, 0.3, 0.2])   # Same
    pi_new2 = torch.FloatTensor([0.4, 0.35, 0.25])  # Similar
    pi_new3 = torch.FloatTensor([0.1, 0.1, 0.8])    # Very different

    for name, pi_new in [("Same", pi_new1), ("Similar", pi_new2), ("Different", pi_new3)]:
        kl = (pi_old * (pi_old.log() - pi_new.log())).sum()
        print(f"  π_new ({name:10s}): {pi_new.numpy()} → KL = {kl:.4f}")


def demonstrate_importance_sampling():
    """Explain the importance sampling ratio used in TRPO/PPO."""
    print("\n" + "=" * 60)
    print("IMPORTANCE SAMPLING: REUSING OLD EXPERIENCE")
    print("=" * 60)

    print("""
    TRPO and PPO can use experience from π_old to estimate
    gradients for π_new using importance sampling:

    E_{a~π_new}[f(a)] = E_{a~π_old}[π_new(a|s)/π_old(a|s) × f(a)]

    The ratio r_t(θ) = π_new(a_t|s_t) / π_old(a_t|s_t)
    - r = 1: same policy (exact sample)
    - r > 1: π_new assigns higher probability to this action
    - r < 1: π_new assigns lower probability

    Surrogate objective:
        L^CPI(θ) = E_t[ r_t(θ) × A_t ]

    This is the "Conservative Policy Iteration" objective.
    It's a good approximation of J(θ) when π_new ≈ π_old.

    TRPO: maximize L^CPI(θ)  subject to KL ≤ δ
    PPO:  add clipping to constrain r_t implicitly
    """)

    # Visualize importance sampling ratio
    fig, ax = plt.subplots(figsize=(10, 5))

    ratios = np.linspace(0, 3, 300)

    # Unclipped surrogate (L_CPI)
    advantage_pos = 1.0
    advantage_neg = -1.0

    ax.plot(ratios, ratios * advantage_pos, 'b-', linewidth=2,
            label='r × A (A > 0, unclipped)')
    ax.plot(ratios, ratios * advantage_neg, 'r-', linewidth=2,
            label='r × A (A < 0, unclipped)')

    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='r = 1 (π_old = π_new)')
    ax.set_xlabel('Importance ratio r = π_new / π_old')
    ax.set_ylabel('L_CPI contribution')
    ax.set_title('Importance Sampling Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('11_ppo/importance_sampling.png', dpi=150, bbox_inches='tight')
    print("Plot saved to '11_ppo/importance_sampling.png'")
    plt.close()


def trpo_algorithm_overview():
    """Show the TRPO algorithm."""
    print("\n" + "=" * 60)
    print("TRPO ALGORITHM OVERVIEW")
    print("=" * 60)

    print("""
    Algorithm: TRPO (Schulman et al., 2015)

    For each iteration:
        1. Collect trajectories using current policy π_old
        2. Compute advantages A_t using GAE
        3. Solve constrained optimization:

            maximize  L^CPI(θ) = E_t[r_t(θ) × A_t]
            subject to  KL(π_old || π_new) ≤ δ

        4. Use conjugate gradient + line search to solve efficiently

    Problem with TRPO:
    ✗ Complex second-order optimization (conjugate gradients)
    ✗ Expensive to compute (Fisher information matrix)
    ✗ Hard to implement correctly
    ✗ Doesn't work well with parameter sharing

    Solution: PPO - a simpler approximation that works just as well!

    PPO's key idea: Instead of HARD constraint (KL ≤ δ),
    use SOFT constraint via clipping of the ratio r_t(θ).
    """)


def main():
    print("\n" + "=" * 60)
    print("WEEK 11 - LESSON 1: TRPO CONCEPT")
    print("Trust Region Policy Optimization Background")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Show catastrophic update problem
    demonstrate_catastrophic_update()

    # 2. KL divergence as distance measure
    demonstrate_kl_divergence()

    # 3. Importance sampling
    demonstrate_importance_sampling()

    # 4. TRPO overview
    trpo_algorithm_overview()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. Large policy updates can destroy performance catastrophically")
    print("2. TRPO constrains the KL divergence between old and new policy")
    print("3. KL(π_old || π_new) ≤ δ ensures safe policy updates")
    print("4. Importance ratio r_t = π_new/π_old enables reusing old data")
    print("5. TRPO is theoretically sound but computationally expensive")
    print("6. PPO approximates TRPO with simpler clipping mechanism")
    print("\nNext: PPO-Clip - the simple and effective solution!")
    print("=" * 60)


if __name__ == "__main__":
    main()
