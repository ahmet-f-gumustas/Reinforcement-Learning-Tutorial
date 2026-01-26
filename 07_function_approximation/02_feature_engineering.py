"""
02 - Feature Engineering

Different feature representations for function approximation.

Demonstrates:
- State aggregation
- Polynomial features
- Fourier basis
- Tile coding
- Radial Basis Functions (RBF)
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt


# =============================================================================
# FEATURE EXTRACTION METHODS
# =============================================================================

class StateAggregation:
    """
    State aggregation: Group states into bins.

    Each bin is a binary feature (1 if state in bin, 0 otherwise).
    """

    def __init__(self, n_bins: int, state_range: Tuple[float, float] = (0, 1)):
        self.n_bins = n_bins
        self.state_min, self.state_max = state_range
        self.bin_width = (self.state_max - self.state_min) / n_bins

    def extract(self, state: float) -> np.ndarray:
        """Extract one-hot feature vector."""
        features = np.zeros(self.n_bins)
        bin_idx = int((state - self.state_min) / self.bin_width)
        bin_idx = max(0, min(self.n_bins - 1, bin_idx))
        features[bin_idx] = 1.0
        return features

    @property
    def n_features(self) -> int:
        return self.n_bins


class PolynomialFeatures:
    """
    Polynomial features: [1, s, s², s³, ..., s^degree]
    """

    def __init__(self, degree: int):
        self.degree = degree

    def extract(self, state: float) -> np.ndarray:
        return np.array([state ** i for i in range(self.degree + 1)])

    @property
    def n_features(self) -> int:
        return self.degree + 1


class FourierBasis:
    """
    Fourier basis features: cos(πi·s) for i = 0, 1, ..., order

    Works well for smooth value functions.
    """

    def __init__(self, order: int):
        self.order = order

    def extract(self, state: float) -> np.ndarray:
        """State should be in [0, 1]."""
        return np.array([np.cos(np.pi * i * state) for i in range(self.order + 1)])

    @property
    def n_features(self) -> int:
        return self.order + 1


class TileCoding:
    """
    Tile coding: Multiple overlapping tilings.

    Each tiling divides the state space into tiles.
    Active features are the tiles containing the current state.
    """

    def __init__(self, n_tilings: int, n_tiles: int,
                 state_range: Tuple[float, float] = (0, 1)):
        """
        Args:
            n_tilings: Number of overlapping tilings
            n_tiles: Number of tiles per tiling
            state_range: (min, max) of state space
        """
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.state_min, self.state_max = state_range

        self.tile_width = (self.state_max - self.state_min) / n_tiles

        # Offset each tiling
        self.offsets = np.linspace(0, self.tile_width, n_tilings, endpoint=False)

    def extract(self, state: float) -> np.ndarray:
        """Extract binary feature vector (one-hot per tiling)."""
        features = np.zeros(self.n_tilings * self.n_tiles)

        for tiling_idx, offset in enumerate(self.offsets):
            # Shift state by offset
            shifted_state = state - self.state_min - offset
            tile_idx = int(shifted_state / self.tile_width)
            tile_idx = max(0, min(self.n_tiles - 1, tile_idx))

            # Set feature
            feature_idx = tiling_idx * self.n_tiles + tile_idx
            features[feature_idx] = 1.0

        return features

    @property
    def n_features(self) -> int:
        return self.n_tilings * self.n_tiles


class TileCoding2D:
    """
    2D Tile coding for two-dimensional state spaces.
    """

    def __init__(self, n_tilings: int, n_tiles: Tuple[int, int],
                 state_ranges: Tuple[Tuple[float, float], Tuple[float, float]]):
        """
        Args:
            n_tilings: Number of overlapping tilings
            n_tiles: (tiles_dim1, tiles_dim2) per tiling
            state_ranges: ((min1, max1), (min2, max2))
        """
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.state_ranges = state_ranges

        self.tile_widths = [
            (state_ranges[i][1] - state_ranges[i][0]) / n_tiles[i]
            for i in range(2)
        ]

        # Random offsets for each tiling
        np.random.seed(42)
        self.offsets = [
            (np.random.random() * self.tile_widths[0],
             np.random.random() * self.tile_widths[1])
            for _ in range(n_tilings)
        ]

    def extract(self, state: Tuple[float, float]) -> np.ndarray:
        """Extract binary feature vector."""
        features_per_tiling = self.n_tiles[0] * self.n_tiles[1]
        features = np.zeros(self.n_tilings * features_per_tiling)

        for tiling_idx, offset in enumerate(self.offsets):
            tile_indices = []
            for dim in range(2):
                shifted = state[dim] - self.state_ranges[dim][0] - offset[dim]
                tile_idx = int(shifted / self.tile_widths[dim])
                tile_idx = max(0, min(self.n_tiles[dim] - 1, tile_idx))
                tile_indices.append(tile_idx)

            # Convert 2D index to 1D
            flat_idx = tile_indices[0] * self.n_tiles[1] + tile_indices[1]
            feature_idx = tiling_idx * features_per_tiling + flat_idx
            features[feature_idx] = 1.0

        return features

    @property
    def n_features(self) -> int:
        return self.n_tilings * self.n_tiles[0] * self.n_tiles[1]


class RBFFeatures:
    """
    Radial Basis Function features.

    x_i(s) = exp(-||s - c_i||² / 2σ²)
    """

    def __init__(self, centers: np.ndarray, sigma: float = 0.2):
        """
        Args:
            centers: Array of RBF center positions
            sigma: Width of Gaussian bumps
        """
        self.centers = np.array(centers)
        self.sigma = sigma

    def extract(self, state: float) -> np.ndarray:
        return np.exp(-((state - self.centers) ** 2) / (2 * self.sigma ** 2))

    @property
    def n_features(self) -> int:
        return len(self.centers)


class RBFFeatures2D:
    """2D Radial Basis Function features."""

    def __init__(self, centers: List[Tuple[float, float]], sigma: float = 0.3):
        self.centers = np.array(centers)
        self.sigma = sigma

    def extract(self, state: Tuple[float, float]) -> np.ndarray:
        state = np.array(state)
        distances_sq = np.sum((self.centers - state) ** 2, axis=1)
        return np.exp(-distances_sq / (2 * self.sigma ** 2))

    @property
    def n_features(self) -> int:
        return len(self.centers)


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_1d_features():
    """Visualize different 1D feature representations."""
    print("\n" + "="*60)
    print("VISUALIZING 1D FEATURES")
    print("="*60)

    states = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Representations for 1D State', fontsize=16)

    # 1. State Aggregation
    ax = axes[0, 0]
    aggregator = StateAggregation(n_bins=5)
    for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
        features = aggregator.extract(s)
        ax.bar(range(5), features, alpha=0.7, label=f's={s}')
    ax.set_xlabel('Bin', fontsize=11)
    ax.set_ylabel('Feature Value', fontsize=11)
    ax.set_title('State Aggregation (5 bins)', fontsize=12)
    ax.legend(fontsize=9)

    # 2. Polynomial Features
    ax = axes[0, 1]
    poly = PolynomialFeatures(degree=4)
    for i in range(5):
        feature_values = [poly.extract(s)[i] for s in states]
        ax.plot(states, feature_values, linewidth=2, label=f's^{i}')
    ax.set_xlabel('State', fontsize=11)
    ax.set_ylabel('Feature Value', fontsize=11)
    ax.set_title('Polynomial Features (degree 4)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Fourier Basis
    ax = axes[0, 2]
    fourier = FourierBasis(order=4)
    for i in range(5):
        feature_values = [fourier.extract(s)[i] for s in states]
        ax.plot(states, feature_values, linewidth=2, label=f'cos({i}πs)')
    ax.set_xlabel('State', fontsize=11)
    ax.set_ylabel('Feature Value', fontsize=11)
    ax.set_title('Fourier Basis (order 4)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. Tile Coding
    ax = axes[1, 0]
    tile_coder = TileCoding(n_tilings=3, n_tiles=4)
    for tiling in range(3):
        active_tiles = []
        for s in states:
            features = tile_coder.extract(s)
            active = np.where(features[tiling*4:(tiling+1)*4] == 1)[0]
            if len(active) > 0:
                active_tiles.append(active[0] + tiling * 0.1)
            else:
                active_tiles.append(np.nan)
        ax.plot(states, active_tiles, linewidth=2, label=f'Tiling {tiling+1}')
    ax.set_xlabel('State', fontsize=11)
    ax.set_ylabel('Active Tile', fontsize=11)
    ax.set_title('Tile Coding (3 tilings × 4 tiles)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. RBF Features
    ax = axes[1, 1]
    centers = np.linspace(0, 1, 5)
    rbf = RBFFeatures(centers, sigma=0.15)
    for i, c in enumerate(centers):
        feature_values = [rbf.extract(s)[i] for s in states]
        ax.plot(states, feature_values, linewidth=2, label=f'c={c:.1f}')
    ax.set_xlabel('State', fontsize=11)
    ax.set_ylabel('Feature Value', fontsize=11)
    ax.set_title('RBF Features (5 centers)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6. Feature Count Comparison
    ax = axes[1, 2]
    methods = ['State Agg\n(10 bins)', 'Polynomial\n(deg 5)', 'Fourier\n(order 5)',
               'Tile Coding\n(4×4)', 'RBF\n(10 centers)']
    n_features = [10, 6, 6, 16, 10]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))
    ax.bar(methods, n_features, color=colors)
    ax.set_ylabel('Number of Features', fontsize=11)
    ax.set_title('Feature Count Comparison', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/feature_types.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved visualization to feature_types.png")
    plt.close()


def visualize_2d_features():
    """Visualize 2D feature representations."""
    print("\n" + "="*60)
    print("VISUALIZING 2D FEATURES")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('2D Feature Representations', fontsize=16)

    # Create grid of states
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)

    # 1. 2D Tile Coding - show active features for one point
    ax = axes[0]
    tile_coder = TileCoding2D(n_tilings=4, n_tiles=(4, 4),
                               state_ranges=((0, 1), (0, 1)))

    # Show which tiles are active for point (0.6, 0.4)
    test_point = (0.6, 0.4)
    features = tile_coder.extract(test_point)

    # Create heatmap of feature activations
    activation_grid = np.zeros((4, 4))
    for tiling in range(4):
        for i in range(4):
            for j in range(4):
                idx = tiling * 16 + i * 4 + j
                if features[idx] > 0:
                    activation_grid[i, j] += 0.25

    im = ax.imshow(activation_grid, cmap='Blues', extent=[0, 1, 0, 1], origin='lower')
    ax.plot(test_point[0], test_point[1], 'r*', markersize=15, label='Test point')
    ax.set_xlabel('State dim 1', fontsize=11)
    ax.set_ylabel('State dim 2', fontsize=11)
    ax.set_title('2D Tile Coding Activation', fontsize=12)
    ax.legend()
    plt.colorbar(im, ax=ax)

    # 2. 2D RBF Features
    ax = axes[1]
    centers = [(i, j) for i in np.linspace(0.2, 0.8, 3) for j in np.linspace(0.2, 0.8, 3)]
    rbf = RBFFeatures2D(centers, sigma=0.2)

    # Sum of all RBF activations
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            features = rbf.extract((X[i, j], Y[i, j]))
            Z[i, j] = np.sum(features)

    im = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
    for c in centers:
        ax.plot(c[0], c[1], 'r+', markersize=10, markeredgewidth=2)
    ax.set_xlabel('State dim 1', fontsize=11)
    ax.set_ylabel('State dim 2', fontsize=11)
    ax.set_title('2D RBF Coverage', fontsize=12)
    plt.colorbar(im, ax=ax)

    # 3. Single RBF feature
    ax = axes[2]
    center = (0.5, 0.5)
    Z_single = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            state = np.array([X[i, j], Y[i, j]])
            Z_single[i, j] = np.exp(-np.sum((state - np.array(center))**2) / (2 * 0.2**2))

    im = ax.contourf(X, Y, Z_single, levels=20, cmap='hot')
    ax.plot(center[0], center[1], 'g*', markersize=15)
    ax.set_xlabel('State dim 1', fontsize=11)
    ax.set_ylabel('State dim 2', fontsize=11)
    ax.set_title('Single RBF Feature', fontsize=12)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/feature_types_2d.png',
                dpi=150, bbox_inches='tight')
    print("\nSaved 2D visualization to feature_types_2d.png")
    plt.close()


def demonstrate_tile_coding_detail():
    """Show tile coding in detail."""
    print("\n" + "="*60)
    print("TILE CODING IN DETAIL")
    print("="*60)

    print("""
    Tile Coding Explained:
    ----------------------
    1. Divide state space into tiles (grid)
    2. Create multiple tilings with different offsets
    3. For each tiling, only one tile is active (binary feature)
    4. Total features = n_tilings × n_tiles

    Benefits:
    - Binary features (fast computation)
    - Good generalization (nearby states share tiles)
    - Adjustable resolution (more tilings = finer resolution)

    Example: 3 tilings, 4 tiles each
    """)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Tile Coding: 3 Tilings with Offsets', fontsize=14)

    tile_coder = TileCoding(n_tilings=3, n_tiles=4)
    test_state = 0.35

    for tiling_idx, ax in enumerate(axes):
        # Draw tiles
        offset = tile_coder.offsets[tiling_idx]
        for i in range(4):
            left = i * tile_coder.tile_width + offset
            ax.axvline(x=left, color='gray', linestyle='--', alpha=0.5)

        # Get active tile
        features = tile_coder.extract(test_state)
        active_idx = np.where(features[tiling_idx*4:(tiling_idx+1)*4] == 1)[0][0]

        # Highlight active tile
        left = active_idx * tile_coder.tile_width + offset
        ax.axvspan(left, left + tile_coder.tile_width, alpha=0.3, color='blue')

        # Mark state
        ax.axvline(x=test_state, color='red', linewidth=2)
        ax.plot(test_state, 0.5, 'r*', markersize=15)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('State', fontsize=11)
        ax.set_title(f'Tiling {tiling_idx + 1} (offset={offset:.3f})', fontsize=12)
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('/home/red/git-projects/Reinforcement-Learning-Tutorial/07_function_approximation/tile_coding_detail.png',
                dpi=150, bbox_inches='tight')
    print(f"\nState = {test_state}")
    print(f"Active features: {np.where(tile_coder.extract(test_state) == 1)[0]}")
    print("\nSaved tile coding detail to tile_coding_detail.png")
    plt.close()


def main():
    """Main demonstration."""
    print("="*60)
    print("FEATURE ENGINEERING FOR FUNCTION APPROXIMATION")
    print("="*60)

    print("""
    Feature engineering is CRITICAL for linear function approximation.

    The features you choose determine:
    - What value functions can be represented
    - How well the agent generalizes
    - Computational efficiency

    Common approaches:
    1. State Aggregation - Simple binning
    2. Polynomial - Good for smooth functions
    3. Fourier Basis - Good for periodic patterns
    4. Tile Coding - Binary, good generalization
    5. RBF - Smooth, local generalization
    """)

    # Run demonstrations
    visualize_1d_features()
    visualize_2d_features()
    demonstrate_tile_coding_detail()

    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  - Features define what can be learned")
    print("  - Tile coding: binary, good generalization")
    print("  - RBF: smooth, local influence")
    print("  - Polynomial: global, can be unstable")
    print("  - Choose features based on problem structure")


if __name__ == "__main__":
    main()
