"""
Geodesic computation on the estimated Riemannian manifold.

Solves the geodesic equation using 4th-order Runge-Kutta:

    d²γ^k/dt² + Γ^k_{ij}(γ(t)) dγ^i/dt dγ^j/dt = 0

Also provides exponential/logarithmic maps and geodesic distance
computation via the shooting method.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from topo_llm.riemannian.connection import ChristoffelEstimator
from topo_llm.riemannian.metric import MetricTensorEstimator
from topo_llm.types import GeodesicResult

logger = logging.getLogger(__name__)


class GeodesicSolver:
    """Solve geodesic equations on the estimated manifold.

    Parameters
    ----------
    metric_estimator : MetricTensorEstimator
        Fitted metric tensor estimator.
    christoffel_estimator : ChristoffelEstimator
        Christoffel symbol estimator.
    dt : float
        Integration time step for RK4.
    max_steps : int
        Maximum number of integration steps.

    Examples
    --------
    >>> solver = GeodesicSolver(metric_est, christoffel_est)
    >>> result = solver.solve(start_idx=0, initial_velocity=v0)
    >>> print(f"Arc length: {result.arc_length:.4f}")
    """

    def __init__(
        self,
        metric_estimator: MetricTensorEstimator,
        christoffel_estimator: ChristoffelEstimator,
        dt: float = 0.01,
        max_steps: int = 1000,
        seed: int = 42,
    ) -> None:
        self.metric = metric_estimator
        self.christoffel = christoffel_estimator
        self.dt = dt
        self.max_steps = max_steps
        self._rng_seed = seed

    def _geodesic_acceleration(
        self,
        position_ambient: np.ndarray,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Compute geodesic acceleration: -Γ^k_{ij} v^i v^j.

        Parameters
        ----------
        position_ambient : np.ndarray
            Current position in ambient space, shape ``(D,)``.
        velocity : np.ndarray
            Current velocity in tangent space, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Acceleration in tangent space, shape ``(m,)``.
        """
        gamma = self.christoffel.compute_at_point(position_ambient)
        m = len(velocity)

        # a^k = -Γ^k_{ij} v^i v^j
        accel = np.zeros(m)
        for k in range(m):
            for i in range(m):
                for j in range(m):
                    accel[k] -= gamma[k, i, j] * velocity[i] * velocity[j]

        return accel

    def solve(
        self,
        start_idx: int,
        initial_velocity: np.ndarray,
    ) -> GeodesicResult:
        """Solve the geodesic equation from a starting point.

        Uses 4th-order Runge-Kutta integration of the first-order system:
            dγ^k/dt = v^k
            dv^k/dt = -Γ^k_{ij}(γ) v^i v^j

        Parameters
        ----------
        start_idx : int
            Index of the starting point in the fitted point cloud.
        initial_velocity : np.ndarray
            Initial velocity in tangent space at the start point,
            shape ``(m,)``.

        Returns
        -------
        GeodesicResult
            Contains the path in both tangent and ambient coordinates,
            velocities, arc length, and step count.

        Notes
        -----
        Complexity is O(max_steps * d^2) where d is the intrinsic
        dimensionality, due to Christoffel symbol interpolation at each
        RK4 step.
        """
        m = self.metric.intrinsic_dim_
        dt = self.dt

        # Starting position and velocity
        x_ambient = self.metric.point_cloud_[start_idx].copy()

        v = initial_velocity.copy()

        # Storage
        tangent_path = [np.zeros(m)]  # tangent displacement from start
        ambient_path = [x_ambient.copy()]
        velocities = [v.copy()]

        tangent_pos = np.zeros(m)
        arc_length = 0.0

        for step in range(self.max_steps):
            # RK4 integration
            # State: (x_ambient, v)
            # dx/dt = T @ v (lift velocity to ambient)
            # dv/dt = -Γ^k_{ij} v^i v^j

            # k1
            a1 = self._geodesic_acceleration(x_ambient, v)
            T_current = self.metric.interpolate_tangent_basis(x_ambient)
            dx1 = T_current @ v

            # k2
            x2 = x_ambient + 0.5 * dt * dx1
            v2 = v + 0.5 * dt * a1
            a2 = self._geodesic_acceleration(x2, v2)
            T2 = self.metric.interpolate_tangent_basis(x2)
            dx2 = T2 @ v2

            # k3
            x3 = x_ambient + 0.5 * dt * dx2
            v3 = v + 0.5 * dt * a2
            a3 = self._geodesic_acceleration(x3, v3)
            T3 = self.metric.interpolate_tangent_basis(x3)
            dx3 = T3 @ v3

            # k4
            x4 = x_ambient + dt * dx3
            v4 = v + dt * a3
            a4 = self._geodesic_acceleration(x4, v4)
            T4 = self.metric.interpolate_tangent_basis(x4)
            dx4 = T4 @ v4

            # Update
            x_ambient = x_ambient + (dt / 6.0) * (dx1 + 2 * dx2 + 2 * dx3 + dx4)
            v = v + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)

            # Update tangent position
            tangent_pos = tangent_pos + dt * v

            # Arc length increment (use metric)
            g_local = self.metric.interpolate_metric(x_ambient)
            ds = np.sqrt(max(v @ g_local @ v, 0.0)) * dt
            arc_length += ds

            # Store
            tangent_path.append(tangent_pos.copy())
            ambient_path.append(x_ambient.copy())
            velocities.append(v.copy())

            # Check for divergence
            if not np.all(np.isfinite(x_ambient)):
                logger.warning("Geodesic diverged at step %d", step)
                break

        return GeodesicResult(
            tangent_path=np.array(tangent_path),
            ambient_path=np.array(ambient_path),
            velocities=np.array(velocities),
            arc_length=float(arc_length),
            n_steps=len(tangent_path),
            converged=np.all(np.isfinite(x_ambient)),
        )

    def exponential_map(
        self,
        idx: int,
        tangent_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the exponential map: Exp_x(v) = γ(1).

        Maps a tangent vector to a point on the manifold by following
        the geodesic from x with initial velocity v for unit time.

        Parameters
        ----------
        idx : int
            Index of the base point.
        tangent_vector : np.ndarray
            Tangent vector at the base point, shape ``(m,)``.

        Returns
        -------
        np.ndarray
            Point in ambient space, shape ``(D,)``.
        """
        # Scale dt so that we integrate from 0 to 1
        original_dt = self.dt
        original_max = self.max_steps
        n_steps = max(int(1.0 / self.dt), 10)
        self.dt = 1.0 / n_steps
        self.max_steps = n_steps

        result = self.solve(idx, tangent_vector)

        # Restore
        self.dt = original_dt
        self.max_steps = original_max

        return result.ambient_path[-1]

    def logarithmic_map(
        self,
        idx_from: int,
        idx_to: int,
    ) -> np.ndarray:
        """Compute the logarithmic map: Log_x(y).

        Finds the tangent vector v at x such that Exp_x(v) = y.
        Uses optimization to minimize ||Exp_x(v) - y||².

        Parameters
        ----------
        idx_from : int
            Index of the source point x.
        idx_to : int
            Index of the target point y.

        Returns
        -------
        np.ndarray
            Tangent vector at idx_from, shape ``(m,)``.
        """
        y_target = self.metric.point_cloud_[idx_to]
        m = self.metric.intrinsic_dim_

        # Initial guess: project the straight-line direction
        T_from = self.metric.tangent_bases_[idx_from]
        x_from = self.metric.point_cloud_[idx_from]
        direction = y_target - x_from
        v0 = T_from.T @ direction  # project to tangent space

        def objective(v_flat: np.ndarray) -> float:
            v = v_flat.reshape(m)
            y_pred = self.exponential_map(idx_from, v)
            return float(np.sum((y_pred - y_target) ** 2))

        result = minimize(
            objective,
            v0.flatten(),
            method="L-BFGS-B",
            options={"maxiter": 100, "ftol": 1e-8},
        )

        return result.x.reshape(m)

    def geodesic_distance(
        self,
        idx_a: int,
        idx_b: int,
        n_shooting: int = 10,
    ) -> float:
        """Approximate geodesic distance between two points.

        Uses the shooting method: tries multiple initial velocities
        aimed toward the target, solves the geodesic, and picks
        the one that gets closest.

        Parameters
        ----------
        idx_a : int
            Index of the first point.
        idx_b : int
            Index of the second point.
        n_shooting : int
            Number of initial velocity perturbations to try.

        Returns
        -------
        float
            Approximate geodesic distance.
        """
        m = self.metric.intrinsic_dim_
        rng = np.random.default_rng(self._rng_seed)

        x_a = self.metric.point_cloud_[idx_a]
        x_b = self.metric.point_cloud_[idx_b]
        T_a = self.metric.tangent_bases_[idx_a]

        # Base direction in tangent space
        direction = T_a.T @ (x_b - x_a)
        base_speed = np.linalg.norm(direction)
        if base_speed < 1e-10:
            return 0.0

        best_distance = float("inf")
        best_arc = float("inf")

        for trial in range(n_shooting):
            if trial == 0:
                v0 = direction.copy()
            else:
                # Perturb the initial velocity
                perturbation = rng.standard_normal(m) * base_speed * 0.1
                v0 = direction + perturbation

            result = self.solve(idx_a, v0)
            endpoint = result.ambient_path[-1]
            endpoint_error = np.linalg.norm(endpoint - x_b)

            if endpoint_error < best_distance:
                best_distance = endpoint_error
                best_arc = result.arc_length

        return best_arc

    def geodesic_distance_matrix(
        self,
        indices: list[int] | None = None,
        n_shooting: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute pairwise geodesic distance matrix.

        Parameters
        ----------
        indices : list[int] | None
            Subset of point indices. None uses all points (expensive).
        n_shooting : int
            Shooting attempts per pair.
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape ``(n, n)``.
        """
        if indices is None:
            indices = list(range(len(self.metric.metric_tensors_)))

        n = len(indices)
        D = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        logger.info("Computing %d geodesic distances...", total_pairs)

        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(total=total_pairs, desc="Geodesic distances", unit="pair")
        else:
            pbar = None

        for i in range(n):
            for j in range(i + 1, n):
                d = self.geodesic_distance(indices[i], indices[j], n_shooting)
                D[i, j] = d
                D[j, i] = d
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        return D
