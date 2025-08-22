"""Sampling utilities for power-law mass functions."""

from __future__ import annotations

import numpy as np


class OptimalSampler:
    """Deterministic sampler implementing the optimal-sampling recipe.

    The algorithm mirrors the one used in the GalIMF project: starting from the
    maximum mass, it iteratively determines the lower boundary that contains an
    integer number of objects. If the resulting mass interval is too narrow
    (smaller than ``resolution`` times the upper mass) several objects are
    grouped together until the interval is sufficiently wide.

    Parameters
    ----------
    mass_function : MassFunction
        Mass function to draw from.
    resolution : float, optional
        Minimum relative bin width; bins narrower than ``resolution`` times the
        upper mass will be merged by increasing the expected number of objects.
    mass_grid_index : float, optional
        Factor used when increasing the number of objects grouped into a bin
        (default 1.01, matching the reference implementation).
    """

    def __init__(self, mass_function, resolution=0.01, mass_grid_index=1.01):
        self.mf = mass_function
        self.resolution = resolution
        self.mass_grid_index = mass_grid_index
        self.edges: list[float] = []   # bin edges (descending order)
        self.counts: list[int] = []    # expected number of objects per bin
        self.masses: list[float] = []  # mean mass of each bin

    def _increase_count(self, m_high: float, n: int) -> tuple[float, int]:
        """Increase ``n`` until the bin width exceeds the resolution."""

        while True:
            n_new = int(round(n * self.mass_grid_index + 1))
            m_low = self.mf.inverse_integral(m_high, n_new)
            if m_high - m_low >= self.resolution * m_high:
                return m_low, n_new
            n = n_new

    def sample(self):
        """Perform optimal sampling and return ``(mass, count)`` pairs."""

        M_min, M_max = self.mf.bounds_sampled
        m_high = M_max  # current upper edge
        self.edges = [m_high]
        self.counts = []

        n = 1  # desired number of objects in the current bin
        while m_high > M_min:
            m_low = self.mf.inverse_integral(m_high, n)

            if m_low <= M_min:
                # Final bin reaches the minimum mass
                n_L = int(self.mf.integral_number(M_min, m_high))
                if n_L > 0:
                    self.edges.append(M_min)
                    self.counts.append(n_L)
                break

            if m_high - m_low < self.resolution * m_high:
                # Bin too narrow: increase object count and retry
                m_low, n = self._increase_count(m_high, n)
                if m_low <= M_min:
                    n_L = int(self.mf.integral_number(M_min, m_high))
                    if n_L > 0:
                        self.edges.append(M_min)
                        self.counts.append(n_L)
                    break

            self.edges.append(m_low)
            self.counts.append(n)
            m_high = m_low
            n = 1

        # compute average mass of each bin using the mass-function integral
        self.masses = []
        for i in range(len(self.counts)):
            m2 = self.edges[i]      # upper edge
            m1 = self.edges[i + 1]  # lower edge
            mass_tot = self.mf.integral_mass(m1, m2)
            self.masses.append(mass_tot / self.counts[i])

        return list(zip(self.masses, self.counts))


class HybridSampler:
    """Hybrid sampler combining optimal sampling and histogram binning.

    High-mass bins are produced using the deterministic :class:`OptimalSampler`
    algorithm.  Once the expected number of objects per logarithmic bin exceeds
    ``transition_N``, the remaining range is partitioned into a fixed number of
    logarithmic histogram bins and filled using analytic integrals.

    Parameters
    ----------
    mass_function : MassFunction
        Mass function to draw from.
    resolution : float, optional
        Minimum relative bin width passed to :class:`OptimalSampler`.
    transition_N : float, optional
        Threshold in expected counts for switching from optimal sampling to a
        logarithmic histogram.
    hist_bins : int, optional
        Number of logarithmic bins used for the histogram part.
    """

    def __init__(self, mass_function, resolution=1.0, transition_N=5.0, hist_bins=30):
        self.mf = mass_function
        self.resolution = resolution
        self.transition_N = transition_N
        self.hist_bins = hist_bins

        self.optimal_bins: list[tuple[float, float, float]] = []
        self.integrated_bins: list[tuple[float, float, float]] = []

    def sample(self):
        """Perform hybrid sampling using top-down binning."""

        M_min, M_max = self.mf.bounds_sampled
        logM_min = np.log10(M_min)
        logM_max = np.log10(M_max)
        hist_bin_width = (logM_max - logM_min) / self.hist_bins
        m_high = M_max

        # Top-down optimal sampling until histogram criterion is met
        while m_high > M_min:
            logM = np.log10(m_high)
            log_m_low_est = logM - hist_bin_width
            m_low_est = max(10 ** log_m_low_est, M_min)
            N_hist_est = self.mf.integral_number(m_low_est, m_high)

            if N_hist_est >= self.transition_N:
                break

            m_low = self.mf.inverse_integral(m_high, self.resolution)
            if m_low >= m_high or abs(m_high - m_low) / m_high < 1e-6:
                break

            m_low = max(m_low, M_min)
            N = self.mf.integral_number(m_low, m_high)
            self.optimal_bins.append((m_low, m_high, N))
            m_high = m_low

        # Remaining mass range is handled via logarithmic histogram bins
        if m_high > M_min:
            m_high_adj = np.nextafter(m_high, M_min)
            m_edges = np.geomspace(M_min, m_high_adj, self.hist_bins)
            for i in range(len(m_edges) - 1):
                m1, m2 = m_edges[i], m_edges[i + 1]
                N = self.mf.integral_number(m1, m2)
                self.integrated_bins.append((m1, m2, N))
        return self.optimal_bins[::-1] + self.integrated_bins
