import numpy as np


class OptimalSampler:
    """Deterministic sampling following the optimal-sampling recipe.

    The algorithm mirrors the one used in the GalIMF project: starting from the
    maximum mass, it iteratively determines the lower boundary that contains an
    integer number of objects. If the resulting mass interval is too narrow
    (smaller than ``resolution`` times the upper mass) several objects are
    grouped together until the interval is sufficiently wide.

    Parameters
    ----------
    mass_function : MassFunction
        Power-law mass function to sample.
    resolution : float, optional
        Minimum relative bin width (default 0.01).
    mass_grid_index : float, optional
        Factor used when increasing the number of objects grouped into a bin
        (default 1.01, as in the reference implementation).
    """

    def __init__(self, mass_function, resolution=0.01, mass_grid_index=1.01):
        self.mf = mass_function
        self.resolution = resolution
        self.mass_grid_index = mass_grid_index
        self.edges = []
        self.counts = []
        self.masses = []

    def _increase_count(self, m_high, n):
        """Increase ``n`` until the bin width exceeds the resolution."""
        while True:
            n_new = int(round(n * self.mass_grid_index + 1))
            m_low = self.mf.inverse_integral(m_high, n_new)
            if m_high - m_low >= self.resolution * m_high:
                return m_low, n_new
            n = n_new

    def sample(self):
        M_min, M_max = self.mf.bounds_sampled
        m_high = M_max
        self.edges = [m_high]
        self.counts = []

        n = 1
        while m_high > M_min:
            m_low = self.mf.inverse_integral(m_high, n)

            if m_low <= M_min:
                n_L = int(self.mf.integral_number(M_min, m_high))
                if n_L > 0:
                    self.edges.append(M_min)
                    self.counts.append(n_L)
                break

            if m_high - m_low < self.resolution * m_high:
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

        # compute average mass of each bin
        k = self.mf.normalization_constant()
        alpha = self.mf.alpha
        self.masses = []
        for i in range(len(self.counts)):
            m2 = self.edges[i]      # upper edge
            m1 = self.edges[i + 1]  # lower edge
            if np.isclose(alpha, 2.0):
                mass_tot = k * np.log(m2 / m1)
            else:
                mass_tot = k / (2 - alpha) * (m2 ** (2 - alpha) - m1 ** (2 - alpha))
            self.masses.append(mass_tot / self.counts[i])

        return list(zip(self.masses, self.counts))


class HybridSampler:
    def __init__(self, mass_function, resolution=1.0, transition_N=5.0, hist_bins=30):
        """Hybrid sampler combining optimal sampling and histogram binning."""
        self.mf = mass_function
        self.resolution = resolution
        self.transition_N = transition_N
        self.hist_bins = hist_bins

        self.optimal_bins = []
        self.integrated_bins = []

    def sample(self):
        """Perform hybrid sampling using top-down binning."""
        M_min, M_max = self.mf.bounds_sampled
        logM_min = np.log10(M_min)
        logM_max = np.log10(M_max)
        hist_bin_width = (logM_max - logM_min) / self.hist_bins
        m_high = M_max

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

        if m_high > M_min:
            m_high_adj = np.nextafter(m_high, M_min)
            m_edges = np.geomspace(M_min, m_high_adj, self.hist_bins)
            for i in range(len(m_edges) - 1):
                m1, m2 = m_edges[i], m_edges[i + 1]
                N = self.mf.integral_number(m1, m2)
                self.integrated_bins.append((m1, m2, N))

        return self.optimal_bins[::-1] + self.integrated_bins
