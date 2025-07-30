import numpy as np
class OptimalSampler:
    def __init__(self, mass_function, resolution=1):
        self.mf = mass_function
        self.resolution = resolution
        self.bins = []

    def sample(self):
        M_min, M_max = self.mf.bounds_sampled
        m_high = M_max
        while m_high > M_min:
            m_low = self.mf.inverse_integral(m_high, self.resolution)
            if m_low >= m_high or abs(m_high - m_low) / m_high < 1e-6:
                break
            m_low = max(m_low, M_min)
            N = self.mf.integral_number(m_low, m_high)
            if N <= 0:
                break
            self.bins.append((m_low, m_high, N))
            m_high = m_low
        return self.bins[::-1]


class HybridSampler:
    def __init__(self, mass_function, resolution=1.0, transition_N=5.0, hist_bins=30):
        """
        Hybrid sampler combining optimal sampling and histogram binning
        from a power-law mass function.

        Parameters:
        -----------
        mass_function : MassFunction
            Instance providing dN/dM and integration/inversion methods.
        resolution : float
            Desired number of expected clusters per bin in optimal sampling (default 1.0).
        transition_N : float
            Threshold number of expected clusters above which histogramming is used.
        hist_bins : int
            Number of logarithmic bins used in histogramming.
        """
        self.mf = mass_function
        self.resolution = resolution
        self.transition_N = transition_N
        self.hist_bins = hist_bins

        self.optimal_bins = []
        self.integrated_bins = []

    def sample(self):
        """
        Perform hybrid sampling using top-down binning.

        Returns:
        --------
        List of tuples: (m_low, m_high, N_expected)
        """
        M_min, M_max = self.mf.bounds_sampled
        logM_min = np.log10(M_min)
        logM_max = np.log10(M_max)
        hist_bin_width = (logM_max - logM_min) / self.hist_bins
        m_high = M_max

        print(f"[1] Starting hybrid sampling from M_max = {M_max:.2e} down to M_min = {M_min:.2e}")

        # === OPTIMAL SAMPLING (high-mass, rare regime) ===
        while m_high > M_min:
            # Estimate what the corresponding histogram bin would contain
            logM = np.log10(m_high)
            log_m_low_est = logM - hist_bin_width
            m_low_est = max(10**log_m_low_est, M_min)
            N_hist_est = self.mf.integral_number(m_low_est, m_high)

            print(f"[2.0] Checking histogram bin from {m_low_est:.2e} to {m_high:.2e} → N ≈ {N_hist_est:.3f}")

            if N_hist_est >= self.transition_N:
                print(f"[3] Switching to histogramming at N = {N_hist_est:.3f} (≥ {self.transition_N})")
                break  # switch to histogramming

            # Do optimal sampling: shrink bin to get ~self.resolution objects
            m_low = self.mf.inverse_integral(m_high, self.resolution)

            # Stop if bin is degenerate
            if m_low >= m_high or abs(m_high - m_low) / m_high < 1e-6:
                print("[3.1] Optimal bin degenerate, stopping.")
                break

            m_low = max(m_low, M_min)
            N = self.mf.integral_number(m_low, m_high)
            print(f"[3.2] OPT bin: {m_low:.2e} – {m_high:.2e}, ΔM = {m_high - m_low:.2e}, N = {N:.3f}")
            self.optimal_bins.append((m_low, m_high, N))
            m_high = m_low  # move to next bin down

        # === HISTOGRAMMING (low-mass, dense regime) ===
        if m_high > M_min:
            m_high_adj = np.nextafter(m_high, M_min)  # avoid overlap
            m_edges = np.geomspace(M_min, m_high_adj, self.hist_bins)

            print(f"[4] Starting histogramming with {self.hist_bins} bins from {M_min:.2e} to {m_high_adj:.2e}")
            for i in range(len(m_edges) - 1):
                m1, m2 = m_edges[i], m_edges[i + 1]
                N = self.mf.integral_number(m1, m2)
                print(f"[4.{i}] HIST bin: {m1:.2e} – {m2:.2e}, N = {N:.3f}")
                self.integrated_bins.append((m1, m2, N))

        # Return all bins: optimal first, then histogram bins
        return self.optimal_bins[::-1] + self.integrated_bins