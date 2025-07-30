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
    def __init__(self, mass_function, resolution=1, min_expected=1.0, hist_bins=30):
        self.mf = mass_function
        self.resolution = resolution
        self.min_expected = min_expected
        self.hist_bins = hist_bins
        self.optimal_bins = []
        self.integrated_bins = []

    def sample(self):
        M_min, M_max = self.mf.bounds_sampled
        m_high = M_max

        while m_high > M_min:
            m_low = self.mf.inverse_integral(m_high, self.resolution)
            if m_low >= m_high or abs(m_high - m_low) / m_high < 1e-6:
                break
            m_low = max(m_low, M_min)
            N = self.mf.integral_number(m_low, m_high)
            if N < self.min_expected:
                break
            self.optimal_bins.append((m_low, m_high, N))
            m_high = m_low

        if m_high > M_min:
            # Avoid overlapping by shifting the lower edge slightly
            m_high_adj = np.nextafter(m_high, M_min)
            m_edges = np.geomspace(M_min, m_high_adj, self.hist_bins)
            for i in range(len(m_edges) - 1):
                m1, m2 = m_edges[i], m_edges[i + 1]
                N = self.mf.integral_number(m1, m2)
                self.integrated_bins.append((m1, m2, N))

        return self.optimal_bins[::-1] + self.integrated_bins