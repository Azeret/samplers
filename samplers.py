from massfunction import MassFunction
import numpy as np

class OptimalSampler:
    def __init__(self, mass_function: MassFunction, resolution=1):
        self.mf = mass_function
        self.resolution = resolution
        self.bins = []

    def sample(self):
        M_min, M_max = self.mf.bounds
        m_high = M_max
        while m_high > M_min:
            m_low = self._find_m_low(m_high, self.resolution)
            if m_low >= m_high:
                break
            if m_low < M_min:
                m_low = M_min
            N = self.mf.integral_number(m_low, m_high)
            if N <= 0:
                break
            self.bins.append((m_low, m_high, N))
            if abs(m_high - m_low) / m_high < 1e-6:
                break
            m_high = m_low
        return self.bins[::-1]

    def _find_m_low(self, m_high, n_target):
        return self.mf.inverse_integral(m_high, n_target)
    
class HybridSampler:
    def __init__(self, mass_function: MassFunction, resolution=1, min_expected=1.0):
        self.mf = mass_function
        self.resolution = resolution
        self.min_expected = min_expected  # below this, use integrated histogram
        self.optimal_bins = []
        self.integrated_bins = []

    def sample(self):
        M_min, M_max = self.mf.bounds
        m_high = M_max

        # Optimal sampling
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

        # Switch to integrated histogram
        m_edges = np.geomspace(M_min, m_high, 30)  # adjustable binning
        for i in range(len(m_edges) - 1):
            m1, m2 = m_edges[i], m_edges[i + 1]
            N = self.mf.integral_number(m1, m2)
            self.integrated_bins.append((m1, m2, N))

        return self.optimal_bins[::-1] + self.integrated_bins    