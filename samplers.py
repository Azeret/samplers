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
            if m_low < M_min:
                m_low = M_min
            N = self.mf.integral_number(m_low, m_high)
            self.bins.append((m_low, m_high, N))
            m_high = m_low
        return self.bins[::-1]  # from low to high

    def _find_m_low(self, m_high, n_target):
        return self.mf.inverse_integral(m_high, n_target)