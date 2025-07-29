class MassFunction:
    def __init__(self, bounds, alpha, total_mass=None):
        self.bounds = bounds  # [M_min, M_max]
        self.alpha = alpha
        self.total_mass = total_mass

    def normalization_constant(self):
        a, b = self.bounds
        if self.alpha == 2:
            return self.total_mass / (np.log(b) - np.log(a))
        else:
            return self.total_mass * (1 - self.alpha) / (b**(1 - self.alpha) - a**(1 - self.alpha))

    def integral_number(self, m1, m2):
        if self.alpha == 1:
            return np.log(m2 / m1)
        return (m2**(1 - self.alpha) - m1**(1 - self.alpha)) / (1 - self.alpha)

    def inverse_integral(self, m_high, n_target):
        """Returns m_low such that \int_{m_low}^{m_high} xi(m) dm = n_target"""
        if self.alpha == 1:
            return m_high / np.exp(n_target)
        return (m_high**(1 - self.alpha) - n_target * (1 - self.alpha))**(1 / (1 - self.alpha))

    def dndm(self, m):
        k = self.normalization_constant()
        return k * m**(-self.alpha)