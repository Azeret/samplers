import numpy as np
from scipy.integrate import quad


class MassFunction:
    def __init__(self, alpha, total_mass, M_min, M_max_theory, M_max_sampled):
        self.alpha = alpha
        self.total_mass = total_mass
        self.M_min = M_min
        self.M_max_theory = M_max_theory
        self.M_max_sampled = M_max_sampled

    @property
    def bounds_theory(self):
        return (self.M_min, self.M_max_theory)

    @property
    def bounds_sampled(self):
        return (self.M_min, self.M_max_sampled)

    def normalization_constant(self):
        a, b = self.bounds_theory
        if self.alpha == 2:
            integral_mass, _ = quad(lambda m: m**(-1), a, b)
        else:
            integral_mass, _ = quad(lambda m: m * m**(-self.alpha), a, b)
        return self.total_mass / integral_mass

    def integral_number(self, m1, m2):
        k = self.normalization_constant()
        if self.alpha == 1:
            return k * np.log(m2 / m1)
        return k * (m2**(1 - self.alpha) - m1**(1 - self.alpha)) / (1 - self.alpha)

    def inverse_integral(self, m_high, n_target):
        k = self.normalization_constant()
        if self.alpha == 1:
            return m_high / np.exp(n_target / k)
        value = m_high**(1 - self.alpha) - (n_target / k) * (1 - self.alpha)
        if value <= 0:
            return self.M_min
        return value**(1 / (1 - self.alpha))

    def dndm(self, m):
        k = self.normalization_constant()
        return k * m**(-self.alpha)