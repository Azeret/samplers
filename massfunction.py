import numpy as np


class MassFunction:
    def __init__(self, alpha, total_mass, M_min, M_max_theory, M_max_sampled):
        self.alpha = float(alpha)
        self.total_mass = float(total_mass)
        self.M_min = float(M_min)
        self.M_max_theory = float(M_max_theory)
        self.M_max_sampled = float(M_max_sampled)
        self._k = None

    @property
    def bounds_theory(self):
        return (self.M_min, self.M_max_theory)

    @property
    def bounds_sampled(self):
        return (self.M_min, self.M_max_sampled)

    def normalization_constant(self):
        if self._k is None:
            a, b = self.bounds_theory
            if np.isclose(self.alpha, 2.0):
                integral_mass = np.log(b / a)
            else:
                integral_mass = (b ** (2 - self.alpha) - a ** (2 - self.alpha)) / (2 - self.alpha)
            self._k = self.total_mass / integral_mass
        return self._k

    def integral_number(self, m1, m2):
        k = self.normalization_constant()
        if np.isclose(self.alpha, 1.0):
            return k * np.log(m2 / m1)
        return k * (m2 ** (1 - self.alpha) - m1 ** (1 - self.alpha)) / (1 - self.alpha)

    def inverse_integral(self, m_high, n_target):
        k = self.normalization_constant()
        if np.isclose(self.alpha, 1.0):
            return m_high / np.exp(n_target / k)
        value = m_high ** (1 - self.alpha) - (n_target / k) * (1 - self.alpha)
        if value <= 0:
            return self.M_min
        return value ** (1 / (1 - self.alpha))

    def dndm(self, m):
        return self.normalization_constant() * np.power(m, -self.alpha)
