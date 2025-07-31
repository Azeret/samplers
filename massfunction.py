import numpy as np
from scipy.integrate import quad


class MassFunction:
    def __init__(self, alpha, total_mass, M_min_theory, M_max_theory, M_min_sampled, M_max_sampled):
        self.alpha = alpha
        self.total_mass = total_mass
        self.M_min_theory = M_min_theory
        self.M_min_sampled = M_min_sampled
        self.M_max_theory = M_max_theory
        self.M_max_sampled = M_max_sampled

    @property
    def bounds_theory(self):
        return (self.M_min_theory, self.M_max_theory)

    @property
    def bounds_sampled(self):
        return (self.M_min_sampled, self.M_max_sampled)

    @property
    def bounds_normalize(self):
        return (self.M_min_theory, self.M_max_sampled)
        
    def normalization_constant(self):
        m1, m2 = self.bounds_normalize
        if self.alpha == 2.:
            integral_mass = (np.log(m2)-np.log(m1))
        else:
            integral_mass = (m2**(2.-self.alpha) - m1**(2.-self.alpha))/(2.-self.alpha)
        return  self.total_mass / integral_mass

    def integral_number(self, m1, m2):
        k = self.normalization_constant()
        if self.alpha == 1:
            return k * np.log(m2 / m1)
        return k * (m2**(1 - self.alpha) - m1**(1 - self.alpha)) / (1 - self.alpha)

    def integral_mass(self, m1, m2 ):
        norm = self.normalization_constant()        
        if self.alpha == 2.:
            M_bin = norm*(np.log(m2)-np.log(m1))
        else:
            M_bin = norm*(m2**(2.-self.alpha) - m1**(2.-self.alpha))/(2.-self.alpha)
        return M_bin
        

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
