"""Mass function utilities.

This module defines :class:`MassFunction`, a helper class that evaluates and
inverts power-law mass functions.  The implementation supports both a single
power-law slope and an arbitrary number of power-law segments separated by
break masses.  All calculations are analytic and follow the notation used in
the reference GalIMF scripts.

The main quantities are

``dN/dm = k_i * m^{-alpha_i}``

within each segment ``i`` bounded by ``m_low[i]`` and ``m_high[i]``.  The
normalisation constants ``k_i`` are chosen such that the distribution is
continuous at the break masses and integrates to ``total_mass`` over the full
theoretical range ``[M_min, M_max_theory]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class MassFunction:
    """Power-law or broken power-law mass function.

    Parameters
    ----------
    slopes : float or Iterable[float]
        Power-law slope :math:`\alpha` (or a sequence of slopes for a broken
        power law).  The mass spectrum in each segment follows
        ``dN/dm = k_i m^{-alpha_i}``.
    total_mass : float
        Total mass formed between ``M_min`` and ``M_max_theory``.
    M_min : float
        Minimum mass considered.
    M_max_theory : float
        Maximum mass used for normalisation of the mass function.
    M_max_sampled : float
        Upper mass used when sampling.  This may be smaller than
        ``M_max_theory``.
    breaks : Iterable[float], optional
        Sequence of increasing break masses separating the power-law
        segments.  For *N* slopes, provide *N-1* break masses.

    Notes
    -----
    ``slopes`` and ``breaks`` are converted to lists internally.  The first
    slope is also exposed via the attribute ``alpha`` for backward
    compatibility with earlier single-slope code.
    """

    slopes: Iterable[float]
    total_mass: float
    M_min: float
    M_max_theory: float
    M_max_sampled: float
    breaks: Iterable[float] | None = None

    def __post_init__(self) -> None:
        self.alpha_list: List[float] = (  # list of slopes for each segment
            [float(self.slopes)]
            if isinstance(self.slopes, (int, float))
            else [float(a) for a in self.slopes]
        )
        self.breaks_list: List[float] = (
            [] if self.breaks is None else [float(b) for b in self.breaks]
        )
        if len(self.alpha_list) - 1 != len(self.breaks_list):
            raise ValueError("Number of slopes must be one more than breaks")

        # Boundaries of each mass segment including the theoretical limits
        self.segment_bounds = [self.M_min] + self.breaks_list + [self.M_max_theory]

        # Prefactors relative to k1 ensuring continuity between segments
        self._prefactors = [1.0]
        for i in range(1, len(self.alpha_list)):
            m_break = self.breaks_list[i - 1]
            a_prev = self.alpha_list[i - 1]
            a_curr = self.alpha_list[i]
            self._prefactors.append(
                self._prefactors[-1] * m_break ** (a_curr - a_prev)
            )

        self.alpha = self.alpha_list[0]  # first slope for backward compatibility
        self._k1: float | None = None  # overall normalisation constant
        self._k_list: List[float] | None = None  # constants for each segment

    # ------------------------------------------------------------------
    # Convenience properties
    @property
    def bounds_theory(self):
        """Return ``(M_min, M_max_theory)``."""

        return (self.M_min, self.M_max_theory)

    @property
    def bounds_sampled(self):
        """Return ``(M_min, M_max_sampled)``."""

        return (self.M_min, self.M_max_sampled)

    # ------------------------------------------------------------------
    # Normalisation and helper utilities
    def _ensure_normalisation(self) -> None:
        """Compute the normalisation constants if not already done."""

        if self._k1 is not None:
            return

        # Integrate assuming k1 = 1 to determine the required normalisation
        integral_mass = 0.0
        for i, alpha in enumerate(self.alpha_list):
            m1 = self.segment_bounds[i]
            m2 = self.segment_bounds[i + 1]
            k = self._prefactors[i]
            if np.isclose(alpha, 2.0):
                seg = k * np.log(m2 / m1)
            else:
                seg = k * (m2 ** (2 - alpha) - m1 ** (2 - alpha)) / (2 - alpha)
            integral_mass += seg

        self._k1 = self.total_mass / integral_mass
        self._k_list = [self._k1 * p for p in self._prefactors]

    # ------------------------------------------------------------------
    # Integral utilities
    def _segment_integral_number(self, i: int, m1: float, m2: float) -> float:
        """Return expected number of objects in segment ``i`` between ``m1``
        and ``m2``."""

        alpha = self.alpha_list[i]
        k = self._k_list[i]
        if np.isclose(alpha, 1.0):
            return k * np.log(m2 / m1)
        return k * (m2 ** (1 - alpha) - m1 ** (1 - alpha)) / (1 - alpha)

    def _segment_integral_mass(self, i: int, m1: float, m2: float) -> float:
        """Return total mass in segment ``i`` between ``m1`` and ``m2``."""

        alpha = self.alpha_list[i]
        k = self._k_list[i]
        if np.isclose(alpha, 2.0):
            return k * np.log(m2 / m1)
        return k * (m2 ** (2 - alpha) - m1 ** (2 - alpha)) / (2 - alpha)

    # ------------------------------------------------------------------
    # Public API
    def normalization_constant(self) -> float:
        """Return the normalisation constant ``k1`` of the first segment."""

        self._ensure_normalisation()
        return self._k1  # type: ignore[return-value]

    def integral_number(self, m1: float, m2: float) -> float:
        """Return expected number of objects between ``m1`` and ``m2``."""

        self._ensure_normalisation()
        if m2 <= m1:
            return 0.0
        total = 0.0
        for i in range(len(self.alpha_list)):
            seg_low = max(m1, self.segment_bounds[i])
            seg_high = min(m2, self.segment_bounds[i + 1])
            if seg_high <= seg_low:
                continue
            total += self._segment_integral_number(i, seg_low, seg_high)
        return total

    def integral_mass(self, m1: float, m2: float) -> float:
        """Return total mass between ``m1`` and ``m2``."""

        self._ensure_normalisation()
        if m2 <= m1:
            return 0.0
        total = 0.0
        for i in range(len(self.alpha_list)):
            seg_low = max(m1, self.segment_bounds[i])
            seg_high = min(m2, self.segment_bounds[i + 1])
            if seg_high <= seg_low:
                continue
            total += self._segment_integral_mass(i, seg_low, seg_high)
        return total

    def inverse_integral(self, m_high: float, n_target: float) -> float:
        """Return the lower mass bound such that the interval ``[m_low, m_high]``
        contains ``n_target`` objects.

        Parameters
        ----------
        m_high : float
            Upper mass boundary of the interval.
        n_target : float
            Desired number of objects within the interval.
        """

        self._ensure_normalisation()

        # Identify which segment contains m_high
        idx = np.searchsorted(self.segment_bounds, m_high, side="right") - 1
        m_top = m_high
        remaining = n_target

        while idx >= 0:
            m_low_bound = self.segment_bounds[idx]
            seg_N = self._segment_integral_number(idx, m_low_bound, m_top)
            if remaining <= seg_N:
                # Solve within this segment
                alpha = self.alpha_list[idx]
                k = self._k_list[idx]
                if np.isclose(alpha, 1.0):
                    return m_top / np.exp(remaining / k)
                value = m_top ** (1 - alpha) - remaining * (1 - alpha) / k
                if value <= 0:
                    return self.M_min
                return value ** (1 / (1 - alpha))

            remaining -= seg_N
            m_top = m_low_bound
            idx -= 1

        return self.M_min

    def dndm(self, m: float | np.ndarray) -> float | np.ndarray:
        """Return the differential number density ``dN/dm`` at mass ``m``.

        The input can be either a scalar or a NumPy array; the returned value
        has the same shape as the input.
        """

        self._ensure_normalisation()
        m_arr = np.asarray(m)
        idx = np.searchsorted(self.segment_bounds, m_arr, side="right") - 1
        idx = np.clip(idx, 0, len(self.alpha_list) - 1)
        k = np.array(self._k_list)[idx]
        alpha = np.array(self.alpha_list)[idx]
        result = k * m_arr ** (-alpha)
        return result if isinstance(m, np.ndarray) else float(result)

