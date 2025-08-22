import numpy as np
import matplotlib.pyplot as plt
from massfunction import MassFunction
from samplers import OptimalSampler, HybridSampler


def M_max_ecl(SFR: float) -> float:
    """Empirical maximum cluster mass from the star-formation rate."""

    return 10 ** (4.83 + 0.75 * np.log10(SFR))


if __name__ == "__main__":
    # Mass-function parameters
    slopes = 2.0  # single power-law slope (can also provide a list for breaks)
    resolution = 1  # minimum relative bin width for the samplers

    # Galaxy/star-formation parameters
    SFR = 1.0e3       # star-formation rate [Msun/yr]
    delta_t = 10e6    # duration of the episode [yr]
    total_mass = SFR * delta_t  # total mass formed in the epoch

    # Mass limits for the cluster mass function
    M_min = 5                  # minimum cluster mass [Msun]
    M_max_theory = 1e9         # theoretical upper limit used for normalisation
    M_max_sampled = M_max_ecl(SFR)  # maximum mass expected from the SFR

    # Build mass function (supports both single and broken power laws)
    mf = MassFunction(
        slopes=slopes,
        total_mass=total_mass,
        M_min=M_min,
        M_max_sampled=M_max_sampled,
        M_max_theory=M_max_theory,
    )

    # Hybrid sampler: optimal sampling at the high-mass end, histogram below
    hybrid_sampler = HybridSampler(
        mf, resolution=resolution, transition_N=10, hist_bins=100
    )
    hybrid_bins = hybrid_sampler.sample()
    hybrid_widths = [b[1] - b[0] for b in hybrid_bins]
    hybrid_mids = [(b[0] + b[1]) / 2 for b in hybrid_bins]
    hybrid_heights = [b[2] / w for b, w in zip(hybrid_bins, hybrid_widths)]

    # Reference analytic curve
    m_vals = np.logspace(np.log10(M_min), np.log10(M_max_sampled), 300)
    ref = mf.dndm(m_vals)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.bar(
        hybrid_mids,
        hybrid_heights,
        width=hybrid_widths,
        alpha=0.4,
        label="Hybrid Sampling",
    )
    plt.plot(m_vals, ref, "k--", label="Analytic Power-law")
    plt.axvline(
        M_max_sampled, color="red", linestyle="--", label=r"$M_{\mathrm{max}}$ from SFR"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cluster Mass $M$ [$M_\odot$]")
    plt.ylabel(r"d$N$/d$M$")
    plt.title("Comparison of Optimal and Hybrid Sampling")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()