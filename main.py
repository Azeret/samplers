import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import pandas as pd

if __name__ == "__main__":
    from massfunction import MassFunction
    from samplers import OptimalSampler
    from samplers import HybridSampler


    def M_max_ecl(SFR):
        return 10**(4.83 + 0.75 * np.log10(SFR))

    alpha = 2.0
    resolution = 1  # clusters per bin

    SFR = 1.e-4  # Msun/yr
    delta_t = 10e6  # 10 Myr window

    M_tot = SFR * delta_t
    M_min = 5
    M_max = M_max_ecl(SFR)
    bounds = [M_min, M_max]

    mf = MassFunction(bounds, alpha, total_mass=M_tot)
    sampler = OptimalSampler(mf, resolution=resolution)
    bins = sampler.sample()

    df = pd.DataFrame(bins, columns=["m_low", "m_high", "N"])
    #df = df[df["N"] > 1e-3]  # filter low-mass bins that clutter the plot
    df["m_center"] = np.sqrt(df["m_low"] * df["m_high"])
    df["dndm_sampled"] = df["N"] / (df["m_high"] - df["m_low"])

    # plot
    plt.figure(figsize=(8, 6))
    plt.bar(
        df["m_center"],
        df["dndm_sampled"],
        width=df["m_high"] - df["m_low"],
        align='center',
        alpha=0.6,
        label="Sampled Histogram (dN/dm)"
    )
    plt.plot(df["m_center"], mf.dndm(df["m_center"]), 'k--', label=r"Analytic $m^{-%.1f}$" % alpha)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cluster Mass [Msun]")
    plt.ylabel("dN/dm")
    plt.title("Optimal Sampling from ECMF with SFR-dependent $M_{\mathrm{ecl,max}}$")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    #plt.show()

    SFR = 1  # M_sun/yr
    delta_t = 10e6  # yr
    total_mass = SFR * delta_t
    M_min = 5
    M_max = M_max_ecl(SFR)

    mf = MassFunction(bounds=(M_min, M_max), alpha=2.0, total_mass=total_mass)

    sampler = HybridSampler(mass_function=mf, resolution=1, min_expected=1.0)
    bins = sampler.sample()

    # Prepare for plotting
    widths = [b[1] - b[0] for b in bins]
    mids = [(b[0] + b[1]) / 2 for b in bins]
    heights = [b[2] / w for b, w in zip(bins, widths)]  # dN/dM

    # Reference powerlaw
    m_vals = np.logspace(np.log10(M_min), np.log10(M_max), 300)
    ref = mf.normalization_constant() * m_vals**(-mf.alpha)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.bar(mids, heights, width=widths, align='center', alpha=0.6, label='Hybrid Sampled')
    plt.plot(m_vals, ref, 'k--', label='Reference Power-law')
    plt.axvline(M_max, color='red', linestyle='--', label=r'$M_{\mathrm{max}}$ from SFR')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Cluster Mass $M$ [$M_\odot$]')
    plt.ylabel(r'd$N$/d$M$')
    plt.legend()
    plt.tight_layout()
    plt.title('Hybrid Sampling from Power-law Mass Function')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.show()