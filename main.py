import numpy as np
import matplotlib.pyplot as plt
from massfunction import MassFunction
from samplers import OptimalSampler, HybridSampler

def M_max_ecl(SFR):
    return 10**(4.83 + 0.75 * np.log10(SFR))

if __name__ == "__main__":
    alpha = 2.0
    resolution = 1
    min_expected = 1.0

    SFR = 1.0e-3  # Msun/yr
    delta_t = 10e6  # yr
    total_mass = SFR * delta_t
    M_min = 5
    M_max_theory = 1e9
    M_max_sampled = M_max_ecl(SFR)

    mf = MassFunction(
        alpha=alpha,
        total_mass=total_mass,
        M_min=M_min,
        M_max_sampled=M_max_sampled,
        M_max_theory=M_max_theory
    )

    # Optimal sampler
    #opt_sampler = OptimalSampler(mf, resolution=resolution)
    #opt_bins = opt_sampler.sample()
    #opt_widths = [b[1] - b[0] for b in opt_bins]
    #opt_mids = [(b[0] + b[1]) / 2 for b in opt_bins]
    #opt_heights = [b[2] / w for b, w in zip(opt_bins, opt_widths)]

    # Hybrid sampler
    hybrid_sampler = HybridSampler(mf, resolution=resolution, min_expected=min_expected)
    hybrid_bins = hybrid_sampler.sample()
    hybrid_widths = [b[1] - b[0] for b in hybrid_bins]
    hybrid_mids = [(b[0] + b[1]) / 2 for b in hybrid_bins]
    hybrid_heights = [b[2] / w for b, w in zip(hybrid_bins, hybrid_widths)]

    # Reference
    m_vals = np.logspace(np.log10(M_min), np.log10(M_max_sampled), 300)
    ref = mf.dndm(m_vals)

    # Plot
    plt.figure(figsize=(9, 6))
    #plt.bar(opt_mids, opt_heights, width=opt_widths, alpha=0.6, label='Optimal Sampling')
    plt.bar(hybrid_mids, hybrid_heights, width=hybrid_widths, alpha=0.4, label='Hybrid Sampling')
    plt.plot(m_vals, ref, 'k--', label='Analytic Power-law')
    plt.axvline(M_max_sampled, color='red', linestyle='--', label=r'$M_{\mathrm{max}}$ from SFR')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cluster Mass $M$ [$M_\odot$]')
    plt.ylabel(r'd$N$/d$M$')
    plt.title('Comparison of Optimal and Hybrid Sampling')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()