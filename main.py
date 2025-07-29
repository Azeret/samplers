if __name__ == "__main__":
    from massfunction import MassFunction
    from samplers import OptimalSampler

    total_mass = 1e10
    bounds = [1, 1e7]
    alpha = 2.0

    mf = MassFunction(bounds, alpha, total_mass=total_mass)
    sampler = OptimalSampler(mf, resolution=1)
    bins = sampler.sample()

    import pandas as pd
    df = pd.DataFrame(bins, columns=["m_low", "m_high", "N"])
    m_centers = (df.m_low * df.m_high)**0.5

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(m_centers, df.N, marker="o", label="Optimal Sampling")
    plt.plot(m_centers, mf.dndm(m_centers), 'k--', label=r"$\propto m^{-%.1f}$" % alpha)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Cluster Mass [Msun]")
    plt.ylabel("dN")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()