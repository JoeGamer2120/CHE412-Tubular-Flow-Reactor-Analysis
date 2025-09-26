import numpy as np
from scipy.integrate import quad


def main():
    """
    Define Variables

    It is extremely important that appropriate units are used such that the
    units will work out, otherwise results will not be correct. I recommend
    using LITERS and SECONDS to ensure units work out.
    """
    k = 0.09283  # [L/mol s]
    Cao = 0.08  # [M]
    Cbo = 0.1  # [M]
    Vr = 3.06243  # [L]         Volume of Reactor
    Vdot = 0.06309  # [L/s]   1 gal per min
    # Vdot = 0.0283905  # [L/s]   0.45 Gal per min
    tau = 194.2  # [s]          Laminar Residence Time

    # Pick Flow Regiem, comment out the other
    # Turb = Turbulent
    # Lam = Laminar
    # For Transition, use Turbulent with Transition Variables

    # I, err = int_turb(k, Cao, Cbo, Vr, Vdot)
    I, err = int_lam(k, Cao, Cbo, tau)
    print("X_bar =", I)
    print("Error: ", err)

    return


# I, err = int_turb(0.09283, 0.08, 0.1, 3.0624, 0.06309)
# print(I)
# print(err)
#
# I, err = int_lam(0.09283, 0.08, 0.1, 194.2)
# print(I)
# print(err)


def Ca(t, k, Cao, Cbo):
    """
    This is the function for the concentration of NaOH.
    """
    return (
        (
            (
                Cao**3 * k
                - 3 * Cao**2 * Cbo * k
                + 3 * Cao * Cbo**2 * k
                - Cbo**3 * k
                + 2
                * np.sqrt(6)
                * np.sqrt(
                    t
                    * (
                        Cao**3 * k
                        - 3 * Cao**2 * Cbo * k
                        + 3 * Cao * Cbo**2 * k
                        - Cbo**3 * k
                        + 6 * t
                    )
                )
                + 12 * t
            )
            * k**2
        )
        ** (1 / 3)
        / (2 * k)
        + (Cao - Cbo) ** 2
        * k
        / (
            2
            * (
                (
                    Cao**3 * k
                    - 3 * Cao**2 * Cbo * k
                    + 3 * Cao * Cbo**2 * k
                    - Cbo**3 * k
                    + 2
                    * np.sqrt(6)
                    * np.sqrt(
                        t
                        * (
                            Cao**3 * k
                            - 3 * Cao**2 * Cbo * k
                            + 3 * Cao * Cbo**2 * k
                            - Cbo**3 * k
                            + 6 * t
                        )
                    )
                    + 12 * t
                )
                * k**2
            )
            ** (1 / 3)
        )
        + Cao / 2
        - Cbo / 2
    )


def X(t, k, Cao, Cbo):
    return 1 - Ca(t, k, Cao, Cbo) / Cao


def turbfunc(t, k, Cao, Cbo, Vr, Vdot):
    """
    Integrand for turbulent flow.
    This function will also be used for transition flow
    """
    RTD = Vr / Vdot
    return X(t, k, Cao, Cbo) * RTD


def lamfunc(t, k, Cao, Cbo, tau):
    """
    Integrand for laminar flow regime.
    This is only applicable when t >= tau/2, as when t is less than tau/2, the
    RTD is 0 making the integrand 0.
    """
    RTD = tau**2 / (2 * tau**3)
    return X(t, k, Cao, Cbo) * RTD


def int_turb(k, Cao, Cbo, Vr, Vdot):
    return quad(turbfunc, 0, np.inf, args=(k, Cao, Cbo, Vr, Vdot))


def int_lam(k, Cao, Cbo, tau):
    return quad(lamfunc, tau / 2, np.inf, args=(k, Cao, Cbo, tau))


if __name__ == "__main__":
    main()
