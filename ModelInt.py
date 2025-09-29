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
    Vdot_turb = 0.06309  # [L/s]   1 gal per min
    Vdot_tran = 0.0283905  # [L/s]   0.45 Gal per min
    tau = 194.2  # [s]          Laminar Residence Time

    # Pick Flow Regiem, comment out the other
    # Turb = Turbulent
    # Lam = Laminar
    # For Transition, use Turbulent with Transition Variables

    turb_xbar = X(Vr / Vdot_turb, k, Cao, Cbo)  # Turbulent
    tran_xbar = X(Vr / Vdot_tran, k, Cao, Cbo)  # Transition
    I, err = int_lam(k, Cao, Cbo, tau)  # Laminar
    I_tran, err_tran = int_lam(k, Cao, Cbo, Vr / Vdot_tran)
    avgtran_xbar = np.average([I_tran, tran_xbar])

    print("Turbulent Xbar =", turb_xbar)
    print()
    print("Transition (Turbulent Method)", tran_xbar)
    print("Transition (Laminar Method)", I_tran)
    print("Transition (Average):", avgtran_xbar)
    print("Transition Integration Est. Error:", err_tran)
    print()
    print("Laminar Xbar:", I)
    print("Error: ", err)
    return


def Ca(t, k, Cao, Cbo):
    """
    This is the function for the concentration of NaOH.
    """
    # Obtained w/ Maple dsolve
    # con_A = (Cao - Cbo) * Cao / (-np.exp(-(Cao - Cbo) * k * t) * Cbo + Cao)

    # Riley
    con_A = (Cao - Cbo) * Cao / (-np.exp(-k * (Cao - Cbo) * t) * Cbo + Cao)
    return con_A


def X(t, k, Cao, Cbo):
    ans = 1 - Ca(t, k, Cao, Cbo) / Cao
    return ans


def turbfunc(k, Cao, Cbo, Vr, Vdot):
    """
    Integrand for turbulent flow.
    This function will also be used for transition flow
    """
    tau = Vr / Vdot
    return X(tau, k, Cao, Cbo)


def lamfunc(t, k, Cao, Cbo, tau):
    """
    Integrand for laminar flow regime.
    This is only applicable when t >= tau/2, as when t is less than tau/2, the
    RTD is 0 making the integrand 0.
    """
    RTD = tau**2 / (2 * t**3)
    lam_func = X(t, k, Cao, Cbo) * RTD
    return lam_func


def int_lam(k, Cao, Cbo, tau):
    return quad(lamfunc, tau / 2, np.inf, args=(k, Cao, Cbo, tau))


if __name__ == "__main__":
    main()
