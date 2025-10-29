import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def main():
    """
    This function call functions to perform the appropriate operations
    to calculate value for the average conversion based on theoretical
    models found in Elements of Chemical Reaction Engineering Reaction
    by H. Scott Fogler.

    It is extremely important that appropriate units are used such that the
    units will work out, otherwise results will not be correct. I recommend
    using LITERS and SECONDS to ensure units work out.
    """
    # Variables from Experiment
    k = 0.09283  # [L/mol s]
    Cao = 0.076854 / 2  # [M]
    Cbo = 0.1 / 2  # [M]
    Vr = 3.06243  # [L]         Volume of Reactor
    Vdot_turb = 0.06309  # [L/s]   1 gal per min
    Vdot_tran = 0.0283905  # [L/s]   0.45 Gal per min
    tau = 194.2  # [s]          Laminar Residence Time

    # Call functions
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

    plot()
    return


def Ca(t, k, Cao, Cbo):
    """
    This is the function for the concentration of NaOH. See IntegrationAttempt3.mw
    for details on how this equation was obtained.
    """
    # Obtained w/ Maple dsolve
    # con_A = (Cao - Cbo) * Cao / (-np.exp(-(Cao - Cbo) * k * t) * Cbo + Cao)

    # Riley
    con_A = (Cao - Cbo) * Cao / (-np.exp(-k * (Cao - Cbo) * t) * Cbo + Cao)
    return con_A


def X(t, k, Cao, Cbo):
    """
    Performs the calculation for conversion.
    Conversion X = 1 - (Ca/Cao)
    """
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
    """
    Calls on the scipy.integrate quad module to integrate for the
    average conversion in laminar flow.
    """
    return quad(lamfunc, tau / 2, np.inf, args=(k, Cao, Cbo, tau))

def plot():
    exp_flowrate = np.array([0.25, 0.45, 1.00])
    exp_conv = np.array([0.562, 0.382, 0.221])
    exp_flowrate_error = np.array([0.00358, 0.00353, 0.0451])
    exp_conv_error = np.array([0.00619, 0.00817, 0.00814])

    fig, ax = plt.subplots()
    ax.errorbar(
        exp_flowrate,
        exp_conv,
        xerr = exp_flowrate_error,
        yerr = exp_conv_error,
        capsize = 4,
        fmt = 'o'
    )

    # Use plt.plot for Turb and Lam models

    plt.show()


if __name__ == "__main__":
    main()
