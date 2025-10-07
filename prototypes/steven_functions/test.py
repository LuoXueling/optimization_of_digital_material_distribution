import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

a1 = 6.84602
a2 = 6.86996
a3 = 6.88268
a4 = 6.93876
b1 = 2.37126
b2 = 2.19610
b3 = 2.56167
b4 = 1.98880

lmbda = sp.Symbol("lambda")
c = sp.Symbol("c")
I1 = sp.Symbol("I1")
I2 = sp.Symbol("I2")
J = 1

# The following only works for J=1 and thus I1=I1bar and I2=I2bar
psi_hat = (
    0.083089 * I1
    + 4.01198
    * sp.log(
        (
            sp.exp(-0.207897 * I1 + 0.135303 * I2)
            / (
                (sp.exp(a1 * c) + 1) ** b1
                * (sp.exp(a2 * c) + 1) ** b2
                * (sp.exp(a3 * c) + 1) ** b3
                * (sp.exp(a4 * c) + 1) ** b4
                + 1
            )
            ** 0.021823
            + 1
        )
        ** 1.16891
        * (
            (
                (sp.exp(a1 * c) + 1) ** b1
                * (sp.exp(a2 * c) + 1) ** b2
                * (sp.exp(a3 * c) + 1) ** b3
                * (sp.exp(a4 * c) + 1) ** b4
                + 1
            )
            ** 0.049792
            * sp.exp(0.228289 * I1 - 0.204157 * I2)
            + 1
        )
        ** 0.733048
        * sp.exp(0.033506 * I1)
        + 1
    )
    + 0.809351 * sp.log(sp.exp(0.009634 * I1) + 1)
)

# Assuming that J=1 and the incompressibility term is omitted.
psi = psi_hat - psi_hat.subs(I1, 3).subs(I2, 3)

# Specifically for the uniaxial tension case
F = sp.Matrix(
    [
        [lmbda, 0, 0],
        [0, lmbda ** (-0.5), 0],
        [0, 0, lmbda ** (-0.5)],
    ]
)

I = sp.Matrix(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)

C = F.T * F


pI1_pC = J ** (-2 / 3) * (I - 1 / 3 * sp.trace(C) * C.inv())
pI2_pC = (
    I1 * pI1_pC - J ** (-4 / 3) * C + 1 / 3 * J ** (-4 / 3) * sp.trace(C**2) * C.inv()
)


# Only works for J=1
pPsi_pI1 = sp.diff(psi, I1)
pPsi_pI2 = sp.diff(psi, I2)

S = 2 * (pPsi_pI1 * pI1_pC + pPsi_pI2 * pI2_pC).subs(
    I1, lmbda**2 + 2 * 1 / lmbda
).subs(I2, 2 * lmbda + lmbda ** (-2))
sigma_wo_cons = J ** (-1) * F * S * F.T
sigma = sigma_wo_cons - sigma_wo_cons[1, 1] * I

# Viscous part
p1 = (
    (1 + sp.exp(-1.381 * c)) ** 0.423
    * (1 + sp.exp(-1.018 * c)) ** 0.446
    * (1 + sp.exp(-0.976 * c)) ** 0.552
    * (1 + sp.exp(-0.296 * c)) ** 0.492
)
p2 = (sp.exp(0.059 * c) + 1) ** 0.054 * (sp.exp(0.111 * c) + 1) ** 0.147
q1 = (sp.exp(0.754 * c) + 1) ** 0.254 * (sp.exp(1.295 * c) + 1) ** 0.264
gamma = 1 / (p1 * p2 / q1 + 1)

c_truth = 0.0
mat_name = "A"

c_truth = 0.4669
mat_name = "DM-50"
c_truth = 2.0895
mat_name = "DM-70"
tau = 10

disable_integration = True
plot_ref = True
plot_all_in_one = True


def plot_once(v, c_truth):
    T = np.linspace(0, 1.5 / v, 500)
    space = T[1] - T[0]

    gamma_truth = gamma.subs(c, c_truth)
    sigma_truth = sigma.subs(c, c_truth)
    s_e_list = []
    x = []
    y = []
    for i, t in enumerate(T):
        print(f"{t}/{T[-1]}")
        s_e = sigma_truth.subs(lmbda, 1.0 + v * t)
        if disable_integration:
            integrated_11 = 0.0
        else:
            s_e_list.append(s_e[0, 0])
            integrated_values_11 = []
            for j, s in enumerate(T[: i + 1]):
                integrated_values_11.append(
                    -(gamma_truth / tau) * np.exp(-(t - s) / tau) * s_e_list[j]
                )
            integrated_11 = 0.0
            if i > 0:
                for j in range(i):
                    integrated_11 += (
                        (integrated_values_11[j] + integrated_values_11[j + 1])
                        * space
                        / 2
                    )
        s11 = s_e[0, 0] + integrated_11

        x.append(1.0 + v * t)
        y.append(s11)
    return x, y


plt.figure(figsize=(4, 2.5), dpi=150)
ax = plt.subplot(111)

if plot_all_in_one:
    c_truth = 0.0
    mat_name = "A"
    ax.plot(
        *plot_once(1, c_truth),
        label=f"{mat_name} (Symbolic)",
        c="#77A6F2",
        ls=":",
        lw=2.5,
    )
    df = pd.read_csv("../../outputs/dogbone-2025-10-06-14-30-33-A/records.csv")
    ax.plot(
        df["stretch"] + 1,
        -df["Config Dirichlet BC 0_y"],
        label=f"{mat_name} (Numerical)",
        c="#77A6F2",
        lw=1.5,
    )
    c_truth = 0.4669
    mat_name = "DM-50"
    ax.plot(
        *plot_once(1, c_truth),
        label=f"{mat_name} (Symbolic)",
        c="#638AC9",
        ls=":",
        lw=2.5,
    )
    df = pd.read_csv("../../outputs/dogbone-2025-10-03-20-59-44-DM50/records.csv")
    ax.plot(
        df["stretch"] + 1,
        -df["Config Dirichlet BC 0_y"],
        label=f"{mat_name} (Numerical)",
        c="#638AC9",
        lw=1.5,
    )
    c_truth = 2.0895
    mat_name = "DM-70"
    ax.plot(
        *plot_once(1, c_truth),
        label=f"{mat_name} (Symbolic)",
        c="#5171A5",
        ls=":",
        lw=2.5,
    )
    df = pd.read_csv("../../outputs/dogbone-2025-10-03-20-59-57-DM70/records.csv")
    ax.plot(
        df["stretch"] + 1,
        -df["Config Dirichlet BC 0_y"],
        label=f"{mat_name} (Numerical)",
        c="#5171A5",
        lw=1.5,
    )
else:
    if plot_ref:
        df = pd.read_csv(f"{mat_name}-0.9e-1.csv", sep=", ")
        ax.plot(df["x"], df["y"], label="0.9e-1 (Steven)", c="#5171A5", ls=":", lw=2.5)

        df = pd.read_csv(f"{mat_name}-0.9e-2.csv", sep=", ")
        ax.plot(df["x"], df["y"], label="0.9e-2 (Steven)", c="#638AC9", ls=":", lw=2.5)

        df = pd.read_csv(f"{mat_name}-0.9e-3.csv", sep=", ")
        ax.plot(df["x"], df["y"], label="0.9e-3 (Steven)", c="#77A6F2", ls=":", lw=2.5)

    if disable_integration:
        ax.plot(*plot_once(1, c_truth), label="Symbolic", c="#5171A5", lw=1.5)
    else:
        ax.plot(*plot_once(0.9e-1, c_truth), label="0.9e-1", c="#5171A5", lw=1.5)
        ax.plot(*plot_once(0.9e-2, c_truth), label="0.9e-2", c="#638AC9", lw=1.5)
        ax.plot(*plot_once(0.9e-3, c_truth), label="0.9e-3", c="#77A6F2", lw=1.5)
ax.set_xlim([1.0, 3.0])
ax.set_ylim([0, 3])
ax.set_xlabel("Stretch " + r"$\xi$")
ax.set_ylabel(r"$\sigma_{11}$" + " (MPa)")
plt.legend(frameon=False, fontsize=7)
plt.tight_layout()
plt.savefig(
    "all_in_one.pdf"
    if plot_all_in_one
    else f"{mat_name}{'' if not disable_integration else '_noint'}.pdf"
)
plt.close()
