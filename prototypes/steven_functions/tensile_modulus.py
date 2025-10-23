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

pk1 = J * sigma * (F.inv()).T

strain = 0.001

cs = np.linspace(0.0, 1, dtype=float)
Es = []

for c_truth in cs:
    pk1_truth = pk1.subs(c, c_truth)
    s_e = pk1_truth.subs(lmbda, 1.0 + strain)
    s11 = s_e[0, 0]
    print(c_truth, s11 / strain)
    Es.append(s11 / strain)

plt.figure(figsize=(4, 3), dpi=300)
plt.plot(
    cs,
    Es,
    c="#A3A3A3",
    label="Symbolic estimation",
    ls=":",
    lw=4.5,
)
z = np.polyfit(cs, np.array(Es, dtype=float), 4)
p = np.poly1d(z)
plt.plot(
    cs,
    p(cs),
    c="#5171A5",
    label="Least squares fitting",
    ls="-",
    lw=2.5,
)
print(p)
yhat = p(cs)  # or [p(z) for z in x]
ybar = np.sum(Es) / len(Es)  # or sum(y)/len(y)
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((np.array(Es) - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r2 = ssreg / sstot
print(r2)
print(z)
formula = (
    r"$E=$"
    + f"{z[0]:.3f}"
    + r"$c^4$"
    + f"{r'$+$' if z[1]>0 else r'$-$'}{abs(z[1]):.3f}"
    + r"$c^3$"
    + f"{r'$+$' if z[2]>0 else r'$-$'}{abs(z[2]):.3f}"
    + r"$c^2$"
    + "\n      "
    + f"{r'$+$' if z[3]>0 else r'$-$'}{abs(z[3]):.3f}"
    + r"$c$"
    + f"{r'$+$' if z[4]>0 else r'$-$'}{abs(z[4]):.3f}"
)
plt.xlabel("Composition " + r"$c$")
plt.ylabel("Tensile modulus " + r"$E$" + " (MPa)")
plt.text(0.05, 0.85, formula, transform=plt.gca().transAxes)
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("fit_tensile_modulus.pdf")
plt.show()
plt.close()
