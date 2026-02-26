import numpy as np
import matplotlib.pyplot as plt

cs = [0, 0.1755, 0.4669, 1.0, 2.0895]
stretch_crit = np.array([2.05, 1.80, 1.79, 1.73, 1.45])
I1_crit = stretch_crit**2 + 2 / stretch_crit
print(I1_crit)
z = np.polyfit(cs, I1_crit, 2)
p = np.poly1d(z)
plt.figure(figsize=(4, 3), dpi=300)
plt.scatter(
    cs,
    I1_crit,
    c="#A3A3A3",
    label="Experimental values",
    # ls=":",
    # lw=4.5,
)
x = np.linspace(np.min(cs), np.max(cs), 100)
plt.plot(
    x,
    p(x),
    c="#5171A5",
    label="Least squares fitting",
    ls="-",
    lw=2.5,
)
print(p)
yhat = p(cs)  # or [p(z) for z in x]
ybar = np.sum(I1_crit) / len(I1_crit)  # or sum(y)/len(y)
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((np.array(I1_crit) - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r2 = ssreg / sstot
print(r2)
print(z)
formula = (
    r"$I_1^{\text{crit}}=$"
    # + f"{z[0]:.3f}"
    # + r"$c^4$"
    # + f"{r'$+$' if z[1]>0 else r'$-$'}{abs(z[1]):.3f}"
    # + r"$c^3$"
    + f"{r'' if z[0]>0 else r'$-$'}{abs(z[0]):.3f}"
    + r"$c^2$"
    # + "\n      "
    + f"{r'$+$' if z[1]>0 else r'$-$'}{abs(z[1]):.3f}"
    + r"$c$"
    + f"{r'$+$' if z[2]>0 else r'$-$'}{abs(z[2]):.3f}"
)
plt.text(0.95, 0.7, formula, ha="right", transform=plt.gca().transAxes)
plt.xlabel("Composition " + r"$c$")
plt.ylabel(r"$I_1^{\text{crit}}$")

plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
plt.savefig("I_1_crit.pdf")
plt.show()
plt.close()
