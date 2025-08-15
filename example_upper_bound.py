import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{amsfonts}\usepackage{amssymb}\usepackage{amsmath}'
})

# Data and parameters
x = np.linspace(1e-8, 0.5, 1000)  # avoid sqrt(negative)
epsilon = 0.1

# Gaussian expression (f1)
f1 = (
    2 * x**2
    - 2 * np.sqrt(x**4 + epsilon**2 / 16.0)
    + (epsilon / 2.0) * (1.0 - np.log(epsilon))
    + (epsilon / 2.0) * np.log(2.0 * np.sqrt(x**4 + epsilon**2 / 16.0) + epsilon / 2.0)
)

# Bernoulli expression (f2)
p = 0.5 + np.sqrt(np.maximum(0.0, 0.25 - x**2))
p = np.clip(p, 1e-16, 1 - 1e-16)  # avoid log(0)
f2 = epsilon * (-p * np.log(p) - (1 - p) * np.log(1 - p))

# x-axis is sigma^2
sigma2 = x**2

# Plot
fig, ax = plt.subplots(figsize=(20/2.54, 7/2.54))  # convert cm->inch
ax.plot(sigma2, f1, 'b-', linewidth=1, label=r'$\mathbb{P}_1,\mathbb{P}_2\sim\mathcal{N}(0,\sigma^2)$')
ax.plot(sigma2, f2, 'r--', linewidth=1, label='Upper bound')
ax.fill_between(sigma2, f2, 0, color='red', alpha=0.3, edgecolor='none', label=r'$\mathbb{P}_1, \mathbb{P}_2\sim\mathcal{B}(p)$')

ax.set_xlabel(r'$\sigma^2$', fontsize=12)
ax.set_ylabel(r'$W_\epsilon({\mathbb{P}}_1, {\mathbb{P}}_2)$', fontsize=12)
ax.set_xlim(min(sigma2), max(sigma2))
ax.set_ylim(0, max(f1)*1.1)
ax.set_yticks(np.arange(0, .15, 0.05))
ax.legend(loc='best', fontsize=12)
ax.grid(True)
plt.tight_layout()

# Save as vector PDF
fig.savefig('plot_linear_policies.pdf', bbox_inches='tight')
plt.show()
