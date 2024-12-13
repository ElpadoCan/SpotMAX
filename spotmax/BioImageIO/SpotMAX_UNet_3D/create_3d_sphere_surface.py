import os

import matplotlib.pyplot as plt
import numpy as np

cwd_path = os.path.dirname(os.path.abspath(__file__))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Make data
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 10 * np.outer(np.cos(u), np.sin(v))
y = 10 * np.outer(np.sin(u), np.sin(v))
z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z)

# Set an equal aspect ratio
ax.set_aspect('equal')

fig_svg_path = os.path.join(cwd_path, '_sphere_3d.svg')
fig.savefig(fig_svg_path, dpi=300)

plt.show()

print('*'*100)
print(
    'Done. Cover saved at the following location:\n\n'
    f'  * {fig_svg_path}\n'
)
print('*'*100)

