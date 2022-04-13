import numpy as np
from numpy.linalg import norm
import skimage.draw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

c = lambda *x: np.array(x, dtype=float)
vector_angle = lambda V, U: np.arccos(norm(np.dot(V, U)) / (norm(V) + norm(U)))

r = 10 # radius of cylinder
C0 = c(10, 10, 10) # first (x,y,z) point of cylinder
C1 = c(99, 90, 15) # second (x,y,z) point of cylinder

C = C1 - C0

X, Y, Z = np.eye(3)

theta = vector_angle(Z, C)
print('theta={} deg'.format(theta / np.pi * 180))

minor_axis = r
major_axis = r / np.cos(theta)
print('major_axis', major_axis)

alpha = vector_angle(X, C0 + C)
print('alpha={} deg'.format(alpha / np.pi * 180))

data = np.zeros([100, 100, 100])
nz, ny, nx = data.shape

for z in range(nz):
    lam = - (C0[2] - z)/C[2]
    P = C0 + C * lam
    y, x = skimage.draw.ellipse(P[1], P[0], major_axis, minor_axis, shape=(ny, nx), rotation=alpha)
    data[z, y, x] = 1


xs, ys, zs = np.nonzero(data)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(xs, ys, zs)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
