import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

filename = input("Enter the name of the file to load: ")
data = np.loadtxt(filename)
t = data[:, 0]
y_t = data[:, 1]
error_y = data[:, 2]

# Constants used listed here:
N = t.size # number of time-domain data points
L = t[t.size-1] # length of time-domain timescale i.e. 0 <= t <= L (with N points)
Q = 5 # length of angular frequency domain i.e. 0 <= w <= Q (with N points)
w = np.linspace(0,Q,N) # angular frequency domain i.e. 0 <= w <= Q (with N points)
cov_y = np.diag(((error_y)*np.sqrt(N))**2)
# create a covariance matrix of the input error
# size = N x N

dt = L / (N-1)
# step in t between consecutive data points between t(n) and t(n+1)

w_column = w[:, np.newaxis] # column vector of (w0, w1, w2... w(P-1))
# size = N

time_domain = dt*np.arange(N) # (0dt, 1dt, 2dt... (N-1)dt)
# size = N

w_cross_t_matrix = w_column * time_domain
# size = N x N (rows: w, column: ndt)

A = (2 / L) * np.exp( -1j * w_cross_t_matrix ) * dt
# gives the matrix A (c.f. f = Ax) with size = N x N

f_w = A @ y_t
# gives the fourier transform at each w with size = N

cov_f = A @ cov_y @ A.conj().T
# cov_f = A*cov_y*A(T) but use @ to do matrix style multiplication instead of element-wise

plt.plot(t,y_t, color="red")
plt.xlabel("Time / s")
plt.ylabel("y(t)")
plt.show()
# optional but shows the signal as a function of time

plt.plot(w,abs(f_w), color="blue", label="F($\omega$)")
plt.ylabel("F($\omega$)")
plt.show()
# optional but shows the signal as a function of angular frequency

fig, ax = plt.subplots(figsize=(7, 6))
# draws a figure and a set of axes with figure size (7,6) 
ax.set_xlabel("Data point, j")
ax.set_ylabel("Data point, i")
ax.set_title("Magnitude of covariance matrix")
im = ax.imshow(np.abs(cov_f), cmap='viridis', origin='upper',
               extent=[0, N-1, N-1, 0])
fig.colorbar(im, ax=ax, label='Covariance')
plt.show()
# optional but shows the covariance matrix as a heatmap

name_without_ext = os.path.splitext(filename)[0]
output_filename = f"{name_without_ext}_output_anal.txt"
np.savetxt(
f"{output_filename}.txt",
np.column_stack([w, abs(f_w), (np.sqrt(abs(np.diag(cov_f))))/np.sqrt(N)]),
header="omega\t|F(omega)|\terror_F",
fmt="%.8e",
delimiter="\t",
)

# Fix linting issues
ruff check . --fix

# Commit the changes
git add .
git commit -m "Fix ruff linting and formatting issues"
git push
