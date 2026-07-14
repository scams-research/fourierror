import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy import stats
import os

filename = input("Enter the name of the file to load: ")
data = np.loadtxt(filename)
t = data[:, 0]
signal = data[:, 1]
sigma_t = data[:, 2]

#def dft(data):
    #"""
    #Discrete Fourier transform
    #"""
    #pass

# Enter constants here
N = t.size # number of time-domain data points
L = t[t.size-1] # length of time-domain timescale i.e. 0 <= t <= L (with N points)
M = 500 # number of samples taken
Q = 5 # length of angular frequency domain i.e. 0 <= w <= Q (with P points)
w = np.linspace(0,Q,N)
cov = np.diag(((sigma_t)*np.sqrt(N))**2)
# create a covariance matrix with (error*sqrt(N))**2
# size = N x N

dt = L / (N-1)

def f_signal(w, signal, dt):
    """
    For an angular frequency w, what is the complex scalar contribution to the signal

    Args:
        w : angular frequency

    Returns:
        f_w = F(w) : the contribution of w to signal
    """
    w_column = w[:, np.newaxis] # column vector!! (w0, w1, w2... w(P-1))
    # column vector !! size = N x 1

    kdelta_t = dt*np.arange(N) # (0dt, 1dt, 2dt... (N-1)dt)
    # size = N

    matrix = w_column * kdelta_t # use matrix[i] to select the "i"th column (for a given w)
    # size = P x N (rows: testing w, column: kdt)

    return (2 / L) * (dt) * np.sum(   signal*(np.exp(-1 * 1j * matrix))   , axis=1)
    # i.e. f_matrix(w,signal) = [ F(w0,i)   F(w1,i)    F(w2,i)     ...]

mv_norm = stats.multivariate_normal(mean=signal, cov=cov)
# define a N-dimensional normal distribution centred around the mean y(t) value at each of the N data points

sample_signals = mv_norm.rvs(size=M)
# creates a nested array of M sub-arrays (each being a random sample) with each sub-array being size = N
# size = M x N

master_array = []
for i in sample_signals:
    master_array.append(abs(f_signal(w, i, dt)))
# master_array has M subarrays (samples) each with N elements (w to be tested) shape = (M,N)

def error(array):
    """
    Returns an array of standard errors of each w value across all M samples

    Args :
        array : master array (nested) of M sub-arrays and N elements

    Returns :
        error(array) : each element is the standard covariance-removed error in each w value
    """
    cov = np.cov(master_array, rowvar=False)
    # rowvar = False takes the variance across all subarrays for a given index rather than within a sub array
    # size = N x N
    var = np.diag(cov)
    # ignores the off-diagonal elements collapsing the 2D matrix into its diagonal; a 1D array
    # size = N
    sd = np.sqrt(var)
    # size = N
    return sd / np.sqrt(M)

name_without_ext = os.path.splitext(filename)[0]
output_filename = f"{name_without_ext}_output_num.txt"
np.savetxt(
f"{output_filename}.txt",
np.column_stack([w, abs(f_signal(w, signal, dt)), error(master_array)]),
header="omega\t|F(omega)|\tsigma_F",
fmt="%.8e",
delimiter="\t",
)

plt.plot(w,error(master_array))
plt.xlabel("Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$")
plt.ylabel("Error in F($\omega$)")
plt.show()

for i in sample_signals:
    plt.plot(w, abs(f_signal(w, i, dt)), 'o', alpha=(1/M), color='blue')
    # plot each f(w) as a faint marker showing the density of data points about each point
plt.xlabel("Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$")
plt.ylabel("F($\omega$)")
plt.show()
# plot a graph of F(w) against x (notice the wavelike nature of F(w) c.f. heisenberg uncertainty principle)

# Format the code
ruff format .

# Fix linting issues
ruff check . --fix

# Commit the changes
git add .
git commit -m "Fix ruff linting and formatting issues"
git push
