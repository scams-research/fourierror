import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from scipy import stats
import os

filename = input("Enter the name of the file to load: ")
# prompts user to enter name of file in directory
data = np.loadtxt(filename)
# loads data from text file as a nested array with size (N,3)
t = data[:, 0]
# takes the first [0] value from each subarray (time)
signal = data[:, 1]
# takes the second [1] value from each subarray (y(t))
sigma_t = data[:, 2]
# takes the third [2] value from each subarray (error in y(t))

#def dft(data):
    #"""
    #Discrete Fourier transform
    #"""
    #pass

# Enter constants here
N = t.size # number of time-domain data points
L = t[t.size-1] # length of time-domain timescale i.e. 0 <= t <= L (with N points)
M = input("Enter the number of random samples you wish to take") # number of samples taken
Q = input("Enter the maximum angular frequency (minimum being zero) you wish to test") # length of angular frequency domain i.e. 0 <= w <= Q (with P points)
w = np.linspace(0,Q,N)
# creates a testing angular frequency domain from 0 to Q with N elements
cov = np.diag(((sigma_t)*np.sqrt(N))**2)
# create a covariance matrix with (error*sqrt(N))**2
# size = N x N

def delta_t(N,L):
    """
    The step in t between consecutive data points from 0 to L

    Args:
        N : number of discrete data points
        L : length of time scale measured to be analysed

    Returns:
        delta_t
    """
    return L / (N-1)

dt = delta_t(N,L)
# refer to this as dt from now on as this is constant

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
# removes the .txt part of input filename
output_filename = f"{name_without_ext}_output.txt"
# rewrites the output file in terms of the input file
np.savetxt(
f"{output_filename}.txt",
np.column_stack([w, abs(f_signal(w, signal, dt)), error(master_array)]),
header="omega\t|F(omega)|\tsigma_F",
fmt="%.8e",
delimiter="\t",
)
# creates a file called inputfile_output.txt in the same place as input file

plt.plot(w,error(master_array))
plt.xlabel("Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$")
plt.ylabel("Error in F($\omega$)")
plt.show()
# optional but displays the error in each F(w) value graphically

for i in sample_signals:
    plt.plot(w, abs(f_signal(w, i, dt)), 'o', alpha=(1/M), color='blue')
    # plot each f(w) as a faint marker showing the density of data points about each point
plt.xlabel("Angular frequency, $\omega$ / $\mathrm{rad}\,\mathrm{s}^{-1}$")
plt.ylabel("F($\omega$)")
plt.show()
# also optional but plot a graph of F(w) against x (notice the wavelike nature of F(w) c.f. heisenberg uncertainty principle)
