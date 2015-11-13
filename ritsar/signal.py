##############################################################################
#                                                                            #
#  This library contains the basic signal processing functions to be used    #
#  with the PySAR module                                                     #
#                                                                            #
##############################################################################


import numpy as np
from numpy import pi, arccosh, sqrt, cos
from scipy.fftpack import fftshift, fft2, ifft2, fft, ifft
from scipy.signal import firwin, filtfilt

#all FT's assumed to be centered at the origin
def ft(f, ax=-1):
    F = fftshift(fft(fftshift(f), axis = ax))
    
    return F
    
def ift(F, ax = -1):
    f = fftshift(ifft(fftshift(F), axis = ax))
    
    return f

def ft2(f, delta=1):
    F = fftshift(fft2(fftshift(f)))*delta**2
    
    return(F)

def ift2(F, delta=1):
    N = F.shape[0]
    f = fftshift(ifft2(fftshift(F)))*(delta*N)**2
    
    return(f)

def RECT(t,T):
    f = np.zeros(len(t))
    f[(t/T<0.5) & (t/T >-0.5)] = 1
    
    return f
    
def taylor(nsamples, S_L=43):
    xi = np.linspace(-0.5, 0.5, nsamples)
    A = 1.0/pi*arccosh(10**(S_L*1.0/20))
    n_bar = int(2*A**2+0.5)+1
    sigma_p = n_bar/sqrt(A**2+(n_bar-0.5)**2)
    
    #Compute F_m
    m = np.arange(1,n_bar)
    n = np.arange(1,n_bar)
    F_m = np.zeros(n_bar-1)
    for i in m:
        num = 1
        den = 1
        for j in n:
            num = num*\
            (-1)**(i+1)*(1-i**2*1.0/sigma_p**2/(\
                            A**2+(j-0.5)**2))
            if i!=j:
                den = den*(1-i**2*1.0/j**2)
            
        F_m[i-1] = num/den
    
    w = np.ones(nsamples)
    for i in m:
        w += F_m[i-1]*cos(2*pi*i*xi)
    
    w = w/w.max()          
    return(w)
    
def upsample(f,size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    F = ft2(f)
    F_pad = np.pad(F, ((y_pad/2,y_pad/2+y_off),(x_pad/2, x_pad/2+x_off)),
                   mode = 'constant')
    f_up = ift2(F_pad)
    
    return(f_up)
    
def upsample1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    F = ft(f)
    F_pad = np.pad(F, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    f_up = ift(F_pad)
    
    return(f_up)
    
def pad1D(f, size):
    x_pad = size-f.size
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
    
    
    f_pad = np.pad(f, (x_pad/2, x_pad/2+x_off),
                   mode = 'constant')
    
    return(f_pad)

def pad(f, size):
    x_pad = size[1]-f.shape[1]
    y_pad = size[0]-f.shape[0]
    
    if np.mod(x_pad,2):
        x_off = 1
    else:
        x_off = 0
        
    if np.mod(y_pad,2):
        y_off = 1
    else:
        y_off = 0
    
    f_pad = np.pad(f, ((y_pad//2,y_pad//2+y_off),(x_pad//2, x_pad//2+x_off)),
                   mode = 'constant')
    
    return(f_pad)
    
def cart2sph(cart):
    x = np.array([cart[:,0]]).T
    y = np.array([cart[:,1]]).T
    z = np.array([cart[:,2]]).T
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    sph = np.hstack([azimuth, elevation, r])
    return sph
    
def sph2cart(sph):
    azimuth     = np.array([sph[:,0]]).T
    elevation   = np.array([sph[:,1]]).T
    r           = np.array([sph[:,2]]).T
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    cart = np.hstack([x,y,z])
    return cart
    
def decimate(x, q, n=None, axis=-1, beta = None, cutoff = 'nyq'):
    if not isinstance(q, int):
        raise TypeError("q must be an integer")
        
    if n == None:
        n = int(np.log2(x.shape[axis]))
        
    if x.shape[axis] < n:
        n = x.shape[axis]-1
    
    if beta == None:
        beta = 1.*n/8
    
    padlen = n/2
    
    if cutoff == 'nyq':
        eps = np.finfo(np.float).eps
        cutoff = 1.-eps
    
    window = ('kaiser', beta)
    a = 1.
    
    b = firwin(n,  cutoff/ q, window=window)
    y = filtfilt(b, [a], x, axis=axis, padlen = padlen)
    
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(None, None, q)
    return y[sl]