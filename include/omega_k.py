##############################################################################
#                                                                            #
#  This is an omega-k algorithm based off of the algorithm prescribed in the #
#  Carrera text.  Only the phase history and platform files are taken as     #
#  inputs, an img_plane dictionary is not required.                          #
#                                                                            #
#  The input phase history needs to have ben demodulated to a fixed          #
#  reference.  If demodulated to scene scenter, the included phs_const_ref   #
#  file can do the conversion for you.  A straight line flight path is       #
#  also assumed.                                                             #
#                                                                            #
#  The first step in the algorithm is to perfrom a 1D FT along azimuth.      #
#  A mathced filter is applied to the resultant data to perfectly compensate #
#  The range curvature of all scatterers having minimum range R_s.  The      #
#  default setting for R_s is the minimum range of scene center.  To         #
#  correct the range curvature for other scatterers, the data is mapped      #
#  onto a new grid.                                                          #
#                                                                            #
##############################################################################

#Include depedencies
import numpy as np
from numpy import exp, sqrt
from numpy.linalg import norm
import signal_processing as sig
from phs_inscribe import inscribe

def wk(phs, platform, img_plane, taylor = 43, upsample = 6):
    
    #Retrieve relevent parameters
    K_r     =   platform['k_r']
    K_y     =   platform['k_y']
    R_s     =   norm(platform['pos'], axis = -1).min()
    nsamples=   platform['nsamples']
    npulses =   platform['npulses']
    
    #Take azimuth FFT
    S_Kx_Kr = sig.ft(phs, ax = 0)
    
    #Create K_r, K_y grid
    [K_r, K_y] = np.meshgrid(K_r, K_y)
    
    #Mathed filter for compensating range curvature
    phase_mf = -R_s*K_r + R_s*sqrt(K_r**2-K_y**2)
    phase_mf = np.nan_to_num(phase_mf)
    S_Kx_Kr_mf = S_Kx_Kr*exp(1j*phase_mf)
    
    #Stolt interpolation
    K_xi_max = np.nan_to_num(sqrt(K_r**2-K_y**2)).max()
    K_xi_min = np.nan_to_num(sqrt(K_r**2-K_y**2)).min()
    K_xi = np.linspace(K_xi_min, K_xi_max, nsamples)
    
    S = np.zeros([npulses,nsamples])+0j
    for i in xrange(npulses):
        K_x = np.nan_to_num(sqrt(K_r[i,:]**2-K_y[i,:]**2))
        S[i,:] += np.interp(K_xi, K_x, S_Kx_Kr_mf[i,:].real)+\
                1j*np.interp(K_xi, K_x, S_Kx_Kr_mf[i,:].imag)
    
    [p1,p2] = inscribe(np.abs(S))
    S_new = S[p1[1]:p2[1],
              p1[0]:p2[0]]
    
    #Create window
    win_x = sig.taylor(S_new.shape[1],taylor)
    win_x = np.tile(win_x, [S_new.shape[0],1])
    
    win_y = sig.taylor(S_new.shape[0],taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,S_new.shape[1]])
    
    win = win_x*win_y
    
    #Apply window
    S_win = S_new*win
    
    #Pad Spectrum
    length = 2**(int(np.log2(S_new.shape[0]*upsample))+1)
    pad_x = length-S_win.shape[1]
    pad_y = length-S_win.shape[0]
    S_pad = np.pad(S_win,((pad_y/2, pad_y/2),(pad_x/2,pad_x/2)), mode = 'constant')
    
    img = sig.ift2(S_pad)
    
    return(img)