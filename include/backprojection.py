##############################################################################
#                                                                            #
#  This is the Backprojection algorithm.  The phase history data as well as  #
#  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
#  locations of each pixel are required, as well as the size of the final    #
#  image (interpreted as [size(v) x size(u)]).                               #
#                                                                            #
##############################################################################

#Include depedencies
import numpy as np
from numpy.linalg import norm
import signal_processing as sig

def bp(phs, platform, img_plane, taylor = 43, upsample = 6):
    
    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']
    
    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples/2]
    
    #Create window
    win_x = sig.taylor(nsamples,taylor)
    win_x = np.tile(win_x, [npulses,1])
    
    win_y = sig.taylor(npulses,taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,nsamples])
    
    win = win_x*win_y
    
    #Filter phase history    
    filt = np.abs(k_r)
    phs_filt = phs*filt*win
    
    #Upsample phase history
    N_fft = 2**(int(np.log2(nsamples*upsample))+1)
    pad = N_fft-nsamples
    phs_pad = np.pad(phs_filt, ((0,0),(pad/2,pad/2)), mode = 'constant')
    
    #Filter phase history and perform FT w.r.t t
    Q = sig.ft(phs_pad)    
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)
    
    #Perform backprojection for each pulse
    img = np.zeros(nu*nv)+0j
    for i in xrange(npulses):
        print("Calculating backprojection for pulse %i" %i)
        
        r0 = np.array([pos[i]]).T
        dr_i = norm(r0)-norm(r-r0, axis = 0)
        
        Q_hat = np.interp(dr_i, dr, Q[i,::-1].real)+\
                1j*np.interp(dr_i, dr, Q[i,::-1].imag)        
        img += Q_hat*np.exp(1j*k_c*dr_i)
        
    img = np.reshape(img, [nv, nu])
    return(img)