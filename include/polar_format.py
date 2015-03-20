##############################################################################
#                                                                            #
#  This is the Polar Format algorithm.  The phase history data as well as    #
#  platform and image plane dictionaries are taken as inputs.                #
#                                                                            #
#  The phase history data is collected on a two-dimensional surface in       #
#  k-space.  For each pulse, a strip of this surface is collected.  The      #
#  first step in this program is to project each strip onto the (ku,kv)      #
#  plane defined by the normal vector contained in the image plane           #
#  dictionary.  This will result in data that is unevenly spaced in (ku,kv). # 
#  This unevenly spaced data is interpolated onto an evenly spaced (ku,kv)   #
#  grid defined in the image plane dictionary.  The interpolation is done    #
#  along the radial direction first, then along the along-track direction.   #
#  Further details of this method are given in both the Jakowitz and Carrera #
#  texts.                                                                    #
#                                                                            #
##############################################################################

#Include depedencies
import numpy as np
from numpy import dot, pi
from numpy.linalg import norm
import signal_processing as sig

def pf(phs, platform, img_plane, taylor = 43):
    
    #Retrieve relevent parameters
    c           =   3.0e8
    npulses     =   platform['npulses']
    f_0         =   platform['f_0']
    pos         =   np.asarray(platform['pos'])
    k           =   platform['k_r']
    R_c         =   platform['R_c']
    n_hat       =   img_plane['n_hat']
    k_ui        =   img_plane['k_u']
    k_vi        =   img_plane['k_v']
    
    #Compute k_xi offset
    psi = pi/2-np.arccos(np.dot(R_c,n_hat)/norm(R_c))
    k_ui = k_ui + 4*pi*f_0/c*np.cos(psi)
    
    #Compute number of samples in scene
    nu = k_ui.size
    nv = k_vi.size
    
    #Compute x and y unit vectors. x defined to lie along R_c.
    #z = cross(vec[0], vec[-1]); z =z/norm(z)
    u_hat = (R_c-dot(R_c,n_hat)*n_hat)/\
            norm((R_c-dot(R_c,n_hat)*n_hat))
    v_hat = np.cross(u_hat,n_hat)
    
    #Compute r_hat, the diretion of k_r, for each pulse
    r_norm = norm(pos,axis=1)
    r_norm = np.array([r_norm]).T
    r_norm = np.tile(r_norm,(1,3))
    
    r_hat = pos/r_norm
    
    #Convert to matrices to make projections easier
    r_hat = np.asmatrix(r_hat)
    u_hat = np.asmatrix([u_hat])
    v_hat = np.asmatrix([v_hat])
    
    k_matrix = np.tile(k,(npulses,1))
    k_matrix = np.asmatrix(k)
    
    #Compute kx and ky meshgrid
    ku = r_hat*u_hat.T*k_matrix; ku = np.asarray(ku)
    kv = r_hat*v_hat.T*k_matrix; kv = np.asarray(kv)
    
    #Create taylor window
    win = sig.taylor(nu, S_L = taylor)
    
    #Radially interpolate kx and ky data from polar raster
    #onto evenly spaced kx_i and ky_i grid for each pulse
    real_rad_interp = np.zeros([npulses,nu])
    imag_rad_interp = np.zeros([npulses,nu])
    ky_new = np.zeros([npulses,nu])
    for i in xrange(npulses):
        print('range interpolating for pulse %i'%(i+1))
        real_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.real[i,:]*win, left = 0, right = 0)
        imag_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.imag[i,:]*win, left = 0, right = 0)
        ky_new[i,:] = np.interp(k_ui, ku[i,:], kv[i,:], left = 0, right = 0)
    
    #Create taylor window
    win = sig.taylor(nv, S_L = taylor)    
    
    #Interpolate in along track direction to obtain polar formatted data
    real_polar = np.zeros([nv,nu])
    imag_polar = np.zeros([nv,nu])
    for i in xrange(nv):
        print('cross-range interpolating for sample %i'%(i+1))
        real_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
            real_rad_interp[::-1,i]*win, left = 0, right = 0)
        imag_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
            imag_rad_interp[::-1,i]*win, left = 0, right = 0)    
    
    real_polar = np.nan_to_num(real_polar)
    imag_polar = np.nan_to_num(imag_polar)    
    phs_polar = np.nan_to_num(real_polar+1j*imag_polar)
    
    img = np.abs(sig.ift2(phs_polar))
    
    return(img, phs_polar)