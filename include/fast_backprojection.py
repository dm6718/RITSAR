##############################################################################
#                                                                            #
#  This is the Fast Backprojection algorithm.  The phase history data as     #
#  well as platform and image plane dictionaries are taken as inputs.        #
#                                                                            #
#  This algorithm is similar to the regular backprojection algorithm.        #
#  Processing is broken up into sub apertures.  For each subaperture,        #
#  data is backprojected onto a polar grid.  Sampling requirements along     #
#  azimuth are significantly relaxed and processing can be performed on      #
#  a grid sparse in azimuth.  After a subaperture has been processed,        #
#  resuts are interpolated onto the user define img_plane.                   #
#                                                                            #
##############################################################################

#Include depedencies
import scipy
import scipy.interpolate
import numpy as np
from numpy import matrix
from numpy.linalg import norm
import signal_processing as sig

def bp(phs, platform, img_plane, subap_size = 0, taylor = 43, upsample = 6):

    #Retrieve relevent parameters
    c = 3.0e8
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   np.asarray(platform['pos'])
    delta_r     =   platform['delta_r']
    f           =   platform['freq']
    u           =   img_plane['u']
    v           =   img_plane['v']
    p           =   img_plane['pixel_locs']
    
    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples/2]
    
    #Derive representation of u_hat and v_hat in (x,y,z) space
    dr = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_r
    
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
    F = sig.ft(phs_pad)    
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)
    
    #Default subaperture size to be sqrt(npulses)
    if not(subap_size):
        subap_size = int(np.sqrt(npulses))
    
    #Perform fast backprojection
    img = np.zeros([nv,nu])+0j
    subap_index = 0
    for i in xrange(0, npulses, subap_size):
        #Calculate subaperture parameters
        ######################################################################
        print("Calculating parameters for subaperture %i" %(subap_index+1))
        interval = np.arange(subap_index*subap_size, (subap_index+1)*subap_size)
        
        #Subaperture position        
        q       = pos[interval].T
        q_sn    = np.array([q[:,subap_size/2]]).T
        l       = norm(q[:,-1]-q[:,0])
        
        #r
        r_max   = norm(p-q_sn, axis = 0).max()
        r_min   = norm(p-q_sn, axis = 0).min()
        r       = np.arange(r_min, r_max, 0.8*delta_r)
        
        #alpha
        q_hat       = (q[:,-1]-q[:,0])/norm(q[:,-1]-q[:,0])
        q_hat       = np.matrix([q_hat]).T
        alpha_tmp   = np.array(matrix(p-q_sn).T*q_hat)[:,0]/norm(p-q_sn, axis = 0)
        alpha_max   = alpha_tmp.max()
        alpha_min   = alpha_tmp.min()
        d_alpha     = c/(2*f.max()*l)
        alpha       = np.arange(alpha_min, alpha_max, 0.2*d_alpha)
        
        #Create (r,alpha) grid
        [rr,aa] = np.meshgrid(r, alpha)
        rr      = rr.flatten()
        aa      = aa.flatten()
        
        #Perform backprojection algorithm on subaperture
        ######################################################################
        sub_img = np.zeros(rr.size)+0j
        for j in xrange(subap_size):
            print('backprojecting pulse %i, subaperture %i'%((i+j+1),(subap_index+1)))
            
            #Define interpolation coordinates
            xi = norm(q[:,j]-q_sn[:,0])*(-1)**(2*j/subap_size+1)
            dr_i = np.sqrt(rr**2+xi**2-2*rr*xi*aa)-norm(q[:,j])
            
            #Perform backprojection on pulse
            I_tmp = F[interval][j,::-1]
            I = np.interp(dr_i, dr, I_tmp.real)+\
                1j*np.interp(dr_i, dr, I_tmp.imag)
            sub_img += I*np.exp(1j*k_c*dr_i)
            
        #Remove carrier
        sub_img = (sub_img*np.exp(-1j*k_c*(rr-norm(q_sn[:,0]))))\
                .reshape([alpha.size,r.size])
        #Upsample polar image
        Nr = max(2**int(np.log2(r.size*upsample)),
                 r.size)
        Na = max(2**int(np.log2(alpha.size*upsample)),
                 alpha.size)
        size = [Na, Nr]
        
        sub_img_up = np.abs(sig.upsample(sub_img, size))
        
        #Redefine upsampled polar coordinates
        r_up = np.linspace(r_min, r_max, Nr)
        alpha_up = np.linspace(alpha_min, alpha_max, Na)
        
        #Interpolate polar data from sub aperture onto cartesian image grid 
        print('Interpolating polar data onto image plane')
        r_i = norm(p-q_sn, axis = 0)        
        a_i = alpha_tmp
        
        #Use RectBivariateSpline to interpolate
        f_real = scipy.interpolate.RectBivariateSpline(
                   alpha_up, r_up, sub_img_up.real, kx=1, ky=1)
        f_imag = scipy.interpolate.RectBivariateSpline(
                   alpha_up, r_up, sub_img_up.imag, kx=1, ky=1)
                   
        img_tmp = f_real.ev(a_i,r_i)+1j*f_imag.ev(a_i,r_i)                   
                   
        img += (img_tmp*np.exp(1j*k_c*(r_i-norm(q_sn[:,0]))))\
                .reshape([nv,nu])
            
        subap_index += 1
        
    return(img)