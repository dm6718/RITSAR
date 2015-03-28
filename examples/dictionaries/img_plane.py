##############################################################################
#                                                                            #
#  This file defines the image plane parameters.  The platform dictionary    #
#  values are included for convenience.  They can be used to specify sample  #
#  spacings in terms of the inherent resolution limits of the system.        #
#  u and v are defined with respect to the image plane frame of reference,   #
#  not the scene frame of reference.                                         #
#                                                                            #
#  'n_hat' defines the image plane orientation w.r.t. to the nominal         # 
#   ground plane                                                             #
#                                                                            #
##############################################################################

#Include dependencies
import numpy as np
from numpy import pi
from numpy.linalg import norm

def img_plane_dict(platform):
    
    #Import relevant platform parameters
    R_c = platform['R_c']
    
    #Define normal vector to image plane
    n_hat = np.array([0,0,1])    
    
    #Define resolution.  This should be less than the system resolution limits
    du = 0.5*platform['delta_r']
    dv = 1.0*du

    #Define image plane parameters
    nsamples = platform['nsamples']
    npulses = platform['npulses']    
    u = np.linspace(-nsamples/2, nsamples/2, nsamples)*du
    v = np.linspace(-npulses/2, npulses/2, npulses)*dv
    
    #Derive image plane spatial frequencies
    k_u = 2*pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nsamples)
    k_v = 2*pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), npulses)
    
    #Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c)/norm(np.cross(n_hat, R_c))
    u_hat = np.cross(v_hat, n_hat)/norm(np.cross(v_hat, n_hat))
    
    #Represent u and v in (x,y,z)
    [uu,vv] = np.meshgrid(u,v)
    uu = uu.flatten(); vv = vv.flatten()
    
    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T 
            )))            
    b = np.asmatrix(np.vstack((uu,vv)))
    pixel_locs = np.asarray(A*b)
    
    #Construct dictionary and return to caller
    img_plane =\
    {
    'n_hat'     :   n_hat,
    'du'        :   du,
    'dv'        :   dv,
    'u'         :   u,
    'v'         :   v,
    'k_u'       :   k_u,
    'k_v'       :   k_v,
    'pixel_locs':   pixel_locs # 3 x N_pixel array specifying x,y,z location
                               # of each pixel
    }

    return(img_plane)