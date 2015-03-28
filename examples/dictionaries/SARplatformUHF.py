##############################################################################
#                                                                            #
#  This file is where all platform parameters are defined.  All coordinates  #
#  are assumed to have the scene center at the origin of a cartesian         #
#  coordinate system (ECF, [lat,long,zenith], for example).                  #
#                                                                            #
#  If passing in data from an axillary file, replace the definitions         #
#  below with expressions that translate the axillary data labels            #
#  into the dictionary keys listed at the bottom of the program.             #
#                                                                            #
##############################################################################

#Include dependencies
import numpy as np
from numpy import pi, cos, sin, arccos
from numpy.linalg import norm

def plat_dict(aux = []):
    #Define universal constants
    c = 3.0e8
    
    #Define platform parameters
    f_0 = 242.4e6   #center frequency
    wvl = c/f_0
    chirprate = 33.375e12
    T_p = 4e-6
    B = T_p*chirprate
    AD_sampling = 256e6
    delta_t = 1.0/AD_sampling
    nsamples = 2048
    npulses = 1268
    squint = pi/2
    graze = 0
    
    #Define platform position
    
    #x
    R0_apCenter = 1000.0
    Xc = -R0_apCenter*cos(graze)
    
    #y
    y = np.linspace(-npulses/2, npulses/2, npulses)*0.6
    
    #z    
    h = R0_apCenter*sin(graze)
    
    #r[npulses , (x,y,z)]
    pos = np.zeros([npulses, 3])
    pos[:,0] = Xc
    pos[:,1] = y
    pos[:,2] = h
    
    #Synthetic aperture length
    L = norm(pos[-1]-pos[0])
    
    #Vector to scene center at synthetic aperture center
    if np.mod(npulses,2)>0:
        R_c = pos[npulses/2]
    else:
        R_c = np.mean(
                pos[npulses/2-1:npulses/2+1],
                axis = 0)
                    
    #Coherent integration angle
    p1_hat = pos[0]/norm(pos[0]); p2_hat = pos[-1]/norm(pos[-1])
    Delta_theta = arccos(np.dot(p1_hat, p2_hat))
    
    #Look angle w.r.t. velocity vector for each pulse (minus pi/2)
    theta = np.zeros(npulses)
    R_vel = pos[-1]-pos[0]
    R_vel_norm = norm(R_vel)
    for i in xrange(npulses):
        num = np.dot(pos[i], R_vel)
        den = norm(pos[i])*R_vel_norm
        theta[i] = arccos(num/den)-pi/2
        
    #Obtain frequency locations of range samples
    #Assuming demodulated lienar FM signal
    t = np.linspace(
        -nsamples/2, nsamples/2, nsamples)*delta_t #demodulated fast time
    B_IF = (t.max()-t.min())*chirprate
    delta_r = c/(2*B_IF)
    freq = f_0+chirprate*t
    omega = 2*pi*freq
    k_r = 2*omega/c
    k_y = np.linspace(-npulses/2,npulses/2,npulses)*2*pi/L
    
    #Construct dictionary and return to caller
    platform = \
    {
        'f_0'       :   f_0,
        'wvl'       :   wvl,
        'chirprate' :   chirprate,
        'T_p'       :   T_p,
        'B'         :   B,
        'B_IF'      :   B_IF,
        'AD_sampling':  AD_sampling,        
        't'         :   t,
        'delta_t'   :   delta_t,        
        'delta_r'   :   delta_r,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'pos'       :   pos,
        'R_c'       :   R_c,
        'L'         :   L,
        'squint'    :   squint,
        'graze'     :   graze,
        'Delta_theta':  Delta_theta,
        'theta'     :   theta,
        'freq'      :   freq,
        'omega'     :   omega,
        'k_r'       :   k_r,
        'k_y'       :   k_y
    }
    
    return(platform)