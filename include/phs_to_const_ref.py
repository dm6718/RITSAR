##############################################################################
#                                                                            #
#  This program converts a phase history that was demodulated using a pulse  #
#  dependant range to scene center to a phase history that is demodulated    #
#  using a fixed reference.  The fixed reference is defined as the minimum   #
#  range to scene center.                                                    #
#                                                                            #
##############################################################################


#Include dependencies
import numpy as np
from numpy import exp, pi, sin
from numpy.linalg import norm
import signal_processing as sig

def phs_const_ref(phs, platform, upchirp = 1):
    
    #Retrieve relevent parameters
    c       =   3.0e8
    f0      =   platform['f_0']
    gamma   =   platform['chirprate']
    pos     =   platform['pos']
    t       =   platform['t']
    npulses =   platform['npulses']
    nsamples=   platform['nsamples']
    dr      =   platform['delta_r']
    theta   =   np.array([platform['theta']]).T
        
    #Define ranges to scene center and new reference range
    #using Carrera's notation
    R_a = norm(pos, axis = -1)    
    R_a = np.array([R_a]).T
    R_s = R_a.min()
    DR = R_a-R_s
    
    #Derive fast time using constant reference
    t = np.tile(t,(npulses,1))
    t_new = t+2*DR/c
    
    #Remove original reference and incorporate new reference into phs
    sgn = (-1)**(upchirp)
    phs = phs*exp(sgn*1j*4*pi*gamma/c*(f0/gamma+t)*DR)
                     
    return([phs, platform])