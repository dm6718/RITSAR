##############################################################################
#                                                                            #
#  Corrects Residual Video Phase using the formulation in Carrera Appendix C #
#                                                                            #
##############################################################################

#Include dependencies
from numpy import exp, pi
import numpy as np
import signal_processing as sig

def RVP_corr(phs, platform):
    
    #Retrieve relevent parameters
    c       =   3.0e8
    gamma   =   platform['chirprate']
    nsamples=   platform['nsamples']
    npulses =   platform['npulses']
    dr      =   platform['delta_r']
    
    #Calculate frequency sample locations w.r.t. demodulated fast time
    f_t = np.linspace(-nsamples/2, nsamples/2, nsamples)*\
            2*gamma/c*dr
    
    #Calculate correction factor
    S_c = exp(-1j*pi*f_t**2/gamma)  
    S_c2 = np.tile(S_c,[npulses,1])
    
    #Filter original signal    
    PHS = sig.ft(phs)
    phs_corr = sig.ift(PHS*S_c2)     
    
    return [phs_corr, platform]