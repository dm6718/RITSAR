#Include dependencies
import numpy as np
from numpy import exp, pi
from numpy.linalg import norm
from . import signal as sig
from . signal import RECT

def simulate_phs(platform, points = [[0,0,0]], amplitudes = [1]):
##############################################################################
#                                                                            #
#  This file takes a list of target locations and amplitudes and saves the   #
#  demodulated signal to './phase_history.npy'. It also outputs the signal   #
#  to the function caller.                                                   #
#                                                                            #
##############################################################################
    
    #Retrieve relevent parameters
    c       =   3.0e8
    gamma   =   platform['chirprate']
    f_0     =   platform['f_0']
    t       =   platform['t']
    pos     =   platform['pos']
    npulses =   platform['npulses']
    nsamples=   platform['nsamples']
    T_p     =   platform['T_p']
    
    #Simulate the phase history for each pulse, for each point
    phs = np.zeros([npulses, nsamples])+0j
    for i in xrange(npulses):
        print('simulating pulse %i'%(i+1))
        
        R_0 = norm(pos[i])
        j=0
        for p in points:
            
            R_t = norm(pos[i]-p)
            dr  = R_t-R_0
            phase = pi*gamma*(2*dr/c)**2-\
                    2*pi*(f_0+gamma*t)*2*dr/c
            
            phs[i,:] += amplitudes[j]*exp(1j*phase)*RECT((t-2*dr/c),T_p)
            
            j+=1
    
    np.save('./phase_history.npy', phs)
    
    return(phs)


def RVP_correct(phs, platform):
##############################################################################
#                                                                            #
#  Corrects Residual Video Phase using the formulation in Carrera Appendix C #
#                                                                            #
##############################################################################
  
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
    
    return (phs_corr)


def phs_to_const_ref(phs, platform, upchirp = 1):
##############################################################################
#                                                                            #
#  This program converts a phase history that was demodulated using a pulse  #
#  dependant range to scene center to a phase history that is demodulated    #
#  using a fixed reference.  The fixed reference is defined as the minimum   #
#  range to scene center.                                                    #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    c       =   3.0e8
    f0      =   platform['f_0']
    gamma   =   platform['chirprate']
    pos     =   platform['pos']
    t       =   platform['t']
    npulses =   platform['npulses']
        
    #Define ranges to scene center and new reference range
    #using Carrera's notation
    R_a = norm(pos, axis = -1)    
    R_a = np.array([R_a]).T
    R_s = R_a.min()
    DR = R_a-R_s
    
    #Derive fast time using constant reference
    t = np.tile(t,(npulses,1))
    
    #Remove original reference and incorporate new reference into phs
    sgn = (-1)**(upchirp)
    phs = phs*exp(sgn*1j*4*pi*gamma/c*(f0/gamma+t)*DR)
                     
    return(phs)