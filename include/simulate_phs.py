##############################################################################
#                                                                            #
#  This file takes a list of target locations and amplitudes and outputs the #
#  demodulated signal to './phase_history.npy'.                              #
#                                                                            #
##############################################################################

#Include dependencies
from signal_processing import RECT
import numpy as np
from numpy import exp, pi
from numpy.linalg import norm

def sim_phs(platform, points = [[0,0,0]], amplitudes = [1], window = 1):
    
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
            
            if window:
                phs[i,:] += amplitudes[j]*exp(1j*phase)*RECT((t-2*dr/c),T_p)
            else:
                phs[i,:] += amplitudes[j]*exp(1j*phase)
            
            j+=1
    
    np.save('./phase_history.npy', phs)