##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar autofocusing algorithm.  The Sandia #
#  dataset is used to generate the original image.  A randomly generated     #
#  10th order polynomial is used to create a phse error which is             #
#  subsequently used to degrade the image in the along-track direction.      #
#  This degraded image is then passed to the auto_focusing algorithm.        #
#  The corrected image as well as the phase estimate is returned at relevant #
#  plots are generated.                                                      #
#                                                                            #
##############################################################################

#Add include directories to default path list
from sys import path
path.append('../')

#Include standard library dependencies
import numpy as np
import matplotlib.pylab as plt
from scipy.stats import linregress

#Include SARIT toolset
from ritsar import phsRead
from ritsar import phsTools
from ritsar import imgTools
from ritsar import signal as sig

#Define directory containing *.au2 and *.phs files
directory = './data/Sandia/'

#Import phase history and create platform dictionary
[phs, platform] = phsRead.Sandia(directory)

#Correct for reisdual video phase
phs_corr = phsTools.RVP_correct(phs, platform)

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform,
                           res_factor = 1.0, n_hat = platform['n_hat'])

#Apply polar format algorithm to phase history data
#(Other options not available since platform position is unknown)
img_pf = imgTools.polar_format(phs_corr, platform, img_plane, taylor = 30)

#Degrade image with random 10th order polynomial phase
coeff = (np.random.rand(10)-0.5)*img_pf.shape[0]
x = np.linspace(-1,1,img_pf.shape[0])
y = np.poly1d(coeff)(x)
slope, intercept, r_value, p_value, std_err = linregress(x,y)
line = slope*x+np.mean(y)
y = y-line
ph_err = np.tile(np.array([y]).T,(1,img_pf.shape[1]))
img_err = sig.ft(sig.ift(img_pf,ax=0)*np.exp(1j*ph_err),ax=0)

#Autofocus image
print('autofocusing')
img_af, af_ph = imgTools.autoFocus2(img_err, win = 'auto')
#img_af, af_ph = imgTools.autoFocus2(img_err, win = 0, win_params = [500,0.8])

#Output image
plt.figure()
plt.plot(x,y,x,af_ph); plt.legend(['true error','estimated error'], loc = 'best')
plt.ylabel('Phase (radians)')

#Output image
plt.figure()
imgTools.imshow(img_af, dB_scale = [-45,0])
