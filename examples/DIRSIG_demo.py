##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using simulated DIRSIG      #
#  data.  Algorithms can be switched in and out by commenting/uncommenting   #
#  the lines of code below.                                                  #
#                                                                            #
##############################################################################

#Add include directories to default path list
from sys import path
path.append('../')

#Include standard library dependencies
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
cmap = cm.Greys_r

#Include SARIT toolset
from ritsar import phsRead
from ritsar import phsTools
from ritsar import imgTools

#Define directory containing *.au2 and *.phs files
directory = './data/DIRSIG/'

#Import phase history and create platform dictionary
[phs, platform] = phsRead.DIRSIG(directory)

#Correct for reisdual video phase
phs_corr = phsTools.RVP_correct(phs, platform)

#Demodulate phase history with constant reference, if needed 
phs_fixed = phsTools.phs_to_const_ref(phs_corr, platform, upchirp = 1)

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.0, aspect = 2.0)

#Apply polar format algorithm to phase history data
img_wk = imgTools.omega_k(phs_fixed, platform, taylor = 43, upsample = 2)
#img_bp = imgTools.backprojection(phs_corr, platform, img_plane, taylor = 43, upsample = 6)

#Output image
plt.imshow(np.abs(img_wk), cmap = cmap)