##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using AFRL Gotcha data.     #
#  Algorithms can be switched in and out by commenting/uncommenting          #
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

#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
pol = 'HH'
directory = './data/AFRL/pass1/'+pol+'/'
start_az = 1

#Import phase history and create platform dictionary
[phs, platform] = phsRead.AFRL(directory, pol, start_az)

#Correct for reisdual video phase
phs_corr = phsTools.RVP_correct(phs, platform)#*\
        #np.exp(1j*platform['af_ph']/(4*np.pi))

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.5, upsample = False)

#Apply algorithm of choice to phase history data
img_bp = imgTools.backprojection(phs_corr, platform, img_plane, taylor = 43, upsample = 6)
img_pf = imgTools.polar_format(phs_corr, platform, img_plane, taylor = 43)

#Output image
plt.imshow(np.abs(img_bp)**(0.5), cmap = cmap)