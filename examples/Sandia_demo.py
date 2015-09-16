##############################################################################
#                                                                            #
#  This is a demonstration of the ritsar toolset using Sandia data.          #
#  Algorithms can be switched in and out by commenting/uncommenting          #
#  the lines of code below.                                                  #
#                                                                            #
##############################################################################

#Add include directories to default path list
from sys import path
path.append('../')

#Include SARIT toolset
from ritsar import phsRead
from ritsar import phsTools
from ritsar import imgTools

#Define directory containing *.au2 and *.phs files
directory = './data/Sandia/'

#Import phase history and create platform dictionary
[phs, platform] = phsRead.Sandia(directory)

#Correct for residual video phase
phs_corr = phsTools.RVP_correct(phs, platform)

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform,
                           res_factor = 1.0, n_hat = platform['n_hat'])

#Apply polar format algorithm to phase history data
#(Other options not available since platform position is unknown)
img_pf = imgTools.polar_format(phs_corr, platform, img_plane, taylor = 30)

#Output image
imgTools.imshow(img_pf, [-45,0])