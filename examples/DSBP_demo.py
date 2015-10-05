##############################################################################
#                                                                            #
#  This is a demonstration of the Digital Spotlight Backprojection algorithm.#
#  Data sets can be switched in and out by commenting/uncommenting the lines #
#  of code below.                                                            #
#                                                                            #
##############################################################################

#Add include directories to default path list
from sys import path
path.append('../')
path.append('./dictionaries')

#Include Dictionaries
from SARplatform import plat_dict

#Include standard library dependencies
import matplotlib.pylab as plt
import numpy as np

#Include SARIT toolset
from ritsar import phsTools
from ritsar import phsRead
from ritsar import imgTools
'''
#simulated DSBP demo
##############################################################################
#Create platform dictionary
platform = plat_dict()

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, aspect = 1)

#Simulate phase history
nsamples = platform['nsamples']
npulses = platform['npulses']
x = img_plane['u']; y = img_plane['v']
points = [[0,0,0],
          [0,-100,0],
          [200,0,0]]
amplitudes = [1,1,1]
phs = phsTools.simulate_phs(platform, points, amplitudes)

#Apply RVP correction
phs_corr = phsTools.RVP_correct(phs, platform)

#Apply algorithm of choice to phase history data
img_bp   = imgTools.backprojection(phs_corr, platform, img_plane, taylor = 17, upsample = 2)
img_DSBP = imgTools.DSBP(phs_corr, platform, img_plane, center = [200,0,0], size = [1000,1000])

#Output image
du = img_plane['du']; dv = img_plane['dv']
#u = img_plane['u']; v = img_plane['v']
u = np.arange(-1000/2,1000/2)*du
v = np.arange(-1000/2,1000/2)*dv
extent = [u.min(), u.max(), v.min(), v.max()]

plt.subplot(1,2,1)
plt.title('Full Backprojection')
imgTools.imshow(img_bp[1024-500:1024+500, 1315-500:1315+500], dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')

plt.subplot(1,2,2)
plt.title('Digital Spotlight Backprojection')
imgTools.imshow(img_DSBP, dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')
plt.tight_layout()

'''
#AFRL DSBP demo
###############################################################################
#Define top level directory containing *.mat file
#and choose polarization and starting azimuth
pol = 'HH'
directory = './data/AFRL/pass1'
start_az = 1

#Import phase history and create platform dictionary
[phs, platform] = phsRead.AFRL(directory, pol, start_az, n_az = 3)

#Create image plane dictionary
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.4, upsample = True, aspect = 1.0)

#Apply algorithm of choice to phase history data
img_bp   = imgTools.backprojection(phs, platform, img_plane, taylor = 17, upsample = 2)
img_DSBP = imgTools.DSBP(phs, platform, img_plane, center = [-15-0.6,22-0.4,0], size = [200,200], ftype = 'iir')

#Output image
du = img_plane['du']; dv = img_plane['dv']
#u = img_plane['u']; v = img_plane['v']
u = np.arange(-200/2,200/2)*du
v = np.arange(-200/2,200/2)*dv
extent = [u.min(), u.max(), v.min(), v.max()]

plt.subplot(1,2,1)
plt.title('Full Backprojection')
imgTools.imshow(img_bp[177-100:177+100,202-100:202+100], dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')

plt.subplot(1,2,2)
plt.title('Digital Spotlight Backprojection')
imgTools.imshow(img_DSBP, dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')
plt.tight_layout()
'''
#DIRSIG DSBP demo
###############################################################################
#Define directory containing *.au2 and *.phs files
directory = './data/DIRSIG/'

#Import phase history and create platform dictionary
[phs, platform] = phsRead.DIRSIG(directory)

#Correct for reisdual video phase
phs_corr = phsTools.RVP_correct(phs, platform)

#Import image plane dictionary from './parameters/img_plane'
img_plane = imgTools.img_plane_dict(platform, res_factor = 1.0, aspect = 1.0)

#Apply algorithm of choice to phase history data
img_bp   = imgTools.backprojection(phs_corr, platform, img_plane, taylor = 17, upsample = 2)
img_DSBP = imgTools.DSBP(phs_corr, platform, img_plane, center = [0,0,0], size = [300,300])

#Output image
du = img_plane['du']; dv = img_plane['dv']
#u = img_plane['u']; v = img_plane['v']
u = np.arange(-300/2,300/2)*du
v = np.arange(-300/2,300/2)*dv
extent = [u.min(), u.max(), v.min(), v.max()]

plt.subplot(1,2,1)
plt.title('Full Backprojection')
imgTools.imshow(img_bp[512-150:512+150, 512-150:512+150], dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')

plt.subplot(1,2,2)
plt.title('Digital Spotlight Backprojection')
imgTools.imshow(img_DSBP, dB_scale = [-25,0], extent = extent)
plt.xlabel('meters'); plt.ylabel('meters')
plt.tight_layout()'''