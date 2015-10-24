##############################################################################
#                                                                            #
#  This is a demonstration of the Fast Factorized Backprojection algorithm.  #
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
from time import time

#Include SARIT toolset
from ritsar import phsTools
from ritsar import phsRead
from ritsar import imgTools

if __name__ == '__main__':
    '''
    #simulated FFBP demo
    ##############################################################################
    #Create platform dictionary
    platform = plat_dict()
    
    #Create image plane dictionary
    img_plane = imgTools.img_plane_dict(platform, aspect = 1, res_factor=0.9)
    
    #Simulate phase history
    nsamples = platform['nsamples']
    npulses = platform['npulses']
    x = img_plane['u']; y = img_plane['v']
    points = [[0,0,0],
              [200,0,0],
              [0,100,0]]
    amplitudes = [1,1,1]
    phs = phsTools.simulate_phs(platform, points, amplitudes)
    
    #Apply RVP correction
    phs = phsTools.RVP_correct(phs, platform)
    
    #full backprojection
    start = time()
    img_bp   = imgTools.backprojection(phs, platform, img_plane, taylor = 17, upsample = 2)
    bp_time = time()-start
    
    #Fast-factorized backprojection without multi-processing
    start = time()
    img_FFBP = imgTools.FFBP(phs, platform, img_plane, taylor = 17, factor_max = 4)
    fbp_time = time()-start
    
    #Fast-factorized backprojection with multi-processing
    start = time()
    img_FFBP = imgTools.FFBPmp(phs, platform, img_plane, taylor = 17, factor_max = 4)
    fbpmp_time = time()-start
    
    #Output image
    u = img_plane['u']; v = img_plane['v']
    extent = [u.min(), u.max(), v.min(), v.max()]
    
    plt.subplot(2,1,1)
    plt.title('Full Backprojection \n \
    Runtime = %i s'%bp_time)
    imgTools.imshow(img_bp, dB_scale = [-25,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,3)
    plt.title('Fast Factorized Backprojection \n w/o multi-processing \n \
    Runtime = %i s'%fbp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-25,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,4)
    plt.title('Fast Factorized Backprojection \n w/ multi-processing \n \
    Runtime = %i s'%fbpmp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-25,0], extent = extent)
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
    [phs, platform] = phsRead.AFRL(directory, pol, start_az, n_az = 4)
    
    #Create image plane dictionary
    img_plane = imgTools.img_plane_dict(platform, res_factor = 1.0, upsample = True, aspect = 1.0)
    
    #full backprojection
    start = time()
    img_bp   = imgTools.backprojection(phs, platform, img_plane, taylor = 17, upsample = 2)
    bp_time = time()-start
    
    #Fast-factorized backprojection without multi-processing
    start = time()
    img_FFBP = imgTools.FFBP(phs, platform, img_plane, taylor = 17, factor_max = 2)
    fbp_time = time()-start
    
    #Fast-factorized backprojection with multi-processing
    start = time()
    img_FFBP = imgTools.FFBPmp(phs, platform, img_plane, taylor = 17, factor_max = 2)
    fbpmp_time = time()-start
    
    #Output image
    u = img_plane['u']; v = img_plane['v']
    extent = [u.min(), u.max(), v.min(), v.max()]
    
    plt.subplot(2,1,1)
    plt.title('Full Backprojection \n \
    Runtime = %i s'%bp_time)
    imgTools.imshow(img_bp, dB_scale = [-30,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,3)
    plt.title('Fast Factorized Backprojection \n w/o multi-processing \n \
    Runtime = %i s'%fbp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-30,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,4)
    plt.title('Fast Factorized Backprojection \n w/ multi-processing \n \
    Runtime = %i s'%fbpmp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-30,0], extent = extent)
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
    
    #full backprojection
    start = time()
    img_bp   = imgTools.backprojection(phs, platform, img_plane, taylor = 17, upsample = 2)
    bp_time = time()-start
    
    #Fast-factorized backprojection without multi-processing
    start = time()
    img_FFBP = imgTools.FFBP(phs, platform, img_plane, taylor = 17, factor_max = 4)
    fbp_time = time()-start
    
    #Fast-factorized backprojection with multi-processing
    start = time()
    img_FFBP = imgTools.FFBPmp(phs, platform, img_plane, taylor = 17, factor_max = 4)
    fbpmp_time = time()-start
    
    #Output image
    u = img_plane['u']; v = img_plane['v']
    extent = [u.min(), u.max(), v.min(), v.max()]
    
    plt.subplot(2,1,1)
    plt.title('Full Backprojection \n \
    Runtime = %i s'%bp_time)
    imgTools.imshow(img_bp, dB_scale = [-25,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,3)
    plt.title('Fast Factorized Backprojection \n w/o multi-processing \n \
    Runtime = %i s'%fbp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-25,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    
    plt.subplot(2,2,4)
    plt.title('Fast Factorized Backprojection \n w/ multi-processing \n \
    Runtime = %i s'%fbpmp_time)
    imgTools.imshow(img_FFBP, dB_scale = [-25,0], extent = extent)
    plt.xlabel('meters'); plt.ylabel('meters')
    plt.tight_layout()'''
