#Include depedencies
import cv2
import numpy as np
from numpy import dot, pi, exp, sqrt, inf
from numpy.linalg import norm
import matplotlib.pylab as plt
from scipy.stats import linregress
from matplotlib import cm
import signal as sig
from scipy.interpolate import interp1d

def phs_inscribe(img):
##############################################################################
#                                                                            #
#  This program enables a user to inscribe data from a signal's spectrum.    #
#  The user clicks and drags from the top-left to bottom-right and the       #
#  corner points are output.  OpenCV is required.                            #
#                                                                            #
##############################################################################

    
    #Define behavior for mouse callback function
    def onMouse(event,x,y, flags, param):
        pos1    = param[0]
        pos2    = param[1]
        img_new = param[2]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            param[0] = (x,y)
            pos1     = param[0]
            
        elif event == cv2.EVENT_LBUTTONUP:
            param[1] = (x,y)
            pos2     = param[1]
            img_out = img_new[pos1[1]:pos2[1],
                              pos1[0]:pos2[0]]
            cv2.imshow('win', img_out)
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            img_out = img_new
            cv2.imshow('win', img_out)
        
    #Create inscribe function
    print('\
Click and drag from top-left to bottom-right\n\
to inscribe phase history \n\n\
Right-click to reset image\n\
Press enter when finished\
         ')
    
    #Scale intensity values for an 8-bit display
    img_new = img-img.min()
    img_new = img_new/img_new.max()
    
    #Confortably size window for a 1920 x 1080 display
    rows = int(1080/2)
    cols = int(1920/2)    
    
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('win', cols, rows)
    
    #Install mouse callback
    pos1 = (0,0)
    pos2 = (-1,-1)
    params = [pos1, pos2, img_new]
    cv2.setMouseCallback('win',onMouse, param = params)
    
    #Display image
    cv2.imshow('win',img_new)
    
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            cv2.destroyAllWindows()
            break

    return(params[0:2])


def polar_format(phs, platform, img_plane, taylor = 20):
##############################################################################
#                                                                            #
#  This is the Polar Format algorithm.  The phase history data as well as    #
#  platform and image plane dictionaries are taken as inputs.                #
#                                                                            #
#  The phase history data is collected on a two-dimensional surface in       #
#  k-space.  For each pulse, a strip of this surface is collected.  The      #
#  first step in this program is to project each strip onto the (ku,kv)      #
#  plane defined by the normal vector contained in the image plane           #
#  dictionary.  This will result in data that is unevenly spaced in (ku,kv). # 
#  This unevenly spaced data is interpolated onto an evenly spaced (ku,kv)   #
#  grid defined in the image plane dictionary.  The interpolation is done    #
#  along the radial direction first, then along the along-track direction.   #
#  Further details of this method are given in both the Jakowitz and Carrera #
#  texts.                                                                    #
#                                                                            #
##############################################################################
        
    #Retrieve relevent parameters
    c           =   3.0e8
    npulses     =   platform['npulses']
    f_0         =   platform['f_0']
    pos         =   np.asarray(platform['pos'])
    k           =   platform['k_r']
    R_c         =   platform['R_c']
    n_hat       =   img_plane['n_hat']
    k_ui        =   img_plane['k_u']
    k_vi        =   img_plane['k_v']
    
    #Compute k_xi offset
    psi = pi/2-np.arccos(np.dot(R_c,n_hat)/norm(R_c))
    k_ui = k_ui + 4*pi*f_0/c*np.cos(psi)
    
    #Compute number of samples in scene
    nu = k_ui.size
    nv = k_vi.size
    
    #Compute x and y unit vectors. x defined to lie along R_c.
    #z = cross(vec[0], vec[-1]); z =z/norm(z)
    u_hat = (R_c-dot(R_c,n_hat)*n_hat)/\
            norm((R_c-dot(R_c,n_hat)*n_hat))
    v_hat = np.cross(u_hat,n_hat)
    
    #Compute r_hat, the diretion of k_r, for each pulse
    r_norm = norm(pos,axis=1)
    r_norm = np.array([r_norm]).T
    r_norm = np.tile(r_norm,(1,3))
    
    r_hat = pos/r_norm
    
    #Convert to matrices to make projections easier
    r_hat = np.asmatrix(r_hat)
    u_hat = np.asmatrix([u_hat])
    v_hat = np.asmatrix([v_hat])
    
    k_matrix = np.tile(k,(npulses,1))
    k_matrix = np.asmatrix(k)
    
    #Compute kx and ky meshgrid
    ku = r_hat*u_hat.T*k_matrix; ku = np.asarray(ku)
    kv = r_hat*v_hat.T*k_matrix; kv = np.asarray(kv)
    
    #Create taylor windows
    win1 = sig.taylor(int(phs.shape[1]), S_L = taylor)
    win2 = sig.taylor(int(phs.shape[0]), S_L = taylor)
    
    #Radially interpolate kx and ky data from polar raster
    #onto evenly spaced kx_i and ky_i grid for each pulse
    real_rad_interp = np.zeros([npulses,nu])
    imag_rad_interp = np.zeros([npulses,nu])
    ky_new = np.zeros([npulses,nu])
    for i in range(npulses):
        print('range interpolating for pulse %i'%(i+1))
        real_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.real[i,:]*win1, left = 0, right = 0)
        imag_rad_interp[i,:] = np.interp(k_ui, ku[i,:], 
            phs.imag[i,:]*win1, left = 0, right = 0)
        ky_new[i,:] = np.interp(k_ui, ku[i,:], kv[i,:])  
    
    #Interpolate in along track direction to obtain polar formatted data
    real_polar = np.zeros([nv,nu])
    imag_polar = np.zeros([nv,nu])
    isSort = (ky_new[npulses/2, nu/2] < ky_new[npulses/2+1, nu/2])
    if isSort:
        for i in range(nu):
            print('cross-range interpolating for sample %i'%(i+1))
            real_polar[:,i] = np.interp(k_vi, ky_new[:,i], 
                real_rad_interp[:,i]*win2, left = 0, right = 0)
            imag_polar[:,i] = np.interp(k_vi, ky_new[:,i], 
                imag_rad_interp[:,i]*win2, left = 0, right = 0)
    else:
        for i in range(nu):
            print('cross-range interpolating for sample %i'%(i+1))
            real_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
                real_rad_interp[::-1,i]*win2, left = 0, right = 0)
            imag_polar[:,i] = np.interp(k_vi, ky_new[::-1,i], 
                imag_rad_interp[::-1,i]*win2, left = 0, right = 0)
    
    real_polar = np.nan_to_num(real_polar)
    imag_polar = np.nan_to_num(imag_polar)    
    phs_polar = np.nan_to_num(real_polar+1j*imag_polar)
    
    img = np.abs(sig.ft2(phs_polar))
    
    return(img)

def backprojection(phs, platform, img_plane, taylor = 20, upsample = 6):
##############################################################################
#                                                                            #
#  This is the Backprojection algorithm.  The phase history data as well as  #
#  platform and image plane dictionaries are taken as inputs.  The (x,y,z)   #
#  locations of each pixel are required, as well as the size of the final    #
#  image (interpreted as [size(v) x size(u)]).                               #
#                                                                            #
##############################################################################
    
    #Retrieve relevent parameters
    nsamples    =   platform['nsamples']
    npulses     =   platform['npulses']
    k_r         =   platform['k_r']
    pos         =   platform['pos']
    delta_r     =   platform['delta_r']
    u           =   img_plane['u']
    v           =   img_plane['v']
    r           =   img_plane['pixel_locs']
    
    #Derive parameters
    nu = u.size
    nv = v.size
    k_c = k_r[nsamples/2]
    
    #Create window
    win_x = sig.taylor(nsamples,taylor)
    win_x = np.tile(win_x, [npulses,1])
    
    win_y = sig.taylor(npulses,taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,nsamples])
    
    win = win_x*win_y
    
    #Filter phase history    
    filt = np.abs(k_r)
    phs_filt = phs*filt*win
    
    #Zero pad phase history
    N_fft = 2**(int(np.log2(nsamples*upsample))+1)
    phs_pad = sig.pad(phs_filt, [npulses,N_fft])
    
    #Filter phase history and perform FT w.r.t t
    Q = sig.ft(phs_pad)    
    dr = np.linspace(-nsamples*delta_r/2, nsamples*delta_r/2, N_fft)
    
    #Perform backprojection for each pulse
    img = np.zeros(nu*nv)+0j
    for i in range(npulses):
        print("Calculating backprojection for pulse %i" %i)
        
        r0 = np.array([pos[i]]).T
        dr_i = norm(r0)-norm(r-r0, axis = 0)
    
        Q_real = np.interp(dr_i, dr, Q[i].real)
        Q_imag = np.interp(dr_i, dr, Q[i].imag)
        
        Q_hat = Q_real+1j*Q_imag        
        img += Q_hat*np.exp(-1j*k_c*dr_i)
    
    r0 = np.array([pos[nsamples/2]]).T
    dr_i = norm(r0)-norm(r-r0, axis = 0)
    img = img*np.exp(-1j*k_c*dr_i)
    img = np.reshape(img, [nv, nu])[::-1,:]
    return(img)
    

def omega_k(phs, platform, taylor = 20, upsample = 6):
##############################################################################
#                                                                            #
#  This is an omega-k algorithm based off of the algorithm prescribed in the #
#  Carrera text.  Only the phase history and platform files are taken as     #
#  inputs, an img_plane dictionary is not required.                          #
#                                                                            #
#  The input phase history needs to have been demodulated to a fixed         #
#  reference.  If demodulated to scene center, the included phs_const_ref    #
#  file can do the conversion for you.  A straight line flight path is       #
#  also assumed.                                                             #
#                                                                            #
#  The first step in the algorithm is to perform a 1D FT along azimuth.      #
#  A matched filter is applied to the resultant data to perfectly compensate #
#  the range curvature of all scatterers having minimum range R_s.  The      #
#  default setting for R_s is the minimum range of scene center.  To         #
#  correct the range curvature for other scatterers, the data is mapped      #
#  onto a new grid.                                                          #
#                                                                            #
##############################################################################

    #Retrieve relevent parameters
    K_r     =   platform['k_r']
    K_y     =   platform['k_y']
    R_s     =   norm(platform['pos'], axis = -1).min()
    nsamples=   platform['nsamples']
    npulses =   platform['npulses']
    
    #Take azimuth FFT
    S_Kx_Kr = sig.ift(phs, ax = 0)
    
    #Create K_r, K_y grid
    [K_r, K_y] = np.meshgrid(K_r, K_y)
    
    #Mathed filter for compensating range curvature
    phase_mf = -R_s*K_r + R_s*sqrt(K_r**2-K_y**2)
    phase_mf = np.nan_to_num(phase_mf)
    S_Kx_Kr_mf = S_Kx_Kr*exp(1j*phase_mf)
    
    #Stolt interpolation
    K_xi_max = np.nan_to_num(sqrt(K_r**2-K_y**2)).max()
    K_xi_min = np.nan_to_num(sqrt(K_r**2-K_y**2)).min()
    K_xi = np.linspace(K_xi_min, K_xi_max, nsamples)
    
    S = np.zeros([npulses,nsamples])+0j
    for i in range(npulses):
        K_x = np.nan_to_num(sqrt(K_r[i,:]**2-K_y[i,:]**2))
        
        f_real = interp1d(K_x, S_Kx_Kr_mf[i,:].real, kind = 'linear', bounds_error = 0, fill_value = 0)
        f_imag = interp1d(K_x, S_Kx_Kr_mf[i,:].imag, kind = 'linear', bounds_error = 0, fill_value = 0)
        
        S[i,:] += f_real(K_xi) + 1j*f_imag(K_xi)
    
    S = np.nan_to_num(S)
    [p1,p2] = phs_inscribe(np.abs(S))
    S_new = S[p1[1]:p2[1],
              p1[0]:p2[0]]
    
    #Create window
    win_x = sig.taylor(S_new.shape[1],taylor)
    win_x = np.tile(win_x, [S_new.shape[0],1])
    
    win_y = sig.taylor(S_new.shape[0],taylor)
    win_y = np.array([win_y]).T
    win_y = np.tile(win_y, [1,S_new.shape[1]])
    
    win = win_x*win_y
    
    #Apply window
    S_win = S_new*win
    
    #Pad Spectrum
    length = 2**(int(np.log2(S_new.shape[0]*upsample))+1)
    pad_x = length-S_win.shape[1]
    pad_y = length-S_win.shape[0]
    S_pad = np.pad(S_win,((pad_y/2, pad_y/2),(pad_x/2,pad_x/2)), mode = 'constant')
    
    img = sig.ift2(S_pad)
    
    return(img)
    
def img_plane_dict(platform, res_factor=1.0, n_hat = np.array([0,0,1]), aspect = 0, upsample = True):
##############################################################################
#                                                                            #
#  This function defines the image plane parameters.  The user specifies the #
#  image resolution using the res_factor.  A res_factor of 1 yields a (u,v)  #
#  image plane whose pixels are sized at the theoretical resolution limit    #
#  of the system (derived using delta_r which in turn was derived using the  #
#  bandwidth.  The user can also adjust the aspect of the image grid.  This  #
#  defaults to nsamples/npulses.                                             #
#                                                                            #
#  'n_hat' is a user specified value that defines the image plane            #
#  orientation w.r.t. to the nominal ground plane.                           #
#                                                                            #
##############################################################################
    
    nsamples = platform['nsamples']
    npulses = platform['npulses']   
    if not(aspect):
        aspect = 1.0*nsamples/npulses
    else:
        npulses = nsamples/aspect
    
    #Import relevant platform parameters
    R_c = platform['R_c']    
    
    #Define resolution.  This should be less than the system resolution limits
    du = res_factor*platform['delta_r']
    dv = aspect*du
    
    #Define image plane parameters
    if upsample:
        nu= 2**int(np.log2(nsamples)+bool(np.mod(np.log2(nsamples),1)))
        nv= 2**int(np.log2(npulses)+bool(np.mod(np.log2(npulses),1)))
    else:
        nu= nsamples
        nv= npulses
        
    u = np.linspace(-nsamples/2, nsamples/2, nu)*du
    v = np.linspace(-npulses/2, npulses/2, nv)*dv
    
    #Derive image plane spatial frequencies
    k_u = 2*pi*np.linspace(-1.0/(2*du), 1.0/(2*du), nu)
    k_v = 2*pi*np.linspace(-1.0/(2*dv), 1.0/(2*dv), nv)
    
    #Derive representation of u_hat and v_hat in (x,y,z) space
    v_hat = np.cross(n_hat, R_c)/norm(np.cross(n_hat, R_c))
    u_hat = np.cross(v_hat, n_hat)/norm(np.cross(v_hat, n_hat))
    
    #Represent u and v in (x,y,z)
    [uu,vv] = np.meshgrid(u,v)
    uu = uu.flatten(); vv = vv.flatten()
    
    A = np.asmatrix(np.hstack((
        np.array([u_hat]).T, np.array([v_hat]).T 
            )))            
    b = np.asmatrix(np.vstack((uu,vv)))
    pixel_locs = np.asarray(A*b)
    
    #Construct dictionary and return to caller
    img_plane =\
    {
    'n_hat'     :   n_hat,
    'u_hat'     :   u_hat,
    'v_hat'     :   v_hat,
    'du'        :   du,
    'dv'        :   dv,
    'u'         :   u,
    'v'         :   v,
    'k_u'       :   k_u,
    'k_v'       :   k_v,
    'pixel_locs':   pixel_locs # 3 x N_pixel array specifying x,y,z location
                               # of each pixel
    }
    
    return(img_plane)
	
	
def autoFocus(img, win = 'auto', win_params = [100,0.5]):
##############################################################################
#                                                                            #
#  This program autofocuses an image using the Phase Gradient Algorithm.     #
#  If the parameter win is set to auto, an adaptive window is used.          #
#  Otherwise, the user sets win to 0 and defines win_params.  The first      #
#  element of win_params is the starting windows size.  The second element   #
#  is the factor by which to reduce it by for each iteration.  This version  #
#  is more suited for an image that is mostly focused.  Below is the paper   #
#  this algorithm is based off of.                                           #
#                                                                            #
#  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
#  autofocus-a robust tool for high resolution sar phase correction,"        #
#  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
#  pp. 827{835, Jul 1994.                                                    #
#                                                                            #
##############################################################################
    
    #Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 30
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1]
        
        #Circularly shift image so max values line up   
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], npulses/2-index[i])
        
        if win == 'auto':
            #Compute window width    
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            width = np.sum(s>-30)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
        else:
            #Compute window width using win_params if win not set to 'auto'    
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
            if width<5:
                break
        
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]
        
        #Fourier Transform
        G = sig.ift(g, ax=0)
        
        #take derivative
        G_dot = np.diff(G, axis=0)
        a = np.array([G_dot[-1,:]])
        G_dot = np.append(G_dot,a,axis = 0)
        
        #Estimate Spectrum for the derivative of the phase error
        phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
                  np.sum(np.abs(G)**2, axis = -1)
                
        #Integrate to obtain estimate of phase error(Jak)
        phi = np.cumsum(phi_dot)
        
        #Remove linear trend
        t = np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        rms.append(np.sqrt(np.mean(phi**2)))
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi    
       
    fig = plt.figure(figsize = (12,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('original')
    ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('autofocused')
    ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title('rms phase error vs. iteration')
    plt.ylabel('Phase (radians)')
    ax3.plot(rms)
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_title('phase error')
    plt.ylabel('Phase (radians)')
    ax4.plot(af_ph)
    plt.tight_layout()
    

    print('number of iterations: %i'%(iii+1))
                     
    return(img_af, af_ph)
    

def autoFocus2(img, win = 'auto', win_params = [100,0.5]):
##############################################################################
#                                                                            #
#  This program autofocuses an image using the Phase Gradient Algorithm.     #
#  If the parameter win is set to auto, an adaptive window is used.          #
#  Otherwise, the user sets win to 0 and defines win_params.  The first      #
#  element of win_params is the starting windows size.  The second element   #
#  is the factor by which to reduce it by for each iteration.  This version  #
#  is more suited for an image that is severely degraded (such as for the    #
#  auto_focusing demo)                                                       #
#  since, for the adaptive window, it uses most of the data for the first    #
#  few iterations.  Below is the paper this algorithm is based off of.       #
#                                                                            #
#  D. Wahl, P. Eichel, D. Ghiglia, and J. Jakowatz, C.V., \Phase gradient    #
#  autofocus-a robust tool for high resolution sar phase correction,"        #
#  Aerospace and Electronic Systems, IEEE Transactions on, vol. 30,          #
#  pp. 827{835, Jul 1994.                                                    #
#                                                                            #
##############################################################################
    
    #Derive parameters
    npulses = int(img.shape[0])
    nsamples = int(img.shape[1])
    
    #Initialize loop variables
    img_af = 1.0*img
    max_iter = 10
    af_ph = 0
    rms = []
    
    #Compute phase error and apply correction
    for iii in range(max_iter):
        
        #Find brightest azimuth sample in each range bin
        index = np.argsort(np.abs(img_af), axis=0)[-1]
        
        #Circularly shift image so max values line up   
        f = np.zeros(img.shape)+0j
        for i in range(nsamples):
            f[:,i] = np.roll(img_af[:,i], npulses/2-index[i])
        
        if win == 'auto':
            #Compute window width    
            s = np.sum(f*np.conj(f), axis = -1)
            s = 10*np.log10(s/s.max())
            #For first two iterations use all azimuth data 
            #and half of azimuth data, respectively
            if iii == 0:
                width = npulses
            elif iii == 1:
                width = npulses/2
            #For all other iterations, use twice the 30 dB threshold
            else:
                width = np.sum(s>-30)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
        else:
            #Compute window width using win_params if win not set to 'auto'    
            width = int(win_params[0]*win_params[1]**iii)
            window = np.arange(npulses/2-width/2,npulses/2+width/2)
            if width<5:
                break
        
        #Window image
        g = np.zeros(img.shape)+0j
        g[window] = f[window]
        
        #Fourier Transform
        G = sig.ift(g, ax=0)
        
        #take derivative
        G_dot = np.diff(G, axis=0)
        a = np.array([G_dot[-1,:]])
        G_dot = np.append(G_dot,a,axis = 0)
        
        #Estimate Spectrum for the derivative of the phase error
        phi_dot = np.sum((np.conj(G)*G_dot).imag, axis = -1)/\
                  np.sum(np.abs(G)**2, axis = -1)
                
        #Integrate to obtain estimate of phase error(Jak)
        phi = np.cumsum(phi_dot)
        
        #Remove linear trend
        t = np.arange(0,nsamples)
        slope, intercept, r_value, p_value, std_err = linregress(t,phi)
        line = slope*t+intercept
        phi = phi-line
        rms.append(np.sqrt(np.mean(phi**2)))
        
        if win == 'auto':
            if rms[iii]<0.1:
                break
        
        #Apply correction
        phi2 = np.tile(np.array([phi]).T,(1,nsamples))
        IMG_af = sig.ift(img_af, ax=0)
        IMG_af = IMG_af*np.exp(-1j*phi2)
        img_af = sig.ft(IMG_af, ax=0)
        
        #Store phase
        af_ph += phi    
       
    fig = plt.figure(figsize = (12,10))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('original')
    ax1.imshow(10*np.log10(np.abs(img)/np.abs(img).max()), cmap = cm.Greys_r)
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('autofocused')
    ax2.imshow(10*np.log10(np.abs(img_af)/np.abs(img_af).max()), cmap = cm.Greys_r)
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title('rms phase error vs. iteration')
    plt.ylabel('Phase (radians)')
    ax3.plot(rms)
    ax4 = fig.add_subplot(2,2,4)
    ax4.set_title('phase error')
    plt.ylabel('Phase (radians)')
    ax4.plot(af_ph)
    plt.tight_layout()
    

    print('number of iterations: %i'%(iii+1))
                     
    return(img_af, af_ph)
    
def imshow(img, dB_scale = [0,0]):
###############################################################################
#                                                                             #
#  This program displays the processed data in dB.  The brightest point in    #
#  the image is used as the reference and the user can define the scale for   #
#  the intensity range.                                                       #
#                                                                             #
###############################################################################

    #Convert to dB
    img = 10*np.log10(np.abs(img)/np.abs(img).max())
    img[img == -inf] = dB_scale[0]

    #Determine if the image is RGB
    if len(img.shape) != 3:
    
        #Display the image
        if dB_scale == [0,0]:
            plt.imshow(img, cmap=cm.Greys_r)
        else:
            plt.imshow(img, cmap=cm.Greys_r,
                       vmin = dB_scale[0], vmax = dB_scale[-1])
    
    #If the image is RGB                 
    else:
        #Display the image
        if dB_scale == [0,0]:
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB)
        else:
            img[img<=dB_scale[0]] = dB_scale[0]
            img[img>=dB_scale[-1]] = dB_scale[-1]
            img_RGB = (img-img.min())/(img.max()-img.min())
            plt.imshow(img_RGB)
