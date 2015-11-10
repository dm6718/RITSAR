#Include dependencies
import numpy as np
from numpy import pi
from numpy.linalg import norm
from scipy.io import loadmat
from scipy.stats import linregress
from fnmatch import fnmatch
import os
import xml.etree.ElementTree as ET

def AFRL(directory, pol, start_az, n_az=3):
##############################################################################
#                                                                            #
#  This function reads in the AFRL *.mat files from the user supplied        #
#  directory and exports both the phs and a Python dictionary compatible     #
#  with ritsar.                                                              #
#                                                                            #
##############################################################################
    
    #Get filenames
    walker = os.walk(directory+'/'+pol)
    w = walker.__next__()
    prefix = '/'+pol+'/'+w[2][0][0:19]
    az_str = []
    fnames = []
    az = np.arange(start_az, start_az+n_az)
    [az_str.append(str('%03d_'%a))      for a in az]
    [fnames.append(directory+prefix+a+pol+'.mat') for a in az_str]
    
    #Grab n_az phase histories
    phs = []; platform = []
    for fname in fnames:
        #Convert MATLAB structure to Python dictionary
        MATdata = loadmat(fname)['data'][0][0]
        
        data =\
        {
        'fp'    :   MATdata[0],
        'freq'  :   MATdata[1][:,0],
        'x'     :   MATdata[2].T,
        'y'     :   MATdata[3].T,
        'z'     :   MATdata[4].T,
        'r0'    :   MATdata[5][0],
        'th'    :   MATdata[6][0],
        'phi'   :   MATdata[7][0],
        }
        
        #Define phase history
        phs_tmp     = data['fp'].T
        phs.append(phs_tmp)
        
        #Transform data to be compatible with ritsar
        c           = 299792458.0
        nsamples    = int(phs_tmp.shape[1])
        npulses     = int(phs_tmp.shape[0])
        freq        = data['freq']
        pos         = np.hstack((data['x'], data['y'], data['z']))
        k_r         = 4*pi*freq/c
        B_IF        = data['freq'].max()-data['freq'].min()
        delta_r     = c/(2*B_IF)
        delta_t     = 1.0/B_IF
        t           = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t
        
        chirprate, f_0, r, p, s\
                    = linregress(t, freq)
                    
        #Vector to scene center at synthetic aperture center
        if np.mod(npulses,2)>0:
            R_c = pos[npulses/2]
        else:
            R_c = np.mean(
                    pos[npulses/2-1:npulses/2+1],
                    axis = 0)
        
        #Save values to dictionary for export
        platform_tmp = \
        {
            'f_0'       :   f_0,
            'freq'      :   freq,
            'chirprate' :   chirprate,
            'B_IF'      :   B_IF,
            'nsamples'  :   nsamples,
            'npulses'   :   npulses,
            'pos'       :   pos,
            'delta_r'   :   delta_r,
            'R_c'       :   R_c,
            't'         :   t,
            'k_r'       :   k_r,
        }
        platform.append(platform_tmp)
    
    #Stack data from different azimuth files
    phs = np.vstack(phs)
    npulses = int(phs.shape[0])
    
    pos = platform[0]['pos']
    for i in range(1, n_az):
        pos = np.vstack((pos, platform[i]['pos']))
                       
    if np.mod(npulses,2)>0:
        R_c = pos[npulses/2]
    else:
        R_c = np.mean(
                pos[npulses/2-1:npulses/2+1],
                axis = 0)
                       
    #Replace Dictionary values
    platform = platform_tmp
    platform['npulses'] =   npulses
    platform['pos']     =   pos
    platform['R_c']     =   R_c
    
    #Synthetic aperture length
    L = norm(pos[-1]-pos[0])

    #Add k_y
    platform['k_y'] = np.linspace(-npulses/2,npulses/2,npulses)*2*pi/L
    
    return(phs, platform)
    
def Sandia(directory):
##############################################################################
#                                                                            #
#  This function reads in the Sandia *.phs and *.au2 files from the user     #
#  supplied directoryand exports both the phs and a Python dictionary        #
#  compatible with ritsar.                                                   #
#                                                                            #
##############################################################################
    
    #get filename containing auxilliary data
    for file in os.listdir(directory):
            if fnmatch(file, '*.au2'):
                aux_fname = directory+file    
    
    #import auxillary data
    f=open(aux_fname,'rb')
    
    #initialize tuple
    record=['blank'] #first record blank to ensure
                     #indices match record numbers
    
    #record 1
    data = np.fromfile(f, dtype = np.dtype([
        ('version','S6'),
        ('phtype','S6'),
        ('phmode','S6'),
        ('phgrid','S6'),
        ('phscal','S6'),
        ('cbps','S6')
        ]),count=1)
    record.append(data[0])
    
    #record 2
    f.seek(44)
    data = np.fromfile(f, dtype = np.dtype([
        ('npulses','i4'),
        ('nsamples','i4'),
        ('ipp_start','i4'),
        ('ddas','f4',(5,)),
        ('kamb','i4')
        ]),count=1)
    record.append(data[0])
    
    #record 3    
    f.seek(44*2)
    data = np.fromfile(f, dtype = np.dtype([
        ('fpn','f4',(3,)),
        ('grp','f4',(3,)),
        ('cdpstr','f4'),
        ('cdpstp','f4')
        ]),count=1)
    record.append(data[0])
    
    #record 4
    f.seek(44*3)
    data = np.fromfile(f, dtype = np.dtype([
        ('f0','f4'),
        ('fs','f4'),
        ('fdot','f4'),
        ('r0','f4')
        ]),count=1)
    record.append(data[0])
    
    #record 5 (blank)rvr_au_read.py
    f.seek(44*4)
    data = []
    record.append(data)
    
    #record 6
    npulses = record[2]['npulses']
    rpoint = np.zeros([npulses,3])
    deltar = np.zeros([npulses,])
    fscale = np.zeros([npulses,])
    c_stab = np.zeros([npulses,3])
    #build up arrays for record(npulses+6)
    for n in range(npulses):
        f.seek((n+5)*44)
        data = np.fromfile(f, dtype = np.dtype([
            ('rpoint','f4',(3,)),
            ('deltar','f4'),
            ('fscale','f4'),
            ('c_stab','f8',(3,))
            ]),count=1)
        rpoint[n,:] = data[0]['rpoint']
        deltar[n] = data[0]['deltar']
        fscale[n] = data[0]['fscale']
        c_stab[n,:] = data[0]['c_stab']
    #consolidate arrays into a 'data' dataype
    dt = np.dtype([
            ('rpoint','f4',(npulses,3)),
            ('deltar','f4',(npulses,)),
            ('fscale','f4',(npulses,)),
            ('c_stab','f8',(npulses,3))
            ])        
    data = np.array((rpoint,deltar,fscale,c_stab)
            ,dtype=dt)
    #write to record file
    record.append(data)
    
    #import phase history
    for file in os.listdir(directory):
        if fnmatch(file, '*.phs'):
            phs_fname = directory+file
            
    nsamples = record[2][1]
    npulses = record[2][0]
    
    f=open(phs_fname,'rb')    
    dt = np.dtype('i2')
        
    phs = np.fromfile(f, dtype=dt, count=-1)
    real = phs[0::2].reshape([npulses,nsamples])  
    imag = phs[1::2].reshape([npulses,nsamples])
    phs = real+1j*imag
    
    #Create platform dictionary
    c       = 299792458.0
    pos     = record[6]['rpoint']
    n_hat   = record[3]['fpn']
    delta_t = record[4]['fs']
    t       = np.linspace(-nsamples/2, nsamples/2, nsamples)*1.0/delta_t
    chirprate = record[4]['fdot']*1.0/(2*pi)
    f_0     = record[4]['f0']*1.0/(2*pi) + chirprate*nsamples/(2*delta_t)
    B_IF    = (t.max()-t.min())*chirprate
    delta_r = c/(2*B_IF)
    freq = f_0+chirprate*t
    omega = 2*pi*freq
    k_r = 2*omega/c
    
    if np.mod(npulses,2)>0:
        R_c = pos[npulses/2]
    else:
        R_c = np.mean(
                pos[npulses/2-1:npulses/2+1],
                axis = 0)
    
    platform = \
    {
        'f_0'       :   f_0,
        'chirprate' :   chirprate,
        'B_IF'      :   B_IF,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'delta_r'   :   delta_r,
        'pos'       :   pos,
        'R_c'       :   R_c,
        't'         :   t,
        'k_r'       :   k_r,
        'n_hat'     :   n_hat
    }
    
    return(phs, platform)


##############################################################################
#                                                                            #
#  This function reads in the DIRSIG xml data as well as the envi header     #
#  file from the user supplied directory. The phs and a Python dictionary    #
#  compatible with ritsar are returned to the function caller.               #
#                                                                            #
##############################################################################
def get(root, entry):
    for entry in root.iter(entry):
        out = entry.text
        
    return(out)

def getWildcard(directory, char):
    for file in os.listdir(directory):
            if fnmatch(file, char):
                fname = directory+file
    
    return(fname)

def DIRSIG(directory):
    from spectral.io import envi
    
    #get phase history
    phs_fname = getWildcard(directory, '*.hdr')
    phs = envi.open(phs_fname).load(dtype = np.complex128)[:,:,0]
    
    #get platform geometry
    ppd_fname = getWildcard(directory, '*.ppd')
    tree = ET.parse(ppd_fname)
    root = tree.getroot()
    
    pos_dirs = []
    for children in root.iter('point'):
        pos_dirs.append(float(children[0].text))
        pos_dirs.append(float(children[1].text))
        pos_dirs.append(float(children[2].text))
    pos_dirs = np.asarray(pos_dirs).reshape([len(pos_dirs)/3,3])
    
    t_dirs=[]
    for children in root.iter('datetime'):
        t_dirs.append(float(children.text))
    t_dirs = np.asarray(t_dirs)
    
    #get platform system paramters
    platform_fname = getWildcard(directory, '*.platform')
    tree = ET.parse(platform_fname)
    root = tree.getroot()
    
    #put metadata into a dictionary
    metadata = root[0]
    keys = []; vals = []
    for children in metadata:
        keys.append(children[0].text)
        vals.append(children[1].text)
    metadata = dict(zip(keys,vals))
    
    #obtain key parameters
    c           = 299792458.0
    nsamples    = int(phs.shape[1])
    npulses     = int(phs.shape[0])
    vp          = float(get(root, 'speed'))
    delta_t     = float(get(root, 'delta'))
    t           = np.linspace(-nsamples/2, nsamples/2, nsamples)*delta_t
    prf         = float(get(root, 'clockrate'))
    chirprate   = float(get(root, 'chirprate'))/2
    T_p         = float(get(root, 'pulseduration'))
    B           = T_p*chirprate
    B_IF        = (t.max() - t.min())*chirprate
    delta_r     = c/(2*B_IF)
    f_0         = float(get(root, 'center'))*1e9
    freq        = f_0+chirprate*t
    omega       = 2*pi*freq
    k_r         = 2*omega/c
    T0          = float(get(root, 'min'))
    T1          = float(get(root, 'max'))
    
    #compute slowtime position
    ti = np.linspace(0,1.0/prf*npulses, npulses)
    x = np.array([np.interp(ti, t_dirs, pos_dirs[:,0])]).T
    y = np.array([np.interp(ti, t_dirs, pos_dirs[:,1])]).T
    z = np.array([np.interp(ti, t_dirs, pos_dirs[:,2])]).T
    pos = np.hstack((x,y,z))
    L = norm(pos[-1]-pos[0])
    k_y = np.linspace(-npulses/2,npulses/2,npulses)*2*pi/L
    
    #Vector to scene center at synthetic aperture center
    if np.mod(npulses,2)>0:
        R_c = pos[npulses/2]
    else:
        R_c = np.mean(
                pos[npulses/2-1:npulses/2+1],
                axis = 0)
                
    #Derived Parameters
    if np.mod(nsamples,2)==0:
        T = np.arange(T0, T1+0*delta_t, delta_t)
    else:
        T = np.arange(T0, T1, delta_t)
    
    #Mix signal
    signal = np.zeros(phs.shape)+0j
    for i in range(0,npulses,1):
        r_0 = norm(pos[i])
        tau_c = 2*r_0/c
        ref = np.exp(-1j*(2*pi*f_0*(T-tau_c)+2*chirprate*(T-tau_c)**2))
        signal[i,:] = ref*phs[i,:]
    
    platform = \
    {
        'f_0'       :   f_0,
        'freq'      :   freq,
        'chirprate' :   chirprate,
        'B'         :   B,
        'B_IF'      :   B_IF,
        'nsamples'  :   nsamples,
        'npulses'   :   npulses,
        'delta_r'   :   delta_r,
        'delta_t'   :   delta_t,
        'vp'        :   vp,
        'pos'       :   pos,
        'R_c'       :   R_c,
        't'         :   t,
        'k_r'       :   k_r,
        'k_y'       :   k_y,
        'metadata'  :   metadata
    }
    
    return(signal, platform)
