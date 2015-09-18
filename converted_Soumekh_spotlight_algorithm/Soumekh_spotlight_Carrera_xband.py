##############################################################################
#                                                                            #
#  This code performs image reconstruction using Mehrdad Soumekh's Spatial   #
#  Frequency interpolation method. It has essentially been copied from his   #
#  MATLAB algorithm for spotlight processing and translated to Python.  The  #
#  parameters used in this file are based on the ones in the Carrera text	  #
#  for the X-band setup.  I could not get this to work due to memory errors. #
#  The A/D sampling requirements are 8 times higher to perform matched       #
#  filtering with the modulated, 185MHz bandwidth signal (plus a 2*f0 gaurd  #
#  band) as opposed to the sampling requirements needed for the demodulated  #
#  signal.  The higher carrier frequency (and subsequently doppler bandwidth)#
#  also results in a 14x increase in the number of synthetic aperture        #
#  samples required to perform the frequency mapping prescribed by Soumekh   #
#  as opposed to polar format mapping.                                       #
#                                                                            #
##############################################################################

from numpy import *
import numpy as np
import ritsar.signal as sig
from scipy.fftpack import *
from ritsar.signal import *
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
plt.set_cmap(cm.Greys)

   ##########################################################
   #   PULSED SPOTLIGHT SAR SIMULATION AND RECONSTRUCTION   #
   ##########################################################


cmap = cm.Greys_r
cj=1j;
pi2=2*pi;
#
c=3e8;                   # propagation speed
f0=185e6/2;                 # baseband bandwidth is 2*f0
w0=pi2*f0;
fc=10.0e9;                # carrier frequency
wc=pi2*fc;
lambda_min=c/(fc+f0);    # Wavelength at highest frequency
lambda_max=c/(fc-f0);    # Wavelength at lowest frequency
kc=(pi2*fc)/c;           # wavenumber at carrier frequency
kmin=(pi2*(fc-f0))/c;    # wavenumber at lowest frequency
kmax=(pi2*(fc+f0))/c;    # wavenumber at highest frequency
#
Xc=10000.;                 # Range distance to center of target area
X0=1050.;                   # target area in range is within [Xc-X0,Xc+X0]
Yc=0.;                  # Cross-range distance to center of target area
Y0=916.;                  # target area in cross-range is within
                         # [Yc-Y0,Yc+Y0]

# Case 1: L < Y0; requires zero-padding of SAR signal in synthetic
# aperture domain
#
L=195.0/2;                 # synthetic aperture is 2*L

# Case 2: L > Y0; slow-time Doppler subsampling of SAR signal spectrum
# reduces computation
#
# L=400;                 # synthetic aperture is 2*L

theta_c=arctan(1.0*Yc/Xc);     # Squint angle
Rc=sqrt(Xc**2+Yc**2);      # Squint radial range
L_min=max(Y0,L);         # Zero-padded aperture is 2*L_min

#
Xcc=Xc/(cos(theta_c)**2); # redefine Xc by Xcc for squint processing

##############################################################
## u domain parameters and arrays for compressed SAR signal ##
##############################################################
#
duc=(Xcc*lambda_min)/(4*Y0);      # sample spacing in aperture domain
                                  # for compressed SAR signal
duc=0.1#duc/1.2;                      # 10 percent guard band; this guard band
                                  # would not be sufficient for targets
                                  # outside digital spotlight filter (use
                                  # a larger guard band, i.e., PRF)
mc=1950#2*ceil(L_min/duc);             # number of samples on aperture
uc=duc*np.array([arange(-mc/2,mc/2)]).T;            # synthetic aperture array
dkuc=pi2/(mc*duc);                # sample spacing in ku domain
kuc=dkuc*np.array([arange(-mc/2,mc/2)]).T;          # kuc array
#
dku=1.0*dkuc;                         # sample spacing in ku domain

##########################################################
##    u domain parameters and arrays for SAR signal     ##
##########################################################
#
if Yc-Y0-L < 0:                            # minimum aspect angle
 theta_min=arctan(1.*(Yc-Y0-L)/(Xc-X0));
else:
 theta_min=arctan(1.*(Yc-Y0-L)/(Xc+X0));
#end;
theta_max=arctan(1.*(Yc+Y0+L)/(Xc-X0));         # maximum aspect angle
#
du=pi/(kmax*(sin(theta_max)-\
                     sin(theta_min))); # sample spacing in aperture
                                       # domain for SAR signal
du=du/1.4;                        # 20 percent guard band
m=2*ceil(pi/(du*dku));            # number of samples on aperture
du=pi2/(m*dku);                   # readjust du
u=du*np.array([arange(-m/2,m/2)]).T;                # synthetic aperture array
ku=dku*np.array([arange(-m/2,m/2)]).T;              # ku array


##########################################################
##       Fast-time domain parmeters and arrays          ##
##########################################################
#
Tp=38.5e-6;                     # Chirp pulse duration
alpha=w0/Tp;                   # Chirp rate
wcm=wc-alpha*Tp;               # Modified chirp carrier
#
if Yc-Y0-L < 0:
 Rmin=Xc-X0;
else:
 Rmin=sqrt((Xc-X0)**2+(Yc-Y0-L)**2);
#end;
Ts=(2/c)*Rmin;                 # start time of sampling
Rmax=sqrt((Xc+X0)**2+(Yc+Y0+L)**2);
Tf=(2/c)*Rmax+Tp;              # end time of sampling
T=Tf-Ts;                       # fast-time interval of measurement
Ts=Ts-.1*T;                    # start slightly earlier (10# guard band)
Tf=Tf+.1*T;                    # end slightly later (10# guard band)
T=Tf-Ts;
Tmin=max(T,(4*X0)/(c*cos(theta_max)));  # Minimum required T
#
dt=1/(4*f0);                 # Time domain sampling (guard band factor 2)
n=2*ceil((.5*Tmin)/dt);      # number of time samples
t=Ts+arange(0,n)*dt;             # time array for data acquisition
dw=pi2/(n*dt);               # Frequency domain sampling
w=wc+dw*arange(-n/2,n/2);        # Frequency array (centered at carrier)
k=w/c;                       # Wavenumber array
#

#############################################################
# Resolution for Broadside: (x,y) domain rotated by theta_c #
#############################################################

DX=1#c/(4*f0);                      # range resolution (broadside)
DY=1#(Xcc*lambda_max)/(4*L);         # cross-range resolution (broadside)


#####################################################
##           Parameters of Targets                 ##
#####################################################
#
ntarget=3;                        # number of targets
# Set ntarget=1 to see "clean" PSF of target at origin
# Try this with other targets

# xn: range;            yn= cross-range;    fn: reflectivity
xn=zeros(ntarget);  yn=1.0*xn;              fn=1.0*xn;

# Targets within digital spotlight filter
#
xn[1-1]=0.;             yn[1-1]=0.;           fn[1-1]=1.;
xn[2-1]=200.;           yn[2-1]=0.;           fn[2-1]=1.;
xn[3-1]=0.;             yn[3-1]=-100.;        fn[3-1]=1.;
#xn[4-1]=-.5*X0;         yn[4-1]=.75*Y0;       fn[4-1]=1.;
#xn[5-1]=-.5*X0+DX;      yn[5-1]=.75*Y0+DY;    fn[5-1]=1.;

# Targets outside digital spotlight filter
# (Run the code with and without these targets)
#  
#xn[6-1]=-1.2*X0;        yn[6-1]=.75*Y0;       fn[6-1]=1.;
#xn[7-1]=.5*X0;          yn[7-1]=1.25*Y0;      fn[7-1]=1.;
#xn[8-1]=1.1*X0;         yn[8-1]=-1.1*Y0;      fn[8-1]=1.;
#xn[9-1]=-1.2*X0;        yn[9-1]=-1.75*Y0;     fn[9-1]=1.;

#####################################################
##                   SIMULATION                    ##
#####################################################
#
s=zeros([mc,n])+0j;     # SAR signal array
#
for i in range(ntarget):   # Loop for each target
    for j in range(int(mc)):
        td=t-2*sqrt((Xc+xn[i])**2+(Yc+yn[i]-uc[j])**2)/c;
        s[j]+=fn[i]*exp(cj*wcm*td+cj*alpha*(td**2))*((td >= 0) & (td <= Tp) & \
        (abs(uc[j]) <= L) & (t < Tf))
#end;
#
s=s*exp(-cj*wc*t);      # Fast-time baseband conversion

# User may apply a slow-time domain window, e.g., power window, on
# simulated SAR signal array "s" here.

G=abs(s)
xg=np.max(np.max(G)); ng=np.min(np.min(G)); cg=255/(xg-ng);
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = (t.min()*1e6, t.max()*1e6, uc.min(), uc.max()), aspect = 'auto');
plt.xlabel('Fast-time t, $\mu$sec')
plt.ylabel('Synthetic Aperture (Slow-time) U, meters')
plt.title('Measured Spotlight SAR Signal')
#

td0=t-2*sqrt(Xc**2+Yc**2)/c;
s0=exp(cj*wcm*td0+cj*alpha*(td0**2))*((td0 >= 0) & (td0 <= Tp));
s0=s0*exp(-cj*wc*t);            # Baseband reference fast-time signal

s=sig.ft(s)*(conj(sig.ft(s0)));  # Fast-time matched filtering
#
G=abs(sig.ift(s));
xg=np.max(np.max(G)); ng=np.min(np.min(G)); cg=255/(xg-ng);
tm=(2*Rc/c)+dt*arange(-n/2,n/2);    # fast-time array after matched filtering
plt.figure()
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = (tm.min()*1e6, tm.max()*1e6, uc.min(), uc.max()), aspect = 'auto');
plt.xlabel('Fast-time t, sec')
plt.ylabel('Synthetic Aperture (Slow-time) U, meters')
plt.title('SAR Signal after Fast-time Matched Filtering')
#
#############################################
#  Slow-time baseband conversion for squint #
#############################################
#
kus=2*kc*sin(theta_c);     # Doppler frequency shift in ku
                                     # domain due to squint
#
s=s*exp(-cj*kus*uc);             # slow-time baseband conversion
fs=sig.ft(s, ax=0);

# Display aliased SAR spectrum
#
G=abs(fs);
xg=np.max(np.max(G)); ng=np.min(np.min(G)); cg=255/(xg-ng);
plt.figure()
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = ((k*c/pi2).min(), (k*c/pi2).max(), kuc.min(), kuc.max()), aspect = 'auto');
plt.xlabel('Fast-time Frequency, Hertz')
plt.ylabel('Synthetic Aperture (Slow-time) Frequency Ku, rad/m')
plt.title('Aliased Spotlight SAR Signal Spectrum')
#

#################################################################
##  Digital Spotlighting and Bandwidth Expansion in ku Domain  ##
##          via Slow-time Compression and Decompression        ##
#################################################################
#
s=s*exp(cj*kus*uc);      # Original signal before baseband
                             # conversion for squint
cs=s*exp(cj*2*k*\
 (ones([mc,n])*sqrt(Xc**2+(Yc-uc)**2))-cj*2*k*Rc);# compression
fcs=sig.ft(cs,ax=0);            # F.T. of compressed signal w.r.t. u
#
G=abs(fcs);
xg=np.max(np.max(G)); ng=np.min(np.min(G)); cg=255/(xg-ng);
plt.figure()
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = ((k*c/pi2).min(), (k*c/pi2).max(), kuc.min(), kuc.max()), aspect = 'auto');
plt.xlabel('Fast-time Frequency, Hertz')
plt.ylabel('Synthetic Aperture (Slow-time) Frequency Ku, rad/m')
plt.title('Compressed Spotlight SAR Signal Spectrum')
#
fp=sig.ift(sig.ft(cs, ax=0));      # Narrow-bandwidth Polar Format Processed
                       # reconstruction
#
PH=arcsin(kuc/(2*kc));   # angular Doppler domain
R=(c*tm)/2;            # range domain mapped from reference
                       # fast-time domain
#
# Full Aperture Digital-Spotlight Filter
#
W_d=((np.abs(R*cos(PH+theta_c)-Xc) < X0)*\
    (np.abs(R*sin(PH+theta_c)-Yc) < Y0));
#
G=(abs(fp)/abs(fp).max()+.1*W_d);
xg=G.max(); ng=G.min(); cg=255/(xg-ng);
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = (((Rc/Xc)*(.5*c*tm-Rc)).min(), ((Rc/Xc)*(.5*c*tm-Rc)).max(), ((kuc*Rc)/(2*kc)).min(), ((kuc*Rc)/(2*kc)).max()), aspect = 'auto');
plt.xlabel('Range x, m')
plt.ylabel('Cross-range y, m')
plt.title('Polar Format SAR Reconstruction with Digital Spotlight Filter')

fd=fp*W_d;                # Digital Spotlight Filtering
fcs=sig.ft(fd);               # Transform to (omega,ku) domain

# Zero-padding in ku domain for slow-time upsampling
#
mz=m-mc;        # number is zeros
fcs=(m/mc)*np.vstack((zeros([mz/2,n]),fcs,zeros([mz/2,n])));
#
cs=sig.ift(fcs, ax=0);              # Transform to (omega,u) domain

s = np.zeros(cs.shape)+0j
s=cs*exp(-cj*2*(k)*\
 (sqrt(Xc**2+(Yc-u)**2))+cj*2*k*Rc);# decompression


#################################################################
#                           CAUTION                             #
# For TDC or backprojection, do not subsample in Doppler domain #
# and do not perform slow-time baseband conversion               #
#################################################################
#
s_ds=1.0*s;                    # Save s(omega,u) array for TDC and
                           # backprojection algorithms

#
s=s*exp(-cj*kus*u);    # Slow-time baseband conversion for squint
fs=sig.ft(s, ax=0);                 # Digitally-spotlighted SAR signal spectrum
#
G=abs(fs);
xg=G.max(); ng=G.min(); cg=255/(xg-ng);
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = ((k*c/pi2).min(), (k*c/pi2).max(), ku.min(), ku.max()), aspect = 'auto');
plt.xlabel('Fast-time Frequency, Hertz')
plt.ylabel('Synthetic Aperture (Slow-time) Frequency Ku, rad/m')
plt.title('Spotlight SAR Signal Spectrum after DS & Upsampling')


##########################################
##    SLOW-TIME DOPPLER SUBSAMPLING     ##
##########################################
#
if Y0 < L:
 ny=2*ceil(1.2*Y0/du);      # Number of samples in y domain
                            # 20 percent guard band
 ms=floor(1.0*m/ny);            # subsampling ratio
 tt=floor(1.0*m/(2*ms));
 I=np.arange(m/2-tt*ms,m/2+1+(tt-1)*ms,ms); # subsampled index in ku domain
 I = np.array(I,dtype=int)
 tt = 1
 ny=int(I.size);           # number of subsamples
 fs=fs[I,:];                # subsampled SAR signal spectrum
 ky=ku[I];                  # subsampled ky array
 dky=dku*ms;                # ky domain sample spacing
else:
 dky=dku;
 ny=m;
 ky=ku;
#end;

dy=pi2/(ny*dky);            # y domain sample spacing
y=dy*np.array([arange(-ny/2,ny/2)]).T;        # cross-range array

##########################################
##             RECONSTRUCTION           ##
##########################################
#
ky=np.tile(ky+kus,(1,n));       # ky array
kx=np.tile((4*k**2),[ny,1])-ky**2;
kx=sqrt(kx*(kx > 0));                  # kx array
#
plt.figure()
plt.scatter(kx.flatten(), ky.flatten())
plt.xlabel('Spatial Frequency k_x, rad/m')
plt.ylabel('Spatial Frequency k_y, rad/m')
plt.title('Spotlight SAR Spatial Frequency Data Coverage')
#
kxmin=kx.min();
kxmax=kx.max();
dkx=pi/X0;        # Nyquist sample spacing in kx domain
nx=2*ceil((.5*(kxmax-kxmin))/dkx); # Required number of
                      # samples in kx domain;
                      # This value will be increased slightly
                      # to avoid negative array index
#
###############################################################
###                                                         ###
###   FIRST TWO OPTIONS FOR RECONSTRUCTION:                 ###
###                                                         ###
###     1. 2D Fourier Matched Filtering and Interpolation   ###
###     2. Range Stacking                                   ###
###                                                         ###
###     Note: For "Range Stacking," make sure that the      ###
###           arrays nx, x, and kx are defined.             ###
###                                                         ###
###############################################################


############################################################
###    2D FOURIER MATCHED FILTERING AND INTERPOLATION    ###
############################################################

# Matched Filtering
#
fs0=(kx > 0)*exp(cj*kx*Xc+cj*ky*Yc+cj*.25*pi\
           -cj*2*k*Rc); # reference signal complex conjugate
fsm=fs*fs0;     # 2D Matched filtering

# Interpolation
#
iis=8;       # number of neighbors (sidelobes) used for sinc interpolator
I=2*iis;
kxs=iis*dkx; # plus/minus size of interpolation neighborhood in KX domain
#
nx=nx+2*iis+4;  # increase number of samples to avoid negative
               #  array index during interpolation in kx domain
KX=kxmin+arange(-iis-2,nx-iis-2)*dkx;     # uniformly-spaced kx points where
                                  # interpolation is done
kxc=KX[nx/2];                   # carrier frequency in kx domain
KX=np.tile(KX, (ny,1));
#
F=zeros([ny,nx])+0j;         # initialize F(kx,ky) array for interpolation

for i in range(int(ny)):                       # for each k loop
 print(i)                              # print i to show that it is running
 F[i]=interp1d(kx[i],fsm[i], kind=3, bounds_error=False, fill_value=0)(KX[i]);
#end
#
#  DISPLAY interpolated spatial frequency domain image F(kx,ky)

KX=KX[0,:]#.';
KY=ky[:,0];

G=abs(F)#';
xg=G.max(); ng=G.min(); cg=255/(xg-ng);
plt.figure()
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = (KX.min(), KX.max(), (KY+kus).min(), (KY+kus).max()), aspect = 'equal');
plt.xlabel('Spatial Frequency k_x, rad/m')
plt.ylabel('Spatial Frequency k_y, rad/m')
plt.title('Wavefront Spotlight SAR Reconstruction Spectrum')

#
f=sig.ift(sig.ift(F, ax = 0));     # Inverse 2D FFT for spatial domain image f(x,y)
#
dx=pi2/(nx*dkx);     # range sample spacing in reconstructed image
x=dx*arange(-nx/2,nx/2); # range array
#
# Display SAR reconstructed image

G=abs(f)#';
xg=G.max(); ng=G.min(); cg=255/(xg-ng);
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = ((Xc+x).min(), (Xc+x).max(), (Yc+y).min(), (Yc+y).max()), aspect = 'equal');
plt.xlabel('Range X, meters')
plt.ylabel('Cross-range Y, meters')
plt.title('Wavefront Spotlight SAR Reconstruction')


#####################################################
##   SAR Image Compression (for Spotlight System)  ##
#####################################################
##
Fc=sig.ft(sig.ft(f*\
  exp(cj*kxc*x+cj*2*kc*sin(theta_c)*y\
 -cj*2*kc*sqrt(((Xc+x)**2)+((Yc+y)**2))), ax=0));
G=abs(Fc)#';
xg=G.max(); ng=G.min(); cg=255/(xg-ng)
plt.figure()
plt.imshow(256-cg*(G-ng)[::-1,:],
           extent = (KX.min(), KX.max(), (KY+kus).min(), (KY+kus).max()), aspect = 'equal');
plt.xlabel('Spatial Frequency k_x, rad/m')
plt.ylabel('Spatial Frequency k_y, rad/m')
plt.title('Compressed Spotlight SAR Reconstruction Spectrum')
