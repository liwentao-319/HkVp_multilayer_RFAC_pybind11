#test for T13
import numpy as np
from LayerModel import rfmini
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import hilbert
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)
import warnings
from scipy.signal import sosfilt
from scipy.signal import zpk2sos


def lowpass(data, freq, df, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)



def zero_phase_bandpass(data,fs,low_cutoff,high_cutoff ,order=5):
    b,a = signal.butter(order, [low_cutoff / (fs / 2), high_cutoff / (fs / 2)], btype='band')
    return signal.filtfilt(b,a,data)


def cos_taper(data,M):
    cos_win = 0.5-0.5*np.cos(np.pi*np.arange(M)/(M-1))
    data[:M] = data[:M]*cos_win
    return data


dt=0.01
#parameters for source time function
freqmin = 0.1
freqmax = 0.5
NSt_min = 3 ##seconds  M0 (Nm) 18 Vallee 2013 --> mw = 5.5 (Geller, 1976)
NSt_max = 100 ##seconds Mw 8 

pretime = 5
Nt=20000

Np = 100

nsv=0.
km2deg = 1/180*np.pi*6378 


nsamp = 2.**int(np.ceil(np.log2(Nt * 2)))
gauss = 8


####design test model#

##test model2

Vss =    np.array([   3.66, 4.5])
Vps =    np.array([    6.7, 8.0])
Rhos =   np.array([   2.7, 3.3])
thickness_c = 35

Vp = Vps.astype('float')
Vs = Vss.astype('float')
Rhos = Rhos.astype('float')
qa  = np.ones(Vp.shape)*50000000000
qb =  np.ones(Vp.shape)*25000000000
nsvp, nsvs = float(Vp[0]), float(Vs[0])
vpvs = nsvp / nsvs
poisson = (2 - vpvs**2)/(2 - 2 * vpvs**2)
ray_param = 0.06

Thicks = np.array([thickness_c, 0])
Thick=Thicks.astype('float')
z = np.cumsum(Thick)
z = np.concatenate(([0], z[:-1]))



ray_param0 = ray_param*km2deg
zz,rr,rf,zrf = rfmini.synrf(z,Vp,Vs,Rhos,qa,qb,ray_param0,gauss,nsamp,1/dt,pretime,nsv,poisson,'P')


rf = rf[:Nt]
rr = rr[:Nt]
zz = zz[:Nt]
rr = lowpass(rr,2,1/dt,zerophase=True)
zz = lowpass(zz,2,1/dt,zerophase=True)

rr_auto = np.correlate(rr,rr,mode='full')[Nt:]
zz_auto = np.correlate(zz,zz,mode='full')[Nt:]

tcos = 3
MM = int(tcos//dt)
print(MM)


rf = rf/rf.max()
rr = rr/rr.max()
zz = zz/zz.max()

rr_auto = rr_auto/(-rr_auto.min())
zz_auto = zz_auto/(-zz_auto.min())


pltlength = 25 
prenpts = int(5/dt)
lengthnpts = int(pltlength/dt) 
rf = rf[prenpts:prenpts+lengthnpts]
rr = rr[prenpts:prenpts+lengthnpts]
zz = zz[prenpts:prenpts+lengthnpts]
rr_auto = rr_auto[:lengthnpts]
zz_auto = zz_auto[:lengthnpts]

times = np.arange(lengthnpts)*dt

rf = cos_taper(rf,MM)
rr = cos_taper(rr,MM)
zz = cos_taper(zz,MM)
rr_auto = cos_taper(rr_auto,MM)
zz_auto = cos_taper(zz_auto,MM)


cm = 1/2.54
fig,ax = plt.subplots(1,1, figsize=(18*cm, 8*cm))

ax.plot(times,zz*2+4, lw=1, c='r')
ax.plot(times,rr*2+3, lw=1, c='b')

ax.plot(times,rf*2+2, lw=1, c='b')
ax.plot(times,zz_auto+1, lw=1, c='r')
ax.plot(times,rr_auto+0, lw=1, c='b')


ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(['R auto',"Z auto","RF","R", "Z"])
ax.set_ylim(-1.5,5.5)
ax.set_xlabel("Time [s]")


plt.savefig("fig1_illustration.pdf",dpi=900)
plt.show()




