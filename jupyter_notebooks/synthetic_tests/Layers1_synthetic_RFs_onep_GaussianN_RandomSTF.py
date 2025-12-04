#test for T13
import numpy as np
from LayerModel import rfmini
from obspy.io.sac.sactrace import SACTrace
import os
import numpy as np
from scipy import signal

rng = np.random.default_rng()

def Add_Gaussian_Noise(Signal,snr):
    """
    add noise that has a normal distribution with a snr

    Created by Wentao Li at 2024/06/17

    """
    As = np.mean(Signal**2)
    std = np.sqrt(As/np.power(10,snr/10))
    noise = np.random.normal(0,std,Signal.shape)
    return Signal+noise


def random_source_time_fun(Lst, cornerfreq, delta):

    Nst = int(Lst//delta)
    signals_random = rng.normal(0,1,size=Nst)
    b, a = signal.butter(3, cornerfreq, btype='lowpass', fs=1.0/delta)
    signals_random = signal.filtfilt(b, a, signals_random)
    ##hanning windows
    win = np.hanning(Nst)
    return signals_random*win
dt=0.01


#parameters for source time function
freqmin = 0.1
freqmax = 0.5
NSt_min = 3 ##seconds  M0 (Nm) 18 Vallee 2013 --> mw = 5.5 (Geller, 1976)
NSt_max = 100 ##seconds Mw 8 

pretime = 5
Nt=20000
Np = 120


nsv=0.
km2deg = 1/180*np.pi*6378 


nsamp = 2.**int(np.ceil(np.log2(Nt * 2)))
gauss = 8

header = {'kstnm': 'TEST', 'stla': 0.0, 'stlo': 0.,
          'evla': 0.0, 'evlo': 0.0, 'evdp': 50, 'nzyear': 2022,
          'nzjday': 57, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
          'nzmsec': 100, 'delta': dt}

####design test model#
thickness_c = 35
SynSavePath = f"Varyp_sediment_crust_mantle_synthetics_c{thickness_c}_dt{dt}"
if not os.path.isdir(SynSavePath):
    os.makedirs(SynSavePath)
##test model1



##test model2

Vss =    np.array([   3.66, 4.5])
Vps =    np.array([    6.7, 8.0])
Rhos =   np.array([   2.7, 3.3])


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
SynSavePath1 = f"{SynSavePath}/crust{thickness_c}_Np{Np}_oneevent"
if not os.path.isdir(SynSavePath1):
    os.makedirs(SynSavePath1)

ray_param0 = ray_param*km2deg
filename="{0}/ray{1:6.5f}.SAC".format(SynSavePath1,ray_param)
zz,rr,rf,zrf = rfmini.synrf(z,Vp,Vs,Rhos,qa,qb,ray_param0,gauss,nsamp,1/dt,pretime,nsv,poisson,'P')
sactrace = SACTrace(data=rr[:Nt],user0=ray_param,kcmpnm='R',**header)
sactrace.write(filename.replace(".SAC",".R.SAC"))
sactrace = SACTrace(data=zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
sactrace.write(filename.replace(".SAC",".Z.SAC"))
##convolve with a random source time function
source_fun = random_source_time_fun(12, 0.4 , dt)
filename1="{0}/ray{1:6.5f}.STF.npz".format(SynSavePath1,ray_param)
np.savez(filename1,x=np.arange(len(source_fun))*dt,y=source_fun)
npts_shift = int(pretime/dt)
srr = signal.convolve(rr,source_fun,mode='same')
srr = np.roll(srr,npts_shift)
szz = signal.convolve(zz,source_fun,mode='same')
szz = np.roll(szz,npts_shift)
sactrace = SACTrace(data=srr[:Nt],user0=ray_param,kcmpnm='R',**header)
sactrace.write(filename.replace(".SAC",".Rs.SAC"))
sactrace = SACTrace(data=szz[:Nt],user0=ray_param,kcmpnm='R',**header)
sactrace.write(filename.replace(".SAC",".Zs.SAC"))
#add gaussian noise and then save 
##zz and rr
snr = 6
noisy_rr = Add_Gaussian_Noise(rr,snr)
sactrace = SACTrace(data=noisy_rr[:Nt],user0=ray_param,kcmpnm='R',**header)
sactrace.write(filename.replace(".SAC",".Rn.SAC"))
noisy_zz = Add_Gaussian_Noise(zz,snr)
sactrace = SACTrace(data=noisy_zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
sactrace.write(filename.replace(".SAC",".Zn.SAC"))

##
noisy_srr = Add_Gaussian_Noise(srr,snr)
sactrace = SACTrace(data=noisy_srr[:Nt],user0=ray_param,kcmpnm='R',**header)
sactrace.write(filename.replace(".SAC",".Rsn.SAC"))
noisy_szz = Add_Gaussian_Noise(szz,snr)
sactrace = SACTrace(data=noisy_szz[:Nt],user0=ray_param,kcmpnm='Z',**header)
sactrace.write(filename.replace(".SAC",".Zsn.SAC"))

