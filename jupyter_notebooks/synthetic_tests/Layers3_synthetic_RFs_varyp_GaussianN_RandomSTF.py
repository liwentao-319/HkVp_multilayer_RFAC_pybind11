#test for T13
import numpy as np
from LayerModel import rfmini
from obspy.io.sac.sactrace import SACTrace
import os
import numpy as np
from scipy import signal



def Add_Gaussian_Noise(Signal,snr):
    """
    add noise that has a normal distribution with a snr

    Created by Wentao Li at 2024/06/17

    """
    As = np.mean(Signal**2)
    std = np.sqrt(As/np.power(10,snr/10))
    noise = np.random.normal(0,std,Signal.shape)
    return Signal+noise


def random_source_time_fun(Nmin, Nmax, freqmin, freqmax, delta):
    rng = np.random.default_rng()
    size = rng.integers(Nmin,Nmax,size=1)
    signals_random = rng.normal(0,1,size=size)
    corner_freq = rng.uniform(freqmin,freqmax,size=1)
    b, a = signal.butter(3, corner_freq, btype='lowpass', fs=1.0/delta)
    signals_random = signal.filtfilt(b, a, signals_random)
    ##hanning windows
    win = np.hanning(size)
    return signals_random*win

def Vp2rho(vp):
    """
    polynomial regression according to Ludwig et al. (1970)

    also called Nafe-Drake curve 

    warning: only for unit of m/s
    """
    rho=1.6612*vp-0.4721*vp**2+0.0671*vp**3-0.0043*vp**4+0.000106*vp**5
    return rho




dt=0.01

#parameters for source time function
freqmin = 0.5
freqmax = 1.0
Nt_min = 3 ##seconds  M0 (Nm) 18 Vallee 2013 --> mw = 5.5 (Geller, 1976)
Nt_max = 100 ##seconds Mw 8 

pretime = 5
Nt=20000

Np = 100

Nmin = int(Nt_min/dt)
Nmax = int(Nt_max/dt)
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
thickness_s = 5
kss = [1.8]
for ks in kss:
    SynSavePath = f"synthetic_data_S{freqmax}/Varyp_2sediment_crust_mantle_synthetics_c{thickness_c}_d1s{thickness_s}_k2s{ks}_dt{dt}_G2"
    if not os.path.isdir(SynSavePath):
        os.makedirs(SynSavePath)
    ##test model1
    # depth_ss = np.array([1,2,3,4,5,6,7,8,9,10])
    # Vss =    np.array([     1.8,    3.66, 4.5])
    # Vps =    np.array([    3.6,    6.7, 8.0])
    # Rhos =   np.array([    2.27,    2.7, 3.3])
    ##test model2
    depth_ss = np.array([5])
    Vss =    np.array([ 3.0/2.5,   4.5/ks,    6.7/1.75, 4.5])
    Vps =    np.array([ 3.0,  4.5,    6.7,  8.0]) ##k 2.3, 1.9, 1.75
    Rhos =   Vp2rho(Vps)


    Vp = Vps.astype('float')
    Vs = Vss.astype('float')
    Rhos = Rhos.astype('float')
    qa  = np.ones(Vp.shape)*50000000000
    qb =  np.ones(Vp.shape)*25000000000
    nsvp, nsvs = float(Vp[0]), float(Vs[0])
    vpvs = nsvp / nsvs
    poisson = (2 - vpvs**2)/(2 - 2 * vpvs**2)
    ray_params = np.random.uniform(0.04,0.08,150)

    for depth_s in depth_ss:
        Thicks = np.array([thickness_s,depth_s, thickness_c, 0])
        Thick=Thicks.astype('float')
        z = np.cumsum(Thick)
        z = np.concatenate(([0], z[:-1]))
        SynSavePath1 = f"{SynSavePath}/sediments_depth{depth_s}"
        if not os.path.isdir(SynSavePath1):
            os.makedirs(SynSavePath1)
        for ray_param in ray_params:
            ray_param0 = ray_param*km2deg
            filename="{0}/ray{1:6.5f}.SAC".format(SynSavePath1,ray_param)
            zz,rr,rf,zrf = rfmini.synrf(z,Vp,Vs,Rhos,qa,qb,ray_param0,gauss,nsamp,1/dt,pretime,nsv,poisson,'P')
            sactrace = SACTrace(data=rr[:Nt],user0=ray_param,kcmpnm='R',**header)
            sactrace.write(filename.replace(".SAC",".R.SAC"))
            sactrace = SACTrace(data=zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
            sactrace.write(filename.replace(".SAC",".Z.SAC"))
            ##convolve with a random source time function
            source_fun = random_source_time_fun(Nmin, Nmax, freqmin, freqmax, dt)
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
            snr = np.random.uniform(low=6,high=15)
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

