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

rng = np.random.default_rng()

def random_source_time_fun(Nstmin, Nstmax, freqmin, freqmax, delta):
    Nstmin_log = np.log10(Nstmin)
    Nstmax_log = np.log10(Nstmax)
    Nst_log = rng.uniform(Nstmin_log,Nstmax_log,size=1)[0]
    Nst = int(np.power(10,Nst_log)//delta)
    signals_random = rng.normal(0,1,size=Nst)
    corner_freq = rng.uniform(freqmin,freqmax,size=1)[0]
    b, a = signal.butter(3, corner_freq, btype='lowpass', fs=1.0/delta)
    signals_random = signal.filtfilt(b, a, signals_random)
    ##hanning windows
    win = np.hanning(Nst)
    return signals_random*win

def Vp2rho(vp):
    """
    polynomial regression according to Ludwig et al. (1970)

    also called Nafe-Drake curve 

    warning: only for unit of km/s
    """
    rho=1.6612*vp-0.4721*vp**2+0.0671*vp**3-0.0043*vp**4+0.000106*vp**5
    return rho

def designMOhotransition(Hmt, Hlv, Hhv, Vp1,Vp2,Vs1,Vs2,Vpl,Vsl, nlayer):
    dh_gradient = Hmt/nlayer
    thicks_gradient = np.ones([nlayer])*dh_gradient
    depth_gradient = np.cumsum(thicks_gradient)
    Vpcl_gradient = Vp1+(Vp2-Vp1)/Hmt*depth_gradient
    Vscl_gradient  =  Vs1+(Vs2-Vs1)/Hmt*depth_gradient
    nlayer_lv = int(Hlv/dh_gradient)
    nlayer_hv = int(Hhv/dh_gradient)
    nalter = int(nlayer/(nlayer_hv+nlayer_lv))
    for i in range(nalter):
        Vpcl_gradient[i*(nlayer_hv+nlayer_lv)+nlayer_hv:(i+1)*(nlayer_hv+nlayer_lv)] = Vpl
        Vscl_gradient[i*(nlayer_hv+nlayer_lv)+nlayer_hv:(i+1)*(nlayer_hv+nlayer_lv)] = Vsl
    return Vpcl_gradient,Vscl_gradient,thicks_gradient




dt=0.01
#parameters for source time function
freqmin = 0.1
freqmax = 1
NSt_min = 3 ##seconds  M0 (Nm) 18 Vallee 2013 --> mw = 5.5 (Geller, 1976)
NSt_max = 100 ##seconds Mw 8 


thickness_ss = [4]
params = [(7.5,0),(7.5,2)]
for param in params:
    thickness_cl, thickness_lowVel = param 
    for thickness_s in thickness_ss:
        pretime = 5
        Nt=20000
        Np = 100
        nsv=0.
        km2deg = 1/180*np.pi*6378 
        nsamp = 2.**int(np.ceil(np.log2(Nt * 2)))
        gauss = 8
        header = {'kstnm': 'TEST', 'stla': 0.0, 'stlo': 0.,
                'evla': 0.0, 'evlo': 0.0, 'evdp': 50, 'nzyear': 2022,
                'nzjday': 57, 'nzhour': 13, 'nzmin': 43, 'nzsec': 17,
                'nzmsec': 100, 'delta': dt}

        Kc = 1.78
        Kcl = 1.78
        Vpc = 6.37
        Vpm = 8.0
        Km = 1.75
        # thickness_cl = 0
        # thickness_lowVel = 0
        thickness_highVel = 2
        Thickness_crust = 35
        ks = 2.4 
        vps = 3.6
        # thickness_s = 4
        SynSavePath = f"synthetic_data_S{freqmax}/Varyp_crust_mantle_gradMoho_synthetics_c{Thickness_crust}_Vpc{Vpc}_kc{Kc}_kcl{Kcl}_Tcl{thickness_cl}_TlV{thickness_lowVel}_THV{thickness_highVel}_hs{thickness_s}_Vps{vps}_Ks{ks}_dt{dt}"
        if not os.path.isdir(SynSavePath):
            os.makedirs(SynSavePath)
        if thickness_cl!=0:
            Vpcl,Vscl,thicks_cl = designMOhotransition(thickness_cl,thickness_lowVel,thickness_highVel,Vpc,Vpm,Vpc/Kc,Vpm/Km,Vpc,Vpc/Kc,20)
            Vss =  np.concatenate([np.array([vps/ks,Vpc/Kc]),Vscl,np.array([Vpm])/Km]) 
            Vps =  np.concatenate([np.array([vps,Vpc]),Vpcl,np.array([Vpm])]) 
            Rhos =   Vp2rho(Vps)
            Thicks = np.concatenate([np.array([thickness_s,Thickness_crust-thickness_cl]),thicks_cl,np.array([0])]) 
        else:
            Vss =  np.concatenate([np.array([vps/ks,Vpc/Kc]),np.array([Vpm])/Km]) 
            Vps =  np.concatenate([np.array([vps,Vpc]),np.array([Vpm])]) 
            Rhos =   Vp2rho(Vps)
            Thicks = np.concatenate([np.array([thickness_s,Thickness_crust-thickness_cl]),np.array([0])]) 
        print(Vps)
        print(Vss)
        print(Thicks)
        Vp = Vps.astype('float')
        Vs = Vss.astype('float')
        Rhos = Rhos.astype('float')
        qa  = np.ones(Vp.shape)*50000000000
        qb =  np.ones(Vp.shape)*25000000000
        nsvp, nsvs = float(Vp[0]), float(Vs[0])
        vpvs = nsvp / nsvs
        poisson = (2 - vpvs**2)/(2 - 2 * vpvs**2)


        Thick=Thicks.astype('float')
        z = np.cumsum(Thick)
        z = np.concatenate(([0], z[:-1]))
        SynSavePath1 = f"{SynSavePath}/sediment_depth{thickness_s}"
        if not os.path.isdir(SynSavePath1):
            os.makedirs(SynSavePath1)
        ray_params = np.random.uniform(0.04,0.08,Np)    
        ##savemodel 
        np.savez(f"{SynSavePath1}/mod.npz",Vs = Vs, Vp = Vp, Rho = Rhos, Thicks=Thick)
        for ray_param in ray_params:
            ray_param0 = ray_param*km2deg
            filename="{0}/ray{1:6.5f}.SAC".format(SynSavePath1,ray_param)
            zz,rr,rf,zrf = rfmini.synrf(z,Vp,Vs,Rhos,qa,qb,ray_param0,gauss,nsamp,1/dt,pretime,nsv,poisson,'P')
            sactrace = SACTrace(data=rr[:Nt],user0=ray_param,kcmpnm='R',**header)
            sactrace.write(filename.replace(".SAC",".R.SAC"))
            sactrace = SACTrace(data=zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
            sactrace.write(filename.replace(".SAC",".Z.SAC"))
            ##convolve with a random source time function
            source_fun = random_source_time_fun(NSt_min, NSt_max, freqmin, freqmax, dt)
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

# SynSavePath1 = f"{SynSavePath}/sediment_depth0_linerayp"
# if not os.path.isdir(SynSavePath1):
#     os.makedirs(SynSavePath1)
# ray_params = np.linspace(0.04,0.08,20)    
# ##savemodel 
# np.savez(f"{SynSavePath1}/mod.npz",Vs = Vs, Vp = Vp, Rho = Rhos, Thicks=Thick)
# for ray_param in ray_params:
#     ray_param0 = ray_param*km2deg
#     filename="{0}/ray{1:6.5f}.SAC".format(SynSavePath1,ray_param)
#     zz,rr,rf,zrf = rfmini.synrf(z,Vp,Vs,Rhos,qa,qb,ray_param0,gauss,nsamp,1/dt,pretime,nsv,poisson,'P')
#     sactrace = SACTrace(data=rr[:Nt],user0=ray_param,kcmpnm='R',**header)
#     sactrace.write(filename.replace(".SAC",".R.SAC"))
#     sactrace = SACTrace(data=zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
#     sactrace.write(filename.replace(".SAC",".Z.SAC"))
#     ##convolve with a random source time function
#     source_fun = random_source_time_fun(NSt_min, NSt_max, freqmin, freqmax, dt)
#     filename1="{0}/ray{1:6.5f}.STF.npz".format(SynSavePath1,ray_param)
#     np.savez(filename1,x=np.arange(len(source_fun))*dt,y=source_fun)
#     npts_shift = int(pretime/dt)
#     srr = signal.convolve(rr,source_fun,mode='same')
#     srr = np.roll(srr,npts_shift)
#     szz = signal.convolve(zz,source_fun,mode='same')
#     szz = np.roll(szz,npts_shift)
#     sactrace = SACTrace(data=srr[:Nt],user0=ray_param,kcmpnm='R',**header)
#     sactrace.write(filename.replace(".SAC",".Rs.SAC"))
#     sactrace = SACTrace(data=szz[:Nt],user0=ray_param,kcmpnm='R',**header)
#     sactrace.write(filename.replace(".SAC",".Zs.SAC"))
#     #add gaussian noise and then save 
#     ##zz and rr
#     snr = np.random.uniform(low=6,high=15)
#     noisy_rr = Add_Gaussian_Noise(rr,snr)
#     sactrace = SACTrace(data=noisy_rr[:Nt],user0=ray_param,kcmpnm='R',**header)
#     sactrace.write(filename.replace(".SAC",".Rn.SAC"))
#     noisy_zz = Add_Gaussian_Noise(zz,snr)
#     sactrace = SACTrace(data=noisy_zz[:Nt],user0=ray_param,kcmpnm='Z',**header)
#     sactrace.write(filename.replace(".SAC",".Zn.SAC"))

#     ##
#     noisy_srr = Add_Gaussian_Noise(srr,snr)
#     sactrace = SACTrace(data=noisy_srr[:Nt],user0=ray_param,kcmpnm='R',**header)
#     sactrace.write(filename.replace(".SAC",".Rsn.SAC"))
#     noisy_szz = Add_Gaussian_Noise(szz,snr)
#     sactrace = SACTrace(data=noisy_szz[:Nt],user0=ray_param,kcmpnm='Z',**header)
#     sactrace.write(filename.replace(".SAC",".Zsn.SAC"))
