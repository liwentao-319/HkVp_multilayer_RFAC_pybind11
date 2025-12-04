from acc.core import spectral_whitening,remove_response
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from scipy.signal import correlate
from scipy.fftpack import fft,ifft
from obspy import read
from scipy import signal 
from AC import Ac
import os

maindir = "seismic_data_RF"
stationname="2F_TB012"
datadir = f"{maindir}/{stationname}"


sampling_rate = 100
length=150
f1 = 0.1
f2 = 1
length = 20
dws = [ 0.06, 0.07, 0.08, 0.09, 0.1,None]
channel = "HHZ"

savedir=f"test/test_dw_moho_f1{f1}_f2{f2}/{stationname}"
if not os.path.isdir(savedir):
    os.makedirs(savedir)

for i,dw in enumerate(dws):
    #plot spectral power
    do_Ac = Ac(datadir=datadir,suffix=f".{channel}.sac",tref=5.,pretime=-5,length=150,f1=f1,f2=f2,tcos=5,df=dw)
    
    do_Ac.cal_ACs()
    do_Ac.stack_ACs(0.04,0.08,0.003,0.003)
    stream_zacm_stack = do_Ac.ACst_stack
    stream_zacm = do_Ac.ACst
    fig = plt.figure(figsize=(8,10),tight_layout=True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for tr in stream_zacm:
        ray_para = tr.stats.sac['user0']
        delta = tr.stats.delta
        starttime = tr.stats.starttime
        tr.trim(starttime,starttime+length)
        data = tr.data/(-1*tr.data.min())*0.001
        times = np.arange(len(data))*delta
        ax1.plot(data+ray_para, times, lw=0.5, color='k')
        data_fill = data.copy()
        data_fill[data_fill>0] = 0
        ax1.fill_betweenx(times, ray_para, ray_para+data_fill, color='red', lw=0.01)
        data_fill = data.copy()
        data_fill[data_fill<0] = 0
        ax1.fill_betweenx(times, ray_para, ray_para+data_fill, color='blue', lw=0.01)


    for tr in stream_zacm_stack:
        ray_para = tr.stats.sac['user0']
        delta = tr.stats.delta
        starttime = tr.stats.starttime
        tr.trim(starttime,starttime+length)
        data = tr.data/(-1*tr.data.min())*0.001
        times = np.arange(len(data))*delta
        ax2.plot(data+ray_para, times, lw=0.5, color='k')
        data_fill = data.copy()
        data_fill[data_fill>0] = 0
        ax2.fill_betweenx(times, ray_para, ray_para+data_fill, color='red', lw=0.01)
        data_fill = data.copy()
        data_fill[data_fill<0] = 0
        ax2.fill_betweenx(times, ray_para, ray_para+data_fill, color='blue', lw=0.01)
    
    ax1.invert_yaxis()
    ax1.set_xlabel("ray_param [s/km]")
    ax1.set_ylabel("Time [s]")

    ax2.invert_yaxis()
    ax2.set_xlabel("ray_param [s/km]")
    ax2.set_ylabel("Time [s]")

    plt.savefig(f"{savedir}/dw{dw}_stack_{channel}.png",dpi=900)
    plt.close("all")






