from random import gauss
import numpy as np
from glob import glob 
import os
from obspy import read
from Hk_stacking_multilayer_vp import Hk_searching_RFAC_multilayer
from AC import Ac
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
list_colornames = ['orange','gold','cyan','skyblue','blueviolet','violet','pink']
plt.rcParams.update({"font.size":5})

def plot_RF_rayp(ax,Ss,rays,sampling,scale,t0=None,t1=None,):
    delta = 1.0/sampling
    if rays.shape[0] != Ss.shape[1]:
        return -1
    for i,rayp in enumerate(rays):
        data = Ss[:,i]
        data = data*scale
        times = np.arange(len(data))*delta
        if t0!=None and t1!=None:
            index = np.logical_and(times>t0, times<t1)
            times = times[index]
            data = data[index]
        ax.plot(data+rayp,times,lw=0.03,c='k')
        data_fill = data.copy()
        data_fill[data_fill>0]=0
        ax.fill_betweenx(times,data_fill+rayp,rayp,color='red',lw=0.001)
        data_fill = data.copy()
        data_fill[data_fill<0]=0
        ax.fill_betweenx(times,data_fill+rayp,rayp,color='blue',lw=0.001)
    ax.grid(axis='y',lw=0.5,ls='--',color='gray',alpha=0.5)
    ax.set_xlabel('ray_param [s/km]')
    ax.set_ylabel('Time [s]')


maindir = "seismic_data_RFAC"
only_plot = False
stationname = "ZQ_161"

datadir=f"{maindir}/{stationname}"

if not only_plot:
    ##read RF data
    stream_rf2 = read(f"{datadir}/*.BHR.sac_g2")

    streams = [stream_rf2]

    dataparam ={
        "delta":0.01,
        "pretimes":[5,],
        "lengths":[100],
        "norm_type":[1],
        "plotlengths":[ 25]
    }

    priorparam = {
        "Nlayer":1,
        "Vps":[6.2],
        "Hs":[[20,40,101]],
        "Ks":[[1.5,2.0,101]]
    }

    phaseparam = [

        [
            [0.5, 0, 1, -1,[]], #Pms
            [0.4, 0, 1,  1,[]], #PmPpms
            [-0.1, 0, 2,  0,[]], #PmPpms
        ],

    ]

    phasestackparams=None

    Hk_RFAC_2layer = Hk_searching_RFAC_multilayer(stationname=stationname,
                                                data_st=streams,
                                                dataparam=dataparam,
                                                priorparam=priorparam,
                                                phaseparam=phaseparam,
                                                phasestackparams=phasestackparams
                                            )

    Hk_RFAC_2layer.do_searching()
    Hk_RFAC_2layer.hk_bootstrap_estimate(1000,8)
    # Hk_RFAC_2layer.save_result_to_figs(savedir=savedir)
    savedir = f"Hk_RF1layer/{stationname}"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    Hk_RFAC_2layer.save_result_to_files(savedir=savedir)
    with open(f"{savedir}/Hk_results.bin", 'wb') as fd:
        pickle.dump(Hk_RFAC_2layer,fd)
##plot figures

savedir = f"Hk_RF1layer/{stationname}"
with open(f"{savedir}/Hk_results.bin",'rb') as fd:
    Hk_RFAC_2layer = pickle.load(fd)
Nlayer = Hk_RFAC_2layer.Nlayer
Ndataset = Hk_RFAC_2layer.Ndataset
cm=1/2.54
fig,ax = plt.subplots(Nlayer,1,figsize=(6*cm,5*Nlayer*cm),layout='constrained')
plt.rcParams.update({"font.size":5})

stationname = Hk_RFAC_2layer.stationname
sampling = 1/Hk_RFAC_2layer.delta


amp = Hk_RFAC_2layer.Hk_amps[0]
Hmin,Hmax,nH = Hk_RFAC_2layer.Hs_range[0]
Kmin,Kmax,nK = Hk_RFAC_2layer.Ks_range[0]
hh=np.linspace(Hmin,Hmax,nH)
kk=np.linspace(Kmin,Kmax,nK)
Hs_op = Hk_RFAC_2layer.optimal_Hs[0]
Ks_op = Hk_RFAC_2layer.optimal_Ks[0]
if len(Hk_RFAC_2layer.std_Hs)==0:
    Hs_std = 0.0
    Ks_std = 0.0
else:
    Hs_std = Hk_RFAC_2layer.std_Hs[0]
    Ks_std = Hk_RFAC_2layer.std_Ks[0]

ax.contourf(hh,kk,amp,levels=25)
ax.plot([Hs_op-3*Hs_std,Hs_op+3*Hs_std],[Ks_op,Ks_op],lw=1,c='black')
ax.plot([Hs_op,Hs_op],[Ks_op-3*Ks_std,Ks_op+3*Ks_std],lw=1,c='black')
ax.set_title(f'0th layer \n' + r'H:{0:4.2f}$\pm${2:4.2f} $\kappa$:{1:4.3f}$\pm${3:4.3f}'.\
            format(Hs_op,Ks_op,3*Hs_std,3*Ks_std))
ax.set_ylabel(r"$\kappa$")
ax.set_xlabel('H [km]')
ax.minorticks_on()
plt.savefig(f"{savedir}/Hk_results.pdf",dpi=900)
plt.savefig(f"{savedir}/Hk_results.png",dpi=900)
plt.close('all')


fig = plt.figure(figsize=(5*cm,5*cm))

bin_nums = Hk_RFAC_2layer.bin_nums
ray_params = np.array(Hk_RFAC_2layer.ray_params)

sampling = 1/Hk_RFAC_2layer.delta

Dataset_labels = [r'RF: $\alpha$=2']

stack_data = Hk_RFAC_2layer.Dataset[0]

if len(bin_nums)!=0:

    axs = fig.subplots(2,1,height_ratios=[1,6],)
    # ax2 = figs[irow,icol].add_subplot(gs[1:])
    ax1 = axs[0]
    ax2 = axs[1]
    bar = ax1.bar(ray_params, bin_nums, color='blue', width=0.001)
    ax1.bar_label(bar,fmt='%d',label_type='edge',size='x-small')
    ax1.set_xticks([])
    ax1.set_ylim(0,max(bin_nums)+4)
    ax1.set_ylabel("Counts")
else:
    ax2 = fig.add_subplot(111)
plot_RF_rayp(ax2,stack_data,ray_params,sampling,0.001,0,30)

id_phase = 0
ray_params = np.sort(ray_params)
for n in range(Hk_RFAC_2layer.Nlayer):
    phases = Hk_RFAC_2layer.phases[n]
    nphase = len(phases)
    Vp = Hk_RFAC_2layer.Vps[n]
    for m in range(nphase):
        weight, data_index, nts, ntp, other_ntpts  = phases[m]
        if data_index!=0:
            continue
        H_n = Hk_RFAC_2layer.optimal_Hs[n]
        K_n = Hk_RFAC_2layer.optimal_Ks[n]
        t = H_n*(nts*(K_n**2/Vp**2-ray_params**2)**(1/2)+ntp*(1/Vp**2-ray_params**2)**(1/2))
        ##add traveling times in previous searched layers
        nlayer_phase = int(len(other_ntpts)/3)
        for k in range(nlayer_phase):
            klayer, nts, ntp = other_ntpts[3*k:3*k+3]
            H_klayer = Hk_RFAC_2layer.optimal_Hs[klayer]
            K_klayer = Hk_RFAC_2layer.optimal_Ks[klayer]
            Vpk = Hk_RFAC_2layer.Vps[k]
            t = t + H_klayer*(nts*(K_klayer**2/Vpk**2-ray_params**2)**(1/2)+ntp*(1/Vpk**2-ray_params**2)**(1/2))

        ax2.plot(ray_params,t,lw=1., ls='--', color=list_colornames[id_phase], label=f"l{n}:phase{m}")
        id_phase +=1


ax2.set_ylim(0,Hk_RFAC_2layer.plotlengths[0])
ax2.invert_yaxis()
xmin,xmax = ax2.get_xlim()
if len(bin_nums)!=0:
    ax1.set_xlim(xmin,xmax)

ax2.legend(loc='lower right')
fig.suptitle(Dataset_labels[0],x=0.5,y=0.9,verticalalignment='baseline')


plt.savefig(f"{savedir}/phasefit.png",dpi=900)
plt.close("all")
