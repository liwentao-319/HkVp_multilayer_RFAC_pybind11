
import numpy as np
from glob import glob 
import os
from obspy import read
from Hk_stacking_2layer_vp_YuDeRe import Hk_searching_RF_2layer_Yu
from AC import Ac
import pandas as pd
import pickle
import numpy as np





maindir = "seismic_data_RFAC"

stationname = "TASTE_T33"

datadir=f"{maindir}/{stationname}"


##read RF data
stream_rf2 = read(f"{datadir}/*.BHR.sac_g5")
streams = [stream_rf2]

dataparam ={
    "delta":0.01,
    "pretimes":[5],
    "lengths":[100],
    "norm_type":[1],
    "plotlengths":[25]
}

priorparam = {
    "Nlayer":2,
    "Vps":[6.36, 3.],
    "Hs":[[20,45,101],[0,7,101]],
    "Ks":[[1.6,2.0,101],[1.5,5,101]]
}

weights = [
    [0.5, 0.4, -0.1],
    [0.05, 0.7, -0.25]
]

phasestackparams=None

Hk_RFDeRe_2layer = Hk_searching_RF_2layer_Yu(stationname=stationname,
                                            data_st=streams,
                                            dataparam=dataparam,
                                            priorparam=priorparam,
                                            weights=weights,
                                            phasestackparams=phasestackparams
                                        )

Hk_RFDeRe_2layer.do_searching()
Hk_RFDeRe_2layer.hk_bootstrap_estimate(100,8)
# Hk_RFAC_2layer.save_result_to_figs(savedir=savedir)
savedir = f"Hk_RF2layer_DeRe/{stationname}"
if not os.path.isdir(savedir):
    os.makedirs(savedir)
Hk_RFDeRe_2layer.save_result_to_files(savedir=savedir)
Hk_RFDeRe_2layer.save_result_to_figs(savedir=savedir,pic_format='png')
with open(f"{savedir}/Hk_results.bin", 'wb') as fd:
    pickle.dump(Hk_RFDeRe_2layer,fd)
