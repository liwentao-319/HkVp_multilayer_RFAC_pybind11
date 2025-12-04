import matplotlib.pyplot as plt 
import numpy as np 
from HkVp_multilayer.hkvp_stacking import HkVp_stacking
from HkVp_multilayer.plotting import Plot_HkVp
import time 
maindir = "synthetic_data_S1.0"
depth = 5
submaindir = f"{maindir}/Varyp_2sediment_crust_mantle_synthetics_c35_d1s5_k2s1.8_dt0.01_G2"
stationname = f"sediments_depth{depth}"
datadir = f"{submaindir}/{stationname}"
delta = 0.01  # Time step in seconds

priorparam = {
    "Nlayer":3,
    "Vps":[3.0, 4.5, 6.7],
    "Hs":[100, 3, 7, 3, 8, 32, 38],
    "Ks":[100, 2.0, 3.0, 1.5, 2.5, 1.6, 1.85]

}

phaseparam = [
    {
       "Pis":       [1, 0,  1, -1],
       "PpPis":     [1, 0,  1,  1], 
        "PpSis":     [-1, 0, 2,  0], 
       "PsSis":     [-1, 0, 3,  -1], 
       "PiP":       [-1, 2, 0,  2], 
       "SiS":       [-1, 4, 2,  0]   
    },
    {
       "Pbs":        [1, 0,  1, -1, 1,  -1], 
       "PpPbs":      [1, 0,  1,  1, 1,   1], 
       "PpSbs":      [-1, 0, 2,  0, 2,   0], 
       "PsSbs":      [-1, 0, 3, -1, 3,  -1], 
       "PbP":        [-1, 2, 0,  2, 0,   2], 
       "SbS":        [-1, 4, 2,  0, 2,   0]  
    },
    {
        "Pms":         [1, 1,  1, -1,  1, -1,  1, -1,],  
        # "PpPms-PbP":   [1, 1,  1, -1,  1, -1,  1,  1,],  
        "PpPms":       [1, 1,  1,  1,  1,  1,  1,  1,], 
        "PmP":         [-1,3,  0,  2,  0,  2,  0,  2,],  
        # "PmP+PbP":     [1, 3,  0,  4,  0,  4,  0,  2,], 
        "bPmSb":       [1, 5,  0,  0,  0,  0,  1,  1,],  
        # "PmS":         [1, 5,  1,  1,  1,  1,  1,  1,],  
    }
]


datanpz = np.load(f"{datadir}/data_stack52_10.npz") ##7 5, 0.08
data_stack = datanpz['data_stack']
ray_params = datanpz['ray_params']
traces_suppress = [1,1,2,2,3,3]  ## the dataset's source
alphas = [5,2,5,2,5,2]    ##   
HkVp_instance = HkVp_stacking(delta,
                            data_stack,
                            alphas,
                            traces_suppress,
                            ray_params,
                            priorparam,
                            phaseparam,
                            2
                            )
# print(111)
HkVp_instance.do_Hk_stacking()
# # amp_vpc=HkVp_instance.do_Vpc_searching(6.,7.,10)
HkVp_instance.do_hk_bootstrap(1000,16)
HkVp_instance.save_result_to_npz(savedir=datadir,savename='Hk_result_52_10')

