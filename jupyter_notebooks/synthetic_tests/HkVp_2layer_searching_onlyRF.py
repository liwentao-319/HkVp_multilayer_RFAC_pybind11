import matplotlib.pyplot as plt 
import numpy as np 
from HkVp_multilayer.hkvp_stacking import HkVp_stacking
from HkVp_multilayer.plotting import Plot_HkVp
import time 


maindir = "synthetic_data_S1.0"
ks = [1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5]
depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in ks:
    submaindir = f"{maindir}/Varyp_sediment_crust_mantle_synthetics_c35_ks{k}_dt0.01"
    for depth in depths:

        ##seaching for specific k and depth
        ##when searching for all k and depth, comment out these lines
        # if k<2.5 or depth<7:
        #     continue
        if k!=2.3 or depth!=3:
             continue
        # if k!=2.3 :
        #     continue
        ##################

        stationname = f"sediments_depth{depth}"
        datadir = f"{submaindir}/{stationname}"
        delta = 0.01  # Time step in seconds
        sed_depth_min = max(0.,depth-2)
        sed_depth_max = depth+2
        sed_k_min = k-0.5
        sed_k_max = k+0.5
        priorparam = {
            "Nlayer":2,
            "Vps":[3.6, 6.7],
            "Hs":[100, sed_depth_min, sed_depth_max, 32, 38],
            "Ks":[100, sed_k_min, sed_k_max, 1.6, 2]
        }


        phaseparam = [
            {
                "Pbs":      [0.6, 0, 1,  -1], 
                "PpPbs":    [0.3, 0, 1,   1], 
                "PpSbs":    [-0.1, 0, 2,  0], 
                # "PsSbs":    [-1, 0, 3, -1],
                # "PbP":      [-1, 2, 0,  2], 
                # "SbS":      [-1, 4, 2,  0]  
            },
            {
                "Pms":       [0.6, 1,  1, -1,  1, -1,],  
                #"PpPms-PbP":[1, 1,  1, -1,  1,  1,],  
                "PpPms":    [0.3,  1,  1,  1,  1,  1,], 
                "PpSms":    [-0.1,  1,  2,  0,  2,  0,], 
                # "PmP":      [-1, 3,  0,  2,  0,  2,],  
                # "PmP+PbP":  [1, 3,  0,  4,  0,  2,],  
                # "bPmSb":    [1,  5,  0,  0,  1,  1,],  
                # "PmS":    [1,  5,  1,  1,  1,  1,],  
            }
        ]

        datanpz = np.load(f"{datadir}/data_stack52_10.npz") ##7 5, 0.08
        data_stack = datanpz['data_stack']
        ray_params = datanpz['ray_params']
        traces_suppress = [1,1,3,3,5,5]  ## the trace number where the phases trace should be suppressed; The traces should have the same data source with their index trace
        alphas = [5,2,5,2,5,2]    ## 
        HkVp_instance = HkVp_stacking(delta,
                                    data_stack,
                                    alphas,
                                    traces_suppress,
                                    ray_params,
                                    priorparam,
                                    phaseparam,
                                    0
                                    )
        HkVp_instance.do_Hk_stacking()
        amp_vpc=HkVp_instance.do_Vpc_searching(6.,7.5,75)
        HkVp_instance.do_hk_bootstrap(1000,16)
        HkVp_instance.save_result_to_npz(savedir=datadir,savename='Hk_result_52_onlyRF')
        del HkVp_instance
