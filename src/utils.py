

from scipy.interpolate import interp1d
import numpy as np
import numpy 
import typing


def Moveout_Correation(delta:float,
                       data:np.ndarray,
                       vps:np.ndarray,
                       vss:np.ndarray,
                       hs:np.ndarray,
                       rayp:float,
                       phasetype:str):
    """
    Moveout_correction for RFs with at leat 2 layers model

    Created by Wentao Li at Nov. 12th, 2024


    """

    if rayp>0.1:
        print(f"Invalid Ray parameter:{rayp}")
        return -1 

    tvs_cum_p = np.cumsum(hs/vps)
    tvs_cum_s = np.cumsum(hs/vss)

    ts_cum_p = np.cumsum(hs*np.sqrt(1/vps**2-rayp**2))
    ts_cum_s = np.cumsum(hs*np.sqrt(1/vss**2-rayp**2))


    if phasetype=="Pbs":
        R = (ts_cum_s[-2]-ts_cum_p[-2])/(tvs_cum_s[-2]-tvs_cum_p[-2])
    elif phasetype=="PpPbs":
        R = (ts_cum_s[-2]+ts_cum_p[-2])/(tvs_cum_s[-2]+tvs_cum_p[-2])
    elif phasetype=="PbP":
        R = (ts_cum_p[-2])/(tvs_cum_p[-2])
    elif phasetype=="SbS":
        R = (ts_cum_s[-2])/(tvs_cum_s[-2])    
    elif phasetype=="Pms":
        R = (ts_cum_s[-1]-ts_cum_p[-1])/(tvs_cum_s[-1]-tvs_cum_p[-1])
    elif phasetype=="PpPms":
        R = (ts_cum_s[-1]+ts_cum_p[-1])/(tvs_cum_s[-1]+tvs_cum_p[-1])
    elif phasetype=="PmP":
        R = (ts_cum_p[-1])/(tvs_cum_p[-1])
    elif phasetype=="bPmSb":
        R = (ts_cum_s[-1]+ts_cum_p[-1]-ts_cum_s[-2]-ts_cum_p[-2])/(tvs_cum_s[-1]+tvs_cum_p[-1]-tvs_cum_s[-2]-tvs_cum_p[-2])
    else:
        print(f"phase type: {phasetype} is not in [Pbs, PpPbs, PbP, SbS, Pms, PpPms, PmP, bPmSb]")
        return -1
    ts = np.arange(len(data))*delta
    ts_stretch = ts/R
    fun_interp = interp1d(ts_stretch,data,fill_value="extrapolate")
    data_stretch = fun_interp(ts)
    return data_stretch