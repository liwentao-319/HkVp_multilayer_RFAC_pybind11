from __future__ import annotations
from HkVp_multilayer.hkvp_stacking import HkVp_stacking 
from HkVp_multilayer.utils import Moveout_Correation
import numpy
import numpy as np 
import numpy.typing
import typing

nparr_float = typing.Annotated[numpy.typing.ArrayLike, numpy.float32]
nparr_int = typing.Annotated[numpy.typing.ArrayLike, numpy.int32]
__all__ = [ 'HkVp_stacking']

class HkVp_stacking:
    def __init__(self,
                delta: float,
                data: nparr_float,
                rayparams: nparr_float, 
                priorparam:dict, 
                phaseparam:dict
                )->None:
        ...
    def do_Hk_stacking(self)->None:
        ...
    def do_Vpc_searching(self,
                         Vpcmin: float,
                         Vpcmax: float,
                         N: int
                         )->None:
        ...
    def do_hk_bootstrap(self,
                              Nb: int,
                              nthread=None)->None:
        ...
    def save_result_to_files(self,
                             savedir:str)->None:
        ...
        
    
class Plot_HkVp(object):
    def __init__(self,savefile:str)->None:
        ...
    def Hk_image(self,savefile:str,dpi:int=900)->None:
        ...
    def Hk_bootstrap_image(self,savefile:str,dpi:int=900)->None:
        ...
    def data_waveforms_rayp(self,savefile:str,
                            dpi:int=900,
                            plotlengths:list = [10,25,10,25,10,25],
                            moveoutphases:list = ['PpPbs','PbP','SbS','PpPms','PmP','bPmSb'])->None:
        ...



def Moveout_Correation(delta:float,
                       data:np.ndarray,
                       vps:np.ndarray,
                       vss:np.ndarray,
                       hs:np.ndarray,
                       rayp:float,
                       phasetype:str)->None:
    ...



    