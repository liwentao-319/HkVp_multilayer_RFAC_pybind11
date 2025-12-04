
import numpy as np
from obspy import read
from obspy.core import Stream
from glob import glob 
from math import floor
from multiprocessing import Pool
import multiprocessing
import math
from numpy.typing import NDArray
from HkVp_multilayer import hk_stacking
import os 

list_colornames = ['orange','gold','cyan','skyblue','blueviolet','violet','pink']


class HkVp_stacking(object):
    """
    Discription   : Modified multilayer H-k stacking
    Author        : Wentao Li
    Time          : Jul 6th, 2023
    Version       : 1.0

    modified at 2025/03/27 by Wentao Li
        copy stream from data_st to self.datast

    Modified at 2025/03/27 by Wentao Li
        I added the function to searth the optimal Vp

    Modified at 2025/08/04 by Wentao Li
        use C++ extention to do H-k grid searching
        add a function to do Vpc searching
        estimate the estimation error for H-k stacking using bootstrap method 

    """
    def __init__(self,
                 delta: float,
                 data: NDArray[np.float32],
                 alphas:list,
                 traces_suppress:list,
                 rayparams: NDArray[np.float32], 
                 priorparam:dict,
                 phaseparam:list,      ##list of dict
                 stack_type:int = 2    ##0: self-defined weights; 1: PWS; 2: PWS2
                 ):
        """

        """
        ## 
        ## each seismic phase will match a index in the dataset
        self.data = data
        nset,nrayp,npts = data.shape
        self.nrayp = nrayp
        self.delta = delta
        self.alphas = alphas
        self.traces_suppress = traces_suppress 
        ## prior information
        self.Nlayer = priorparam["Nlayer"]
        self.Vps = priorparam["Vps"]
        self.Hs = priorparam["Hs"]
        self.Ks = priorparam["Ks"]
        
        self.rayparams = rayparams
        ##phases information
        phasenum = 0
        self.phasesize = [len(param.keys()) for param in phaseparam]  # Assuming all layers have the same size
        for nphase in self.phasesize:
           # print(nphase)
            phasenum += nphase
        self.phasenum = phasenum
        self.phasetimes = np.zeros([phasenum,nrayp])

        phaseparam_expanded = []
        allphasenamelist = []
        for paramlayer in phaseparam:
            phasenames = paramlayer.keys()
            allphasenamelist += phasenames
            for phasename in phasenames:
                param = paramlayer[phasename]
                phaseparam_expanded = phaseparam_expanded + param
        self.phaselist = [float(num) for num in phaseparam_expanded]
        self.stack_type = stack_type
        self.allphasenamelist = allphasenamelist
        ## initialize other parameters
        self.Vpc_optimal = self.Vps[-1]
        self.Hs_optimal = [0 for i in range(self.Nlayer)]
        self.Ks_optimal = [0 for i in range(self.Nlayer)]
        self.Hs_std = [0 for i in range(self.Nlayer)]
        self.Ks_std = [0 for i in range(self.Nlayer)]
        self.Hs_randoms = [0]
        self.Ks_randoms = [0]
        self.stacked_image = None 
        self.phase_weights = None
        self.Vpc_searching_Vpcs = [0]
        self.Vpc_searching_Hcs = [0]
        self.Vpc_searching_Kcs = [0]
        self.Vpc_searching_Amps = [0]

    def do_Hk_stacking(self):

        hk_instance = hk_stacking.Hk_stacking_multilayer_Vp(
                                        self.Nlayer,
                                        self.delta,
                                        self.rayparams,
                                        self.Vps,
                                        self.Hs,
                                        self.Ks,
                                        self.phasesize,
                                        self.phaselist,
                                        self.alphas,
                                        self.traces_suppress,
                                        self.data)
                                
        
        if self.stack_type == 0:
            hk_instance.Hk_stacking()
        elif self.stack_type == 1:
            hk_instance.Hk_stacking_PWS1()
        elif self.stack_type == 2:
            hk_instance.Hk_stacking_PWS2()
        else:
            raise KeyError("increct stack type")
            
        self.Hs_optimal = hk_instance.get_Hs_optimal()
        self.Ks_optimal = hk_instance.get_Ks_optimal()
        self.stacked_image = hk_instance.get_stacked_image()
        self.phase_weights = hk_instance.get_weights()
        self.phasetimes = hk_instance.get_phasetimes()


    def do_Vpc_searching(self,
                         Vpcmin: float,
                         Vpcmax: float,
                         N: int
                         ):
        """
        Searching the optimal bulk crustal Vp
        """
        Vpcs = np.linspace(Vpcmin,Vpcmax,N)

        Stacked_Amps_opt = []
        Stacked_Hcs_opt = []
        Stacked_Kcs_opt = []
        hh = np.linspace(self.Hs[-2], self.Hs[-1],int(self.Hs[0]))
        kk = np.linspace(self.Ks[-2], self.Ks[-1],int(self.Ks[0]))
        for Vpc in Vpcs:
            vps = [self.Vps[0],Vpc]
            hk_instance = hk_stacking.Hk_stacking_multilayer_Vp(self.Nlayer,
                                self.delta,
                                self.rayparams,
                                vps,
                                self.Hs,
                                self.Ks,
                                self.phasesize,
                                self.phaselist,
                                self.alphas,
                                self.traces_suppress,
                                self.data)
            if self.stack_type == 0:
                hk_instance.Hk_stacking()
            elif self.stack_type == 1:
                hk_instance.Hk_stacking_PWS1()
            elif self.stack_type == 2:
                hk_instance.Hk_stacking_PWS2()
            else:
                raise KeyError("increct stack type")

            stacked_images = hk_instance.get_stacked_image()
            stacked_image_crust = stacked_images[-1].T
            index_max=stacked_image_crust.argmax()
            ik=floor(index_max/self.Hs[0])
            ih=index_max%self.Hs[0]
            Stacked_Amps_opt.append(stacked_image_crust[ik,ih])
            Stacked_Hcs_opt.append(hh[ih])
            Stacked_Kcs_opt.append(kk[ik])


        self.Vpc_searching_Vpcs = Vpcs
        self.Vpc_searching_Amps = Stacked_Amps_opt
        self.Vpc_searching_Hcs = Stacked_Hcs_opt
        self.Vpc_searching_Kcs = Stacked_Kcs_opt

    def do_hk_bootstrap(self,
                        Nb: int,
                        nthread=4):

        ntrace = len(self.rayparams)
        rgn = np.random.default_rng(12345)
        random_indexes = [rgn.choice(ntrace, ntrace, replace=True) for _ in range(Nb)]
        if nthread is not None and nthread > 1:
            Hs_randoms = [[] for _ in range(self.Nlayer)]
            Ks_randoms = [[] for _ in range(self.Nlayer)]            
            nwork = int(Nb / nthread)
            timeout_per_batch = nwork*60
            args_pool = [random_indexes[i*nwork : (i+1)*nwork] for i in range(nthread)]  
            with Pool(processes=nthread) as pool:
                async_results = []
                for args in args_pool:
                    async_results.append(pool.apply_async(self._bootstrap, (args,)))
                # Collect results with timeout control
                for i, result in enumerate(async_results):
                    try:
                        # Get result with timeout (timeout_per_batch seconds per batch)
                        Hs_batch, Ks_batch = result.get(timeout=timeout_per_batch)
                        
                        for j in range(self.Nlayer):
                            Hs_randoms[j].extend(Hs_batch[j])
                            Ks_randoms[j].extend(Ks_batch[j])
                            
                    except multiprocessing.TimeoutError:
                        print(f"Batch {i} timed out after {timeout_per_batch} seconds")
                        pool.terminate()  # Kill all processes immediately
                        raise RuntimeError("Bootstrap terminated due to timeout")
                    except Exception as e:
                        print(f"Error in batch {i}: {str(e)}")
                        pool.terminate()
                        raise
            Hs_randoms = [np.array(h) for h in Hs_randoms]
            Ks_randoms = [np.array(k) for k in Ks_randoms]
        else:
            Hs_randoms, Ks_randoms = self._bootstrap(random_indexes)
        
        self.Hs_randoms = Hs_randoms
        self.Ks_randoms = Ks_randoms
        print("Calculate the standard deviation")
        self.Hs_std = [np.std(h) for h in Hs_randoms]
        self.Ks_std = [np.std(k) for k in Ks_randoms]
        print("Finished")

    def _bootstrap(self,indexes):
        print(f"Process {multiprocessing.current_process().pid} working (Job Number : {len(indexes)})")
        Hs_ops = []
        Ks_ops = []

        for j in range(self.Nlayer):
            Hs_ops.append([])
            Ks_ops.append([])
        ##preassign buffer
        buf_shape = (self.data.shape[0], len(indexes[0]), self.data.shape[2])
        buffer = np.empty(buf_shape, dtype=self.data.dtype)
        for index in indexes:
            np.take(self.data, index, axis=1, out=buffer)
            random_rayparams = [self.rayparams[iii] for iii in index]
            hk_instance = hk_stacking.Hk_stacking_multilayer_Vp(self.Nlayer,
                                            self.delta,
                                            random_rayparams,
                                            self.Vps,
                                            self.Hs,
                                            self.Ks,
                                            self.phasesize,
                                            self.phaselist,
                                            self.alphas,
                                            self.traces_suppress,
                                            buffer)
            if self.stack_type == 0:
                hk_instance.Hk_stacking()
            elif self.stack_type == 1:
                hk_instance.Hk_stacking_PWS1()
            elif self.stack_type == 2:
                hk_instance.Hk_stacking_PWS2()
            else:
                raise KeyError("increct stack type")
            Hs_op = hk_instance.get_Hs_optimal()
            Ks_op = hk_instance.get_Ks_optimal()
            for j in range(self.Nlayer):
                Hs_ops[j].append(Hs_op[j])
                Ks_ops[j].append(Ks_op[j])
        print(f"Process {multiprocessing.current_process().pid} finished")

        return Hs_ops, Ks_ops

    def save_result_to_npz(self,savename,savedir="."):
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        savefile = f"{savedir}/{savename}_Hk_result.npz"
        np.savez(savefile, delta = self.delta,
                           rayparams = self.rayparams,
                           data = self.data,
                           Nlayer = self.Nlayer,
                           phasesize = self.phasesize,
                           phaselist = self.phaselist,
                           allphasenamelist = self.allphasenamelist,
                           phaseweights = self.phase_weights,
                           phasetimes = self.phasetimes,
                           Vps = self.Vps,
                           Hs = self.Hs, 
                           Ks = self.Ks, 
                           Hs_optimal = self.Hs_optimal,
                           Hs_std = self.Hs_std,
                           Ks_optimal = self.Ks_optimal,
                           Ks_std = self.Ks_std,
                           Hs_randoms = self.Hs_randoms,
                           Ks_randoms = self.Ks_randoms,
                           stacked_image=self.stacked_image,
                           Vpc_searching_Vpcs = self.Vpc_searching_Vpcs,
                           Vpc_searching_Amps = self.Vpc_searching_Amps,
                           Vpc_searching_Hcs = self.Vpc_searching_Hcs,
                           Vpc_searching_Kcs = self.Vpc_searching_Kcs
                )

    


