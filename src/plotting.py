import numpy as np 
import matplotlib.pyplot as plt 
import os 
from HkVp_multilayer.hkvp_stacking import HkVp_stacking
from HkVp_multilayer.utils import Moveout_Correation
CM2INCH = 1/2.54
list_colornames = ['white','white','white','skyblue','blueviolet','violet','pink']


class Plot_HkVp(object):
    def __init__(self,savefile:str):
        """
        savefile is the file saved by HkVp_stacking.save_result_to_npz() 

        the dirt returned by np.load(savefile) has the following keys:
            delta, 
            rayparams,
            data,
            Nlayer, 
            phasesize, 
            phaselist,
            phaseweights,
            phasetimes, 
            Hs, 
            Ks,
            Vps, 
            Hs_optimal,
            Hs_std, 
            Ks_optimal,
            Ks_std,
            Hs_randoms, 
            Ks_randoms,
            stacked_image,
        """
        ## each seismic phase will match a index in the dataset
        self.hk_results = np.load(savefile)
        self.Hk_pltwidth = 7.5*CM2INCH
        self.Hk_pltheight = 6*CM2INCH
        self.phasefit_pltwidth = 5*CM2INCH
        self.phasefit_pltheight = 4*CM2INCH
        plt.rcParams.update({
            'axes.linewidth': 0.5,       
            'xtick.major.width': 0.5,     
            'ytick.major.width': 0.5,   
            'xtick.minor.width': 0.4,    
            'ytick.minor.width': 0.4,   
            'xtick.major.size': 4,       
            'ytick.major.size': 4,   
            'xtick.minor.size': 2, 
            'ytick.minor.size': 2, 
            'font.size':8
        })





    def Hk_image(self,savefile:str,dpi:int=900,true_model=[]):
        Nlayer = int(self.hk_results['Nlayer'])
        Hs_optimal = self.hk_results['Hs_optimal']
        Ks_optimal = self.hk_results['Ks_optimal']
        Hs_std = self.hk_results['Hs_std']
        Ks_std = self.hk_results['Ks_std']
        Hs_randoms = self.hk_results['Hs_randoms']
        Ks_randoms = self.hk_results['Ks_randoms']
        stacked_image = self.hk_results['stacked_image']
        Hs = self.hk_results['Hs']
        Ks = self.hk_results['Ks']
        phaseweights = self.hk_results['phaseweights']
        phasesize = self.hk_results['phasesize']
        phaselist = self.hk_results['phaselist']
        allphaselist = self.hk_results['allphasenamelist']
        if fig==None:
            fig = plt.figure(figsize=(self.Hk_pltwidth*Nlayer,self.Hk_pltheight),tight_layout=True)
        # subfigs = fig.subfigures(1,Nlayer,wspace=0.07)
        ##plot phase fitting results
        phasecount = 0
        currentindex = 0
        if len(true_model)==Nlayer:
            plot_truemodel = True
        else:
            plot_truemodel = False



        for j in range(Nlayer):
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            ax = fig.add_subplot(1,Nlayer,j+1)
            # ax.set_aspect('equal')

            ##plot Hk_image
            # ax.set_title(f"Layer {j+1}", fontsize=14)
            ax.set_xlabel('H [km]', fontsize=10)
            ax.set_ylabel(r'$\kappa$', fontsize=10)
            ax.minorticks_on()
            ax.grid(True,ls='--',lw=0.5,alpha=0.3,c='w')
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            image = stacked_image[j,:,:].T
            hh = np.linspace(Hs[1+j*2], Hs[1+j*2+1],int(Hs[0]))
            kk = np.linspace(Ks[1+j*2], Ks[1+j*2+1],int(Ks[0]))
            ax.contourf(hh,kk,image,levels=25)
            ax.set_xlim(Hs[1+j*2],Hs[1+j*2+1])
            ax.set_ylim(Ks[1+j*2],Ks[1+j*2+1])
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            ax.errorbar(h_opt,k_opt,xerr=h_std,yerr=k_std,fmt='o-', color='#2E86AB', ecolor='#A23B72',elinewidth=1.0,capsize=2, capthick=1.0, markersize=2 )
            if plot_truemodel:
                H_true,k_true = true_model[j]
                ax.scatter(H_true,k_true,marker='+',markersize=8,color='red')
            if j==0 and Nlayer==2:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{sed}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{sed}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            elif j==Nlayer-1:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{crystal\ crust}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{crystal\ crust}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            else:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                        rf"$H_{{\mathrm{{sed}}{j+1}}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                        rf"$\kappa_{{\mathrm{{sed}}{j+1}}}={k_opt:.3f} \pm {k_std:.3f} $", 
                        fontsize=6, color='white', ha='left', va='top')

            nphase = phasesize[j]
            phaseweights_str = "Contributions [%]\n"

            weight_sum = 0.0
            for iphase in range(nphase):
                weight_sum += abs(phaseweights[phasecount+iphase])

            for iphase in range(nphase):
                itrace = phaselist[currentindex + iphase*(2 * (j + 2))+1]
                
                phasename = allphaselist[phasecount+iphase]

                weight = abs(phaseweights[phasecount+iphase])/weight_sum*100
                tempstr = f"{phasename}: "
                phaseweights_str = phaseweights_str+tempstr

                tempstr = f"{weight:3.1f}\n"
                phaseweights_str = phaseweights_str+tempstr

                # ax.text(ray_params[-1],phasetime[0],f"{abs(weight):.3f}",ha="right",va="bottom",fontsize=6,bbox={'facecolor':"white",'edgecolor':'white','pad':0})#

            phasecount += nphase
            currentindex += nphase * (2 * (j + 2))

            ax.text(xmin + 0.99 * (xmax - xmin), 
                    ymin + 0.99 * (ymax - ymin), 
                    phaseweights_str, 
                    multialignment="right",
                    fontsize=6, 
                    color='white', 
                    ha='right', 
                    va='top')

        plt.savefig(savefile,dpi=dpi,bbox_inches='tight')


    def Hk_image_onlyRF(self,savefile:str,dpi:int=900,true_model=[]):
        Nlayer = int(self.hk_results['Nlayer'])
        Hs_optimal = self.hk_results['Hs_optimal']
        Ks_optimal = self.hk_results['Ks_optimal']
        Hs_std = self.hk_results['Hs_std']
        Ks_std = self.hk_results['Ks_std']
        Hs_randoms = self.hk_results['Hs_randoms']
        Ks_randoms = self.hk_results['Ks_randoms']
        stacked_image = self.hk_results['stacked_image']
        Hs = self.hk_results['Hs']
        Ks = self.hk_results['Ks']
        phaseweights = self.hk_results['phaseweights']
        phasesize = self.hk_results['phasesize']
        phaselist = self.hk_results['phaselist']
        allphaselist = self.hk_results['allphasenamelist']
        fig = plt.figure(figsize=(self.Hk_pltwidth*Nlayer,self.Hk_pltheight),tight_layout=True)
        # subfigs = fig.subfigures(1,Nlayer,wspace=0.07)
        ##plot phase fitting results
        phasecount = 0
        currentindex = 0
        if len(true_model)==Nlayer:
            plot_truemodel = True
        else:
            plot_truemodel = False



        for j in range(Nlayer):
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]


            ax = fig.add_subplot(1,Nlayer,j+1)
            # ax.set_aspect('equal')

            ##plot Hk_image
            # ax.set_title(f"Layer {j+1}", fontsize=14)
            ax.set_xlabel('H [km]', fontsize=10)
            ax.set_ylabel(r'$\kappa$', fontsize=10)
            ax.minorticks_on()
            ax.grid(True,ls='--',lw=0.5,alpha=0.3,c='w')
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            image = stacked_image[j,:,:].T
            hh = np.linspace(Hs[1+j*2], Hs[1+j*2+1],int(Hs[0]))
            kk = np.linspace(Ks[1+j*2], Ks[1+j*2+1],int(Ks[0]))
            ax.contourf(hh,kk,image,levels=25)
            ax.set_xlim(Hs[1+j*2],Hs[1+j*2+1])
            ax.set_ylim(Ks[1+j*2],Ks[1+j*2+1])
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            ax.errorbar(h_opt,k_opt,xerr=h_std,yerr=k_std,fmt='o-', color='#2E86AB', ecolor='#A23B72',elinewidth=1.0,capsize=2, capthick=1.0, markersize=2 )
            if plot_truemodel:
                H_true,k_true = true_model[j]
                ax.scatter(H_true,k_true,marker='+',markersize=8,color='red')
            if j==0 and Nlayer==2:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{sed}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{sed}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            elif j==Nlayer-1:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{crystal\ crust}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{crystal\ crust}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            else:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                        rf"$H_{{\mathrm{{sed}}{j+1}}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                        rf"$\kappa_{{\mathrm{{sed}}{j+1}}}={k_opt:.3f} \pm {k_std:.3f} $", 
                        fontsize=6, color='white', ha='left', va='top')


            nphase = phasesize[j]
            phaseweights_str = "Weights [%]\n"

            weight_sum = 0.0
            for iphase in range(nphase):
                weight_sum += abs(phaseweights[phasecount+iphase])

            for iphase in range(nphase):
                itrace = phaselist[currentindex + iphase*(2 * (j + 2))+1]
                
                phasename = allphaselist[phasecount+iphase]

                weight = abs(phaseweights[phasecount+iphase])/weight_sum*100
                tempstr = f"{phasename}: "
                phaseweights_str = phaseweights_str+tempstr

                tempstr = f"{weight:3.1f}\n"
                phaseweights_str = phaseweights_str+tempstr

                # ax.text(ray_params[-1],phasetime[0],f"{abs(weight):.3f}",ha="right",va="bottom",fontsize=6,bbox={'facecolor':"white",'edgecolor':'white','pad':0})#

            phasecount += nphase
            currentindex += nphase * (2 * (j + 2))

            ax.text(xmin + 0.99 * (xmax - xmin), 
                    ymin + 0.99 * (ymax - ymin), 
                    phaseweights_str, 
                    multialignment="right",
                    fontsize=6, 
                    color='white', 
                    ha='right', 
                    va='top')

        plt.savefig(savefile,dpi=dpi,bbox_inches='tight')



    def Hk_bootstrap_image(self,savefile:str,dpi:int=900):
        Nlayer = int(self.hk_results['Nlayer'])
        Hs_optimal = self.hk_results['Hs_optimal']
        Ks_optimal = self.hk_results['Ks_optimal']
        Hs_std = self.hk_results['Hs_std']
        Ks_std = self.hk_results['Ks_std']
        Hs_randoms = self.hk_results['Hs_randoms']
        Ks_randoms = self.hk_results['Ks_randoms']
        stacked_image = self.hk_results['stacked_image']
        Hs = self.hk_results['Hs']
        Ks = self.hk_results['Ks']
        phaseweights = self.hk_results['phaseweights']
        allphaselist = ["Pbs","PpPbs","PpSbs","PsSbs","PpPbs+Sbs","PbP","SbS",
                         "Pms","PpPms-PbP","PpPms","PmP","PmP+PbP","bPmSb","PmS"]
        phasesize = self.hk_results['phasesize']
        phaselist = self.hk_results['phaselist']
        if len(allphaselist)==len(phaseweights):
            plotphasename=True
        else:
            plotphasename=False
        fig = plt.figure(figsize=(self.Hk_pltwidth*Nlayer,self.Hk_pltheight))
        # subfigs = fig.subfigures(1,Nlayer,wspace=0.07)
        bootstrap_color = "brown"

        ##plot phase fitting results
        phasecount = 0
        currentindex = 0


        for j in range(Nlayer):
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            hs_random = Hs_randoms[j]
            ks_random = Ks_randoms[j]

            ax = fig.add_subplot(1,2,j+1)
            # ax.set_aspect('equal')

            ##plot Hk_image
            # ax.set_title(f"Layer {j+1}", fontsize=14)
            ax.set_xlabel('H [km]', fontsize=10)
            ax.set_ylabel(r'$\kappa$', fontsize=10)
            ax.minorticks_on()
            ax.grid(True,ls='--',lw=0.5,alpha=0.3,c='w')
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            image = stacked_image[j,:,:].T
            hh = np.linspace(Hs[1+j*2], Hs[1+j*2+1],int(Hs[0]))
            kk = np.linspace(Ks[1+j*2], Ks[1+j*2+1],int(Ks[0]))
            ax.contourf(hh,kk,image,levels=25)
            ax.set_xlim(Hs[1+j*2],Hs[1+j*2+1])
            ax.set_ylim(Ks[1+j*2],Ks[1+j*2+1])
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()
            ax.scatter(hs_random,ks_random,s=5,marker='+',color=bootstrap_color,alpha=0.04)
            ax.scatter([h_opt],[k_opt],marker='+',color='black')
            if j==0 and Nlayer==2:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{sed}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{sed}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            elif j==Nlayer-1:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                    rf"$H_\mathrm{{crystal\ crust}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{crystal\ crust}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='top')
            else:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                        rf"$H_{{\mathrm{{sed}}{j+1}}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                        rf"$\kappa_{{\mathrm{{sed}}{j+1}}}={k_opt:.3f} \pm {k_std:.3f} $", 
                        fontsize=6, color='white', ha='left', va='top')
            
            nphase = phasesize[j]
            phaseweights_str = "PHASE WEIGHTS\n"
            for iphase in range(nphase):
                itrace = phaselist[currentindex + iphase*(2 * (j + 2))+1]
                
                phasename = allphaselist[phasecount+iphase]

                weight = abs(phaseweights[phasecount+iphase])
                tempstr = f"{phasename}: "
                phaseweights_str = phaseweights_str+tempstr

                tempstr = f"{weight:.3f}\n"
                phaseweights_str = phaseweights_str+tempstr

                    # ax.text(ray_params[-1],phasetime[0],f"{abs(weight):.3f}",ha="right",va="bottom",fontsize=6,bbox={'facecolor':"white",'edgecolor':'white','pad':0})#
                    
            phasecount += nphase
            currentindex += nphase * (2 * (j + 2))

            ax.text(xmin + 0.99 * (xmax - xmin), 
                    ymin + 0.99 * (ymax - ymin), 
                    phaseweights_str, 
                    multialignment="right",
                    fontsize=6, 
                    color='white', 
                    ha='right', 
                    va='top')


            ##plot bootstrap randoms along Hs
            ax1 = ax.inset_axes([1.05, 0, 0.2, 1 ],sharey=ax)
            ax1.tick_params(axis="y", labelleft=False)
            binwidth = (ymax-ymin)/40.
            bins = np.arange(ymin,ymax+binwidth,binwidth)
            ax1.hist(ks_random,bins=bins,orientation='horizontal',color=bootstrap_color)
            ax1.set_ylim(Ks[1+j*2],Ks[1+j*2+1])
            ax1.set_xlabel("Num")

            ##plot bootstrap randoms along Hs
            ax2 = ax.inset_axes([0, 1.05, 1, 0.2],sharex=ax)
            ax2.tick_params(axis="x", labelbottom=False)
            binwidth = (xmax-xmin)/40.
            bins = np.arange(xmin,xmax+binwidth,binwidth)
            ax2.hist(hs_random,bins=bins,color=bootstrap_color)
            ax2.set_xlim(Hs[1+j*2],Hs[1+j*2+1])
            ax2.set_ylabel("Num")
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.52,hspace=0.15)
        plt.savefig(savefile,dpi=dpi,bbox_inches='tight')

    def data_Hk_Phasefitting(self,savefile:str,
                            dpi:int=900,
                            plotlengths:list = [10,25,10,25,10,25],
                            moveoutphases:list = ['PpPbs','PpPms','PbP','PmP','SbS','bPmSb']):
        labels = ["(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)","(i)"]
        labelindex = 0
        Nlayer = int(self.hk_results['Nlayer'])        
        fig = plt.figure(layout='constrained', figsize=(15/2.54, 16/2.54))
        subfigs = fig.subfigures(2,1,wspace=0.01,hspace=0.06,height_ratios=[2.,4])
        hkfig = subfigs[0]
        phasefig = subfigs[1]

        ##plot hk image

        Hs_optimal = self.hk_results['Hs_optimal']
        Ks_optimal = self.hk_results['Ks_optimal']
        Hs_std = self.hk_results['Hs_std']
        Ks_std = self.hk_results['Ks_std']
        Hs_randoms = self.hk_results['Hs_randoms']
        Ks_randoms = self.hk_results['Ks_randoms']
        stacked_image = self.hk_results['stacked_image']
        Hs = self.hk_results['Hs']
        Ks = self.hk_results['Ks']
        phaseweights = self.hk_results['phaseweights']
        phasesize = self.hk_results['phasesize']
        phaselist = self.hk_results['phaselist']
        allphaselist = self.hk_results['allphasenamelist']
        # subfigs = fig.subfigures(1,Nlayer,wspace=0.07)
        ##plot phase fitting results
        phasecount = 0
        currentindex = 0
        axs = hkfig.subplots(1,Nlayer)
        hkfig.set_tight_layout=True


        for j in range(Nlayer):
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]

            ax = axs[j]
            if Nlayer==2:
                pos =  [1.0/9.0+(1.0/9.0+1.0/3.0)*j,0.1,1.0/3.0,0.8]
            elif Nlayer==3:
                pos = [(0.15)/Nlayer+1/Nlayer*j,0.1,0.8/3.0,0.8]
            else:
                print("Unsupport Nlayer = ", Nlayer)
                return -1
            ax.set_position(pos)
            # ax.set_aspect('equal')
            ##plot Hk_image
            # ax.set_title(f"Layer {j+1}", fontsize=14)
            ax.set_xlabel('H [km]', fontsize=10)
            if j==0:
                ax.set_ylabel(r'$\kappa$', fontsize=10)
            ax.minorticks_on()
            ax.grid(True,ls='--',lw=0.5,alpha=0.3,c='w')
            h_opt = Hs_optimal[j]
            k_opt = Ks_optimal[j]
            h_std = Hs_std[j]
            k_std = Ks_std[j]
            image = stacked_image[j,:,:].T
            hh = np.linspace(Hs[1+j*2], Hs[1+j*2+1],int(Hs[0]))
            kk = np.linspace(Ks[1+j*2], Ks[1+j*2+1],int(Ks[0]))
            ax.contourf(hh,kk,image,levels=25)
            ax.set_xlim(Hs[1+j*2],Hs[1+j*2+1])
            ax.set_ylim(Ks[1+j*2],Ks[1+j*2+1])
            xmin,xmax = ax.get_xlim()
            ymin,ymax = ax.get_ylim()

            ax.errorbar(h_opt,k_opt,xerr=h_std,yerr=k_std,fmt='o-', color='#2E86AB', ecolor='#A23B72',elinewidth=1.0,capsize=2, capthick=1.0, markersize=2 )
            if j==0 and Nlayer==2:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.01 * (ymax - ymin), 
                    rf"$H_\mathrm{{sed}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{sed}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='bottom')
            elif j==Nlayer-1:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.01 * (ymax - ymin), 
                    rf"$H_\mathrm{{crystal\ crust}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                    rf"$\kappa_\mathrm{{crystal\ crust}}={k_opt:.3f} \pm {k_std:.3f} $" , 
                    fontsize=6, color='white', ha='left', va='bottom')
            else:
                ax.text(xmin + 0.01 * (xmax - xmin), ymin + 0.01 * (ymax - ymin), 
                        rf"$H_{{\mathrm{{sed}}{j+1}}}={h_opt:.2f} \pm {h_std:.2f} \mathrm{{km}}$"+"\n"+
                        rf"$\kappa_{{\mathrm{{sed}}{j+1}}}={k_opt:.3f} \pm {k_std:.3f} $", 
                        fontsize=6, color='white', ha='left', va='bottom')
            ax.text(xmin-(xmax-xmin)*0.02,ymax+(ymax-ymin)*0.05,labels[labelindex],fontsize=10,va='bottom',ha='right')
            labelindex +=1
            nphase = phasesize[j]
            phaseweights_str = "Contributions [%]\n"

            weight_sum = 0.0
            for iphase in range(nphase):
                weight_sum += abs(phaseweights[phasecount+iphase])

            for iphase in range(nphase):
                itrace = phaselist[currentindex + iphase*(2 * (j + 2))+1]
                
                phasename = allphaselist[phasecount+iphase]

                weight = abs(phaseweights[phasecount+iphase])/weight_sum*100
                tempstr = f"{phasename}: "
                phaseweights_str = phaseweights_str+tempstr

                tempstr = f"{weight:3.1f}\n"
                phaseweights_str = phaseweights_str+tempstr

                # ax.text(ray_params[-1],phasetime[0],f"{abs(weight):.3f}",ha="right",va="bottom",fontsize=6,bbox={'facecolor':"white",'edgecolor':'white','pad':0})#

            phasecount += nphase
            currentindex += nphase * (2 * (j + 2))

            ax.text(xmin + 0.99 * (xmax - xmin), 
                    ymin + 0.99 * (ymax - ymin), 
                    phaseweights_str, 
                    multialignment="right",
                    fontsize=6, 
                    color='white', 
                    ha='right', 
                    va='top')



        axs = phasefig.subplots(2,3)
        data_stack = self.hk_results['data']
        delta = self.hk_results['delta']
        Vps = np.array(self.hk_results['Vps'])
        Vss = np.array(self.hk_results['Vps'])/np.array(self.hk_results['Ks_optimal'])
        Hs = np.array(self.hk_results['Hs_optimal'])
        ray_params = self.hk_results['rayparams']
        phasetimes = self.hk_results['phasetimes']
        phasesize = self.hk_results['phasesize']
        phaselist = self.hk_results['phaselist']
        Nlayer = int(self.hk_results['Nlayer'])
        allphaselist = self.hk_results['allphasenamelist']
        phaseweights = self.hk_results['phaseweights']
        if len(allphaselist)==len(phaseweights):
            plotphasename=True
        else:
            plotphasename=False
        for j in range(data_stack.shape[0]):
            
            pltdata = data_stack[j]

            irow = j%2
            icol = j//2
            ax = axs[irow,icol]
            # ax2 = figs[irow,icol].add_subplot(gs[1:]
            
            self.plot_RF_rayp(ax,pltdata,ray_params,1/delta,0.0015,0,plotlengths[j])
            ax.set_xlim(0.036,0.084)

            ax.invert_yaxis()
            ax.tick_params(axis="y", labelleft=False)
            
            ##plot stacked RF
            ax_inset = ax.inset_axes([-0.4, 0, 0.3, 1 ],sharey=ax)
            ## do moveout correction
            stacked_data = np.zeros_like(pltdata[0])
            for k,rayp in enumerate(ray_params):
                stacked_data += Moveout_Correation(delta,pltdata[k],Vps,Vss,Hs,rayp,moveoutphases[j])
            stacked_data = stacked_data/len(ray_params)
            ##normalized by max(abs())
            stacked_data = stacked_data/max(abs(stacked_data))
            ts = np.arange(len(stacked_data))*delta
            data_fill = stacked_data.copy()
            data_fill[data_fill>0]=0
            ax_inset.fill_betweenx(ts,data_fill,0,color='red',lw=0.001)
            data_fill = stacked_data.copy()
            data_fill[data_fill<0]=0
            ax_inset.fill_betweenx(ts,data_fill,0,color='blue',lw=0.001)
            #ax_inset.plot(stacked_data,ts,lw=0.5,color='blue')


            ##plot phase fitting results
            phasecount = 0
            currentindex = 0
            text_pre = None

            for ilayer in range(Nlayer):
                nphase = phasesize[ilayer]
                for iphase in range(nphase):
                    itrace = phaselist[currentindex + iphase*(2 * (ilayer + 2))+1]
                    if j==itrace:
                        phasetime = phasetimes[phasecount+iphase]
                        weight = phaseweights[phasecount+iphase]
                        ax.plot(ray_params,phasetime,lw=0.6,color=list_colornames[ilayer],ls='--')
                        # ax.text(ray_params[-1],phasetime[0],f"{abs(weight):.3f}",ha="right",va="bottom",fontsize=6,bbox={'facecolor':"white",'edgecolor':'white','pad':0})#
                        if plotphasename:
                            phasename = allphaselist[phasecount+iphase]
                            # phasename.replace("+")
                            phasename_x = 0.8
                            phasename_y = phasetime[0]
                            phasename_ha = "left"
                            text_now = ax_inset.text(phasename_x,phasename_y,phasename,
                                                     ha=phasename_ha,va="center",fontsize=6,
                                                     bbox={'facecolor':"white",'edgecolor':'white','pad':0})
                            # text_pre = text_now

                phasecount += nphase
                currentindex += nphase * (2 * (ilayer + 2))

            # ax_inset.tick_params(axis="x", labelbottom=False,)
            ax_inset.set_xlim(-1,1)
            ax_inset.set_xticks([-1,0,1])
            ax_inset.set_xticklabels(["-1","0","1"],color='blue',fontsize=8) 
            ax_inset.set_ylim(0,plotlengths[j])
            ax_inset.invert_yaxis()
            xmin,xmax = ax_inset.get_xlim()
            ymin,ymax = ax_inset.get_ylim()
            ax_inset.text(xmin-(xmax-xmin)*0.02,ymax+(ymax-ymin)*0.05,labels[labelindex],fontsize=10,va='bottom',ha='right')
            labelindex += 1
            # ax2.legend(loc='upper left') 
            if irow==1:
                ax.set_xlabel('Ray Param [s/km]',fontsize=8)
                ax_inset.set_xlabel("Normalized\nAmp",fontsize=8)
            else:
                ax.set_xticklabels([])
                ax_inset.set_xticklabels([])

            if icol==0:
                ax_inset.set_ylabel('Time [s]',fontsize=8)
            else:
                ax_inset.set_yticklabels([])

        #plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.52,hspace=0.12)
        plt.savefig(savefile,dpi=dpi,bbox_inches='tight')
    
    @staticmethod
    def plot_RF_rayp(ax,Ss,rays,sampling,scale,t0=None,t1=None,):
        delta = 1.0/sampling
        if rays.shape[0] != Ss.shape[0]:
            return -1
        for i,rayp in enumerate(rays):
            data = Ss[i,:]
            times = np.arange(len(data))*delta
            if t0!=None and t1!=None:
                index = np.logical_and(times>t0, times<t1)
                times = times[index]
                data = data[index]
            data = data/(data.max()-data.min())*scale
            # ax.plot(data+rayp,times,lw=0.1,c='k')
            data_fill = data.copy()
            data_fill[data_fill>0]=0
            ax.fill_betweenx(times,data_fill+rayp,rayp,color='red',lw=0.001,alpha=0.5)
            data_fill = data.copy()
            data_fill[data_fill<0]=0
            ax.fill_betweenx(times,data_fill+rayp,rayp,color='blue',lw=0.001,alpha=0.5)
        ax.grid(axis='y',lw=0.5,ls='--',color='gray',alpha=0.5)

    @staticmethod
    def check_text_overlap(text1, text2, fig):

        fig.canvas.draw()
        

        bbox1 = text1.get_window_extent()
        bbox2 = text2.get_window_extent()

        transform = fig.transFigure.inverted()
        bbox1_fig = bbox1.transformed(transform)
        bbox2_fig = bbox2.transformed(transform)
        
        return bbox1_fig.overlaps(bbox2_fig)
    
    def get_optmodel(self):

        Hs_optimal = self.hk_results['Hs_optimal']
        Ks_optimal = self.hk_results['Ks_optimal']
        Vps = self.hk_results['Vps']

        return [Vps,Ks_optimal,Hs_optimal]


    def get_HkVp_opt(self):
        Vpcs = self.hk_results['Vpc_searching_Vpcs']
        Amps_opt = self.hk_results['Vpc_searching_Amps']
        Hcs_opt = self.hk_results['Vpc_searching_Hcs']
        Kcs_opt = self.hk_results['Vpc_searching_Kcs']
        amps_op = np.array(Amps_opt)
        Kcs_op = np.array(Kcs_opt)
        Hcs_op = np.array(Hcs_opt)
        arg_op = np.argmax(Amps_opt)
        Vpc_op = Vpcs[arg_op]
        Kc_op = Kcs_op[arg_op]
        Hc_op = Hcs_op[arg_op]
        return Vpc_op,Kc_op,Hc_op



    def plot_HkVp(self,savefile:str,
                        dpi:int=900):
        Vpcs = self.hk_results['Vpc_searching_Vpcs']
        Amps_opt = self.hk_results['Vpc_searching_Amps']
        Hcs_opt = self.hk_results['Vpc_searching_Hcs']
        Kcs_opt = self.hk_results['Vpc_searching_Kcs']

        fig,axs = plt.subplots(1,3,sharey=True,figsize=(15/2.54,6/2.54),tight_layout=True)
        amps_op = np.array(Amps_opt)
        Kcs_op = np.array(Kcs_opt)
        Hcs_op = np.array(Hcs_opt)
        arg_op = np.argmax(Amps_opt)
        Vpc_op = Vpcs[arg_op]
        Kc_op = Kcs_op[arg_op]
        Hc_op = Hcs_op[arg_op]

        axs[0].plot(Amps_opt,Vpcs,lw=0.8,color='r')
        indexs = np.logical_and(Vpcs>=6.6,Vpcs<=6.8)
        axs[0].fill_between([Amps_opt.min(),Amps_opt.max()],[6.6,6.6],[6.8,6.8],lw=0.1,fc='lightblue',alpha=0.5)
        axs[0].plot([Amps_opt.min(),Amps_opt.max()],[Vpc_op,Vpc_op],lw=0.5,color='gray')
        axs[0].text(Amps_opt.min(),Vpc_op,f"Vp={Vpc_op:.2f}",va='bottom',ha='left',color='gray')
        axs[0].text(Amps_opt.min(),6.7,f"Vp={6.7:.2f}",va='bottom',ha='left',color='black')
        xmin,xmax = axs[0].get_xlim()
        ymin,ymax = axs[0].get_ylim()
        axs[0].text(xmin + 0.99 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                rf"$dVp = {0.1:.3f} \mathrm{{km/s}}$", 
                fontsize=8, color='black', ha='right', va='top')
        axs[0].text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), "(a)", fontsize=8,ha='left',va='top')
        axs[0].set_ylabel("Vp [km/s]")
        axs[0].set_xlabel("Opt Amplitude")
        # axs[0].set_ylim(6.5,7.0)

        axs[1].plot(Kcs_opt,Vpcs,lw=0.8,color='r')

        axs[1].fill_between([Kcs_opt.min(),Kcs_opt.max()],[6.6,6.6],[6.8,6.8],lw=0.1,fc='lightblue',alpha=0.5)
        Kcs_op_errs = Kcs_op[indexs]
        Kcs_op_errs_min = Kcs_op_errs.min()
        Kcs_op_errs_max = Kcs_op_errs.max()
        axs[1].errorbar(np.array([Kc_op]),np.array([Vpc_op]),xerr=np.array([[Kc_op-Kcs_op_errs_min,Kcs_op_errs_max-Kc_op]]).T,fmt='o',linewidth=2,capsize=6) 

        # axs[1].fill_betweenx([Vpcs.min(),Vpcs.max()],[Kcs_op_errs_min,Kcs_op_errs_min],[Kcs_op_errs_max,Kcs_op_errs_max],lw=0.1,fc='green',alpha=0.5)
        axs[1].plot([Kcs_opt.min(),Kcs_opt.max()],[Vpc_op,Vpc_op],lw=0.5,color='gray')
        xmin,xmax = axs[1].get_xlim()
        ymin,ymax = axs[1].get_ylim()
        Kcs_op_errs_mean = (Kc_op-Kcs_op_errs_min+Kcs_op_errs_max-Kc_op)/2.0
        axs[1].text(xmin + 0.99 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                rf"d$\kappa = - {Kcs_op_errs_mean:.3f}$", 
                fontsize=8, color='black', ha='right', va='top')
        axs[1].plot([Kc_op,Kc_op],[Vpcs.min(),Vpc_op],lw=0.5,color='gray')
        axs[1].text(Kc_op-0.0005,Vpcs.min(),fr"$\kappa={Kc_op:.3f}$",va='bottom',ha='right',color='black',rotation='vertical')
        axs[1].text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), "(b)", fontsize=8,ha='left',va='top')
        # axs[1].scatter(Kc_op,Vpc_op)
        # axs[1].text(Kc_op,Vpc_op,f"Kc_op={Kc_op:.3f}",ha='left',color='blue')
        # axs[1].set_xticks(ticks=np.arange(1.65,1.85+0.01,0.05))
        axs[1].set_xlabel(r"Opt $\kappa$")

        axs[2].plot(Hcs_opt,Vpcs,lw=0.8,color='r')
        axs[2].fill_between([Hcs_opt.min(),Hcs_opt.max()],[6.6,6.6],[6.8,6.8],lw=0.1,fc='lightblue',alpha=0.5)
        Hcs_op_errs = Hcs_op[indexs]
        Hcs_op_errs_min = Hcs_op_errs.min()
        Hcs_op_errs_max = Hcs_op_errs.max()    
        axs[2].errorbar(np.array([Hc_op]),np.array([Vpc_op]),xerr=np.array([[Hc_op-Hcs_op_errs_min,Hcs_op_errs_max-Hc_op]]).T,fmt='o',linewidth=2,capsize=6) 
        # axs[2].fill_betweenx([Vpcs.min(),Vpcs.max()],[Hcs_op_errs_min,Hcs_op_errs_min],[Hcs_op_errs_max,Hcs_op_errs_max],lw=0.1,fc='green',alpha=0.5)
        axs[2].plot([Hcs_opt.min(),Hcs_opt.max()],[Vpc_op,Vpc_op],lw=0.5,color='gray')
        axs[2].plot([Hc_op,Hc_op],[Vpcs.min(),Vpc_op],lw=0.5,color='gray')
        axs[2].text(Hc_op+0.1,Vpcs.min(),f"H={Hc_op:.3f}",va='bottom',ha='left',color='black',rotation='vertical')
        xmin,xmax = axs[2].get_xlim()
        ymin,ymax = axs[2].get_ylim()
        Hcs_op_errs_mean = (Hc_op-Hcs_op_errs_min+Hcs_op_errs_max-Hc_op)/2.0
        axs[2].text(xmin + 0.99 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), 
                rf"$dH = + {Hcs_op_errs_mean:.3f} \mathrm{{km}}$", 
                fontsize=8, color='black', ha='right', va='top')
        axs[2].text(xmin + 0.01 * (xmax - xmin), ymin + 0.99 * (ymax - ymin), "(c)", fontsize=8,ha='left',va='top')
        axs[2].scatter(Hc_op,Vpc_op)
        # axs[2].set_xticks()
        axs[2].set_xlabel(" Opt Thickness [km]")
        plt.savefig(savefile,dpi=dpi,bbox_inches='tight')


