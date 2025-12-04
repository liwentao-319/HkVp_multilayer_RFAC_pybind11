import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

def dereverberation_filter(r0, t0, freqs):

    """
    resonance removal filter
    reference: Yu et al., 2014; Zhang and Olugboji 2021;
    written by Wentao Li
    Date: 2024/06/04
    """
    complex_numbers = -2*1j*np.pi*freqs*t0
    return 1+r0*np.exp(complex_numbers)

def remove_resonance(rdata,dt,r0,t0):
    """
    remove resonance in time series, rdata with sampling interval to dt
    the frequency of resonance is 1/t0, and the attenuation of each peak is r0
    assuming the length of rdata is even
    written by Wentao Li
    Date: 2024/06/04

    """
    #number points in frequency domain
    n = len(rdata)
    if (n%2)!=0:
        rdata = np.insert(rdata,-1,0)
        n += 1
    ##transform from time-domain to frequency domain and get its positive frequency part
    spec_rdata = fft(rdata)
    ##frequency interval
    df = 1/n/dt
    ##calculate the corresponding frequencies
    freqs = df*np.concatenate([np.arange(n//2),-1*np.flip(np.arange(1,n//2+1))])
    ##design dereverberation filter
    filter = dereverberation_filter(r0,t0,freqs)
    spec_rdata_ = spec_rdata*filter

    return ifft(spec_rdata_) 
