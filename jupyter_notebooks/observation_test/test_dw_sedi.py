from acc.core import spectral_whitening,remove_response
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from scipy.signal import correlate
from scipy.fftpack import fft,ifft
from obspy import read
from scipy import signal 
from scipy.fftpack import fft, ifft, fftshift, ifftshift, next_fast_len
import os

def smooth_func(x, window_len=None, window='flat', method='zeros'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.

    :param x: the input signal (numpy array)
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects\n
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))\n
        'reflect': pad reflected signal on both ends (same)\n
        'clip': pad signal on both ends with the last valid value (same)\n
        None: no handling of border effects
        (len(smooth(x)) = len(x) - len(window_len) + 1)
    """
    if window_len is None:
        return x
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    elif method == 'clip':
        s = np.r_[x[0] * np.ones((window_len - 1) // 2), x,
                  x[-1] * np.ones(window_len // 2)]
    else:
        s = x

    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)

    return signal.fftconvolve(w / w.sum(), s, mode='valid')

def spectral_whitening(spec,smooth,sr):
    waterlevel=1e-5
    nfft = len(spec)
    smooth = int(smooth * nfft / sr)
    spec_ampl = np.abs(spec)
    spec_ampl /= np.max(spec_ampl)
    spec_ampl_raw = np.copy(spec_ampl)
    spec_ampl = ifftshift(smooth_func(fftshift(spec_ampl), smooth))
    spec_ampl_smth = np.copy(spec_ampl)
    spec_ampl[spec_ampl < waterlevel] = waterlevel
    scale = np.max(spec_ampl_raw) / np.max(spec_ampl_smth)
    spec /= spec_ampl * scale
    return spec

def cos_taper(data,M):
    cos_win = 0.5-0.5*np.cos(np.pi*np.arange(M)/(M-1))
    data[:M] = data[:M]*cos_win
    return data

def zero_phase_bandpass(data,fs,low_cutoff,high_cutoff ,order=5):
    b,a = signal.butter(order, [low_cutoff / (fs / 2), high_cutoff / (fs / 2)], btype='band')
    return signal.filtfilt(b,a,data)

depth=5
datadir="Varyp_sediment_crust_mantle_synthetics_c35_dt0.01"
stationname=f"sediments_depth{depth}"
sampling_rate = 100
length=100
pltlength=4
cos_t = 0.4
f1 = 0.2
f2 = 4
channel='Zn'
dws = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,2,3,4,None]
autocorr_spectra_powers = []
autocorrs = []
savedir = f"test_dw_sediment/sediment_depth{depth}"
if not os.path.isdir(savedir):
    os.makedirs(savedir)

for dw in dws:
    files = glob(f"{datadir}/{stationname}/*.{channel}.SAC")
    datas_fft = []
    data_auto = []
    for file in files:
        st = read(file)
        tr = st[0]
        tr.detrend("linear")
        tr.detrend("demean")
        t_start = tr.stats.starttime
        tr.trim(t_start+5-5,t_start+5-5+length)
        if len(tr.data)!=length*tr.stats.sampling_rate+1:
            continue
        #make sure the same sampling rate 
        if tr.stats.sampling_rate!=sampling_rate:
            tr.interpolate(sampling_rate)
        #spectral whitening for each trace
        tr.taper(max_percentage=0.01)
        datas_fft.append(fft(tr.data))
    ##average the whitened spetra
    data_fft_mean = np.mean(np.stack(datas_fft,axis=1),axis=1)
    delta = tr.stats.delta
    npts = len(data_fft_mean)
    df = 1.0/delta/npts

    ##spectral whitening on the average spectra
    if dw is not None:
        data_fft_mean = spectral_whitening(data_fft_mean,dw,1/delta)
    data_autocorr_spetra = data_fft_mean*np.conj(data_fft_mean)
    fs = np.arange(npts//2)*df
    autocorr_spectra_powers.append((fs,data_autocorr_spetra[:npts//2]))

    ##cal ifft of auto
    data_autocorr = ifft(data_autocorr_spetra)

    ###surpress the big amp at zero time
    M_cos = int(cos_t/delta)
    data_autocorr = cos_taper(data_autocorr,M_cos)
    data_autocorr = zero_phase_bandpass(data_autocorr,1/delta,f1,f2,order=5)

    npts2 = int(pltlength/delta)
    data_autocorr = data_autocorr[:npts2]
    times = np.arange(len(data_autocorr))*delta
    autocorrs.append((times,data_autocorr))

cm =1/2.54
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212,sharex=ax1)

for i,dw in enumerate(dws):
    #plot spectral power 
    fs,spectral_power = autocorr_spectra_powers[i]
    index = fs<5
    fs = fs[index]
    spectral_power = spectral_power[index]
    spectral_power = spectral_power/spectral_power.max()
    ax1.plot(spectral_power+i,fs,lw=0.5,c='k')
    ax1.fill_betweenx(fs,spectral_power+i,i,color='gray',lw=0.001)

    ##plot autocorrelation
    times, autocorr = autocorrs[i]
    autocorr = autocorr/np.abs(autocorr.min())*0.9
    ax2.plot(autocorr+i,times,lw=0.5,c='k')
    autocorr1 = autocorr.copy()
    autocorr1[autocorr1>0]=0
    ax2.fill_betweenx(times,autocorr1+i,i,color='gray',lw=0.001)

ax1.set_xticks(np.arange(len(dws)))
ax1.set_xticklabels(dws)
ax1.set_xlabel('dw [Hz] ')
ax1.set_ylabel('Freqs [Hz]')
ax1.invert_yaxis()


ax2.set_xticks(np.arange(len(dws)))
ax2.set_xticklabels(dws)
ax2.set_xlabel('dw [Hz]')
ax2.set_ylabel('Time [s]')
ax2.invert_yaxis()

plt.savefig(f"{savedir}/{stationname}_f1{f1}_f2{f2}_{channel}.png",dpi=900)
plt.show()







