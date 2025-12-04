
from obspy import read
from glob import glob
from obspy.core import Stream
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from math import floor
from scipy import signal 
import os
from scipy.fftpack import fft, ifft, fftshift, ifftshift, next_fast_len
from scipy import signal

"""
The Functions of 'smooth_func' and 'spectral_whitening' are from the software "ACC: Auto-Correlogram Calculation in seismology" created by Weijia Sun

"""


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


def spectral_whitening(tr, smooth=None, filter=None,
                       waterlevel=1e-8, corners=2, zerophase=True):
    """
    Apply spectral whitening to data

    Data is divided by its smoothed (Default: None) amplitude spectrum.

    :param tr: trace to manipulate
    :param smooth: length of smoothing window in Hz
        (default None -> no smoothing)
    :param filter: filter spectrum with bandpass after whitening
        (tuple with min and max frequency)
        (default None -> no filter)
    :param waterlevel: waterlevel relative to mean of spectrum
    :param mask_again: weather to mask array after this operation again and
        set the corresponding data to 0
    :param corners: parameters parsing to filter,
    :param zerophase: parameters parsing to filter

    :return: whitened data
    """

    sr = tr.stats.sampling_rate
    data = np.copy(tr.data)
    # data = _fill_array(data, fill_value=0)
    # mask = np.ma.getmask(data)

    # transform to frequency domain
    nfft = next_fast_len(len(data))
    spec = fft(data, nfft)

    # amplitude spectrum
    spec_ampl = np.abs(spec)

    # normalization
    spec_ampl /= np.max(spec_ampl)
    spec_ampl_raw = np.copy(spec_ampl)

    # smooth
    if smooth:
        smooth = int(smooth * nfft / sr)
        spec_ampl = ifftshift(smooth_func(fftshift(spec_ampl), smooth))
        spec_ampl_smth = np.copy(spec_ampl)

    # save guard against division by 0
    spec_ampl[spec_ampl < waterlevel] = waterlevel

    # make the spectrum have the equivalent amplitude before/after smooth
    if smooth:
        scale = np.max(spec_ampl_raw) / np.max(spec_ampl_smth)
        spec /= spec_ampl * scale
    else:
        spec /= spec_ampl

    # FFT back to time domain
    ret = np.real(ifft(spec, nfft)[:len(data)])
    tr.data = ret

    # filter
    if filter is not None:
        tr.filter(type="bandpass", freqmin=filter[0], freqmax=filter[1],
                  corners=corners, zerophase=zerophase)

    return tr



class AC():
    """
    Discription   : calculate Autocorrelation of one channel of seismic record
    Author        : Wentao Li
    Time          : Dec 12th, 2023
    Version       : 1.0
    
    Modified History:
    """

    def __init__(self, datadir, suffix, tref, pretime, length, f1, f2, tcos, df=None):
        """
        datadir          [str]: path storing data
        suffix           [str]: suffix name of target channel
        tref             [str]: name of sachead to store reference time
        pretime        [float]: left time of time window refer to tref
        length         [float]: time length of time window, [tref+pretime, tref+pretime+length]
        f1,f2          [float]: cut-off frequencies of bandpass filter 
        tcos           [float]: time width of cos taper to suppress strong peak near t=0
        df             [float]: frequency width of spectral whitening 
        
        """
        self.datadir = datadir
        self.suffix = suffix
        self.ACst = Stream()
        self.ACst_stack = Stream()
        self.pretime = pretime
        self.length = length
        self.f1 = f1
        self.f2 = f2
        self.tcos = tcos
        self.df = df
        self.tref = tref

    def cal_ACs(self, savedir=None, savesuffix=None):
        filenames = glob(f"{self.datadir}/*{self.suffix}")
        for filename in filenames:
            st = read(filename)
            tr = st[0]
            ##error check
            if np.any(np.isnan(tr.data)):
                continue
            if np.any(np.isinf(tr.data)):
                continue
            if np.max(tr.data)==0:
                continue

            ###
            if isinstance(self.tref,float):
                tp = self.tref
            elif isinstance(self.tref,str):
                tp = tr.stats.sac[self.tref]
            else:
                print("wrong data type for tref")
                continue
            ray_param = tr.stats.sac['user0']
            # snr = self.cal_snr(tr.data,tp-1,30,tr.stats.sampling_rate)
            # if snr < snrmin:
            #     continue     
            ##cut data from tp+pretime to tp+pretime+length
            t_start = tr.stats.starttime
            tr.trim(t_start+tp+self.pretime,t_start+tp+self.pretime+self.length)
            if len(tr.data)!=self.length*tr.stats.sampling_rate+1:
                print(filename, len(tr.data),self.length*tr.stats.sampling_rate+1)
                continue
            ##preprecessing incudes desample,detrend,
            # tr.interpolate(self.sampling_rate)
            ##preprecessing incudes desample,detrend,
            # tr.interpolate(self.sampling_rate)
            delta = tr.stats.delta
            tr.detrend(type="linear")
            tr.detrend(type="demean")
            ## spetral whitening
            tr1 = spectral_whitening(tr=tr,smooth=self.df)
            ##cal autocorrelation
            npts = len(tr1.data)
            data_auto = np.correlate(tr1.data,tr1.data,mode='full')
            M_cos = int(self.tcos/delta)
            tr1.data = self.cos_taper(data_auto[npts:],M_cos)
            tr1.filter("bandpass",freqmin=self.f1,freqmax=self.f2,zerophase=True,corners=5)
            if np.any(np.isnan(tr.data)):
                continue
            if savedir is not None:
                filename_ = filename.split("/")[-1]
                filename_ = filename_.replace(self.suffix,savesuffix)
                savefile = f"{savedir}/{filename_}"
                tr.write(savefile,format='SAC')
            self.ACst.append(tr)


    @staticmethod
    def cos_taper(data,M):
        cos_win = 0.5-0.5*np.cos(np.pi*np.arange(M)/(M-1))
        data[:M] = data[:M]*cos_win
        return data
    
    @staticmethod
    def cal_snr(data,ref_time,length,sampling):
        ref_pts=floor((ref_time)*sampling)
        first_pts=floor((ref_time-length)*sampling)
        last_pts= floor((ref_time+length)*sampling)
        noisePower=np.average(data[first_pts:ref_pts]**2)
        signalPower=np.average(data[ref_pts:last_pts]**2)
        if noisePower<10e-8:
            noisePower = 10e-8
        snr=10*np.log10(signalPower/noisePower)
        return(snr)
    @staticmethod
    def zero_phase_bandpass(data,fs,low_cutoff,high_cutoff ,order=5):
        sos = signal.butter(order, [low_cutoff / (fs / 2), high_cutoff / (fs / 2)], btype='band',output='sos')
        return signal.sosfiltfilt(sos,data)


