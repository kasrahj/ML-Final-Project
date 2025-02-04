import pandas as pd
import numpy as np 
import librosa
import pywt
from scipy.signal import butter, sosfilt


class Denoiser:

    def __init__(self):
        pass

    def bandpass_filter(self, signal, sr, lowcut=300, highcut=3400, order=5):
        nyquist = 0.5 * sr
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')
        filtered_signal = sosfilt(sos, signal)
        return filtered_signal
    
    def wavelet_denoise(self, signal, wavelet='db1', level=4):
        coeffs = pywt.wavedec(signal, wavelet, mode='symmetric', level=level)
        # Apply soft thresholding to detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Noise estimate
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
        denoised_signal = pywt.waverec(coeffs, wavelet, mode='symmetric')
        return denoised_signal
    
    def denoise_data(self, signal, sr):
        return self.bandpass_filter(signal, sr)
        return self.wavelet_denoise(signal)
        return self.wavelet_denoise(self.bandpass_filter(signal, sr))