#  # -------------- Gramian Angular Field (GAF)------------------
#  # https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3

#  # Math
# import math
# import numpy as np
# import matplotlib.pyplot as plt

#  # Tools
# def tabulate(x, y, f):
#     """Return a table of f(x, y). Useful for the Gram-like operations."""
#     return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

# def cos_sum(a, b):
#     """To work with tabulate."""
#     return(math.cos(a+b))


# # support = np.array([1,2,3,5,1])
# # support = np.arange(0, 150)
# support = np.random.randint(100, size=(150))
# serie = np.cos(support)
   
# """Compute the Gramian Angular Field of an image"""
# # Min-Max scaling
# min_ = np.amin(serie)
# max_ = np.amax(serie)
# scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

# # Floating point inaccuracy!
# scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
# scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

# # Polar encoding
# phi = np.arccos(scaled_serie)
# # Note! The computation of r is not necessary
# r = np.linspace(0, 1, len(scaled_serie))

# # GAF Computation (every term of the matrix)
# gaf = tabulate(phi, phi, cos_sum)

# # Show the image for the first time series
# # plt.figure(figsize=(5, 5))
# plt.imshow(gaf, cmap='rainbow', origin='lower')
# # plt.imshow(gaf, origin='lower')
# # plt.title('Gramian Transition Field', fontsize=18)
# # plt.colorbar(fraction=0.0457, pad=0.04)
# plt.tight_layout()
# plt.axis('off')
# plt.show()


# #--------------- Markov Transition Field (MTF) ----------------

# import numpy as np
# import matplotlib.pyplot as plt
# from pyts.image import MarkovTransitionField
# from pyts.datasets import load_gunpoint

# #X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
# X, _, _, _ = load_gunpoint(return_X_y=True)

# #X = X_train

# # MTF transformation
# mtf = MarkovTransitionField(image_size=50, n_bins =10, strategy='quantile')
# X_mtf = mtf.fit_transform(X)

# # Show the image for the first time series
# plt.figure(figsize=(5, 5))
# plt.imshow(X_mtf[2], cmap='rainbow', origin='lower')
# plt.title('Markov Transition Field', fontsize=18)
# plt.colorbar(fraction=0.0457, pad=0.04)
# plt.tight_layout()
# plt.show()



# #plt.plot(X[0])
# #plt.show()



# # --------------SPECTOGRAM------------------------
# # https://wiki.aalto.fi/display/ITSP/Spectrogram+and+the+STFT

# import numpy as np
# from scipy import signal
# from scipy.fft import fftshift
# import matplotlib.pyplot as plt

# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500*np.cos(2*np.pi*0.25*time)
# carrier = amp * np.sin(2*np.pi*3e3*time + mod)
# # noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# # noise *= np.exp(-time/5)
# noise = 0
# x = carrier + noise

# # plt.plot(time, x)

# f, t, Sxx = signal.spectrogram(x, fs)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# # plt.axis('off')
# plt.show()

# # plt.plot(Sxx)
# # plt.savefig('foo.png')
# # plt.show()


# # plt.figure(figsize=(5, 4))
# # plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
# # # plt.pcolormesh(times, freqs, spectrogram, shading='gouraud')
# # # plt.title('Spectrogram')
# # # plt.ylabel('Frequency band')
# # # plt.xlabel('Time window')
# # plt.tight_layout()

# nfft = 1024
# powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(x,NFFT=nfft,noverlap = nfft/2,Fs=1)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()


##-------------------------SCALOGRAM---------------------------
## https://towardsdatascience.com/multiple-time-series-classification-by-using-continuous-wavelet-transformation-d29df97c0442
import pywt
import matplotlib.pyplot as plt
import numpy as np
x = [3,7,1,1,-2,5,4,6]
cA, cD = pywt.dwt(x, 'haar')   #Single level wavelet transformation.... there is also called a multi level

print(cA)   #Approximated COff
print(cD)   #Detailed COff

scales = np.arange(1, 33) # range of scales
wavelet = 'mexh'

coeffs, freqs = pywt.cwt(x, scales, wavelet = wavelet)
plt.imshow(coeffs, cmap = 'coolwarm', aspect = 'auto')
plt.axis('on')
plt.ylabel('Scale')
plt.xlabel('Time')
plt.show()