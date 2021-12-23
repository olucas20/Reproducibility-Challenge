#!pip install -U julius
import numpy as np
import torch
import random
from torchaudio.transforms import Vol
from julius.filters import lowpass_filter, highpass_filter
from torchaudio_augmentations import Delay, PitchShift, RandomResizedCrop, Reverb

#B.1 Random Crop:
#	n_samples s{20736, 43740, 59049} for sample rates 8000, 16000, 22050Hz
#	To call random crop, call RandomResizedCrop(s)(audio)
def RandomResize(audio, n_samples=59049):
    return RandomResizedCrop(n_samples)(audio)


#B.2 Polarity Inversion:
#	Multiplies the amplitude by -1
def PolInversion(audio):
	return torch.neg(audio)

#B.3 Additive White Gaussian Noise:
#	White Gaussian noise added to the signal with SNR of 80 decibels
def AddGausNoise(audio, min_snr=0.001, max_snr=1.0):
	#Draw random samples from a normal Gaussian distribution
	std = torch.std(audio)
	noise_std = random.uniform(min_snr * std, max_snr * std)
	noise = np.random.normal(0.0, noise_std, size = audio.shape).astype(np.float32)
	return audio + noise

#B.4 Gain Reduction:
#	Gain of audio reduced at random between -6 and 0 decibels
def GainReduction(audio, gain_min_db=-6, gain_max_db=0):
	#Pick a random gain augmentation
	gain_to_apply = random.uniform(gain_min_db, gain_max_db)
	#Apply gain augmentation
	audio = Vol(gain_to_apply, gain_type = 'db')(audio)

	return audio

#B.5 Frequency Filter:
#	Applies a lowpass or highpass filter determined by a coin flip
#	Highpass: low_freq = 2200Hz, high_freq: high_freq = 4000Hz
#	Lowpass: low_freq = 200Hz, high_freq = 1200Hz
def freq_to_apply(low_freq, high_freq):
	return random.uniform(low_freq, high_freq)

def cut(freq, sample_rate):
	return float(freq)/float(sample_rate)

def FrequencyFilter(audio, sample_rate=22050, lowpass_freq_low=2200, lowpass_freq_high=4000,
					highpass_freq_low=200, highpass_freq_high=1200):
	#Coin flip choice for highpass or lowpass
	coin_flip = random.randint(0,1)

	if coin_flip == 1:	#Apply high pass
		freq = freq_to_apply(highpass_freq_low, highpass_freq_high)
		cut_freq = cut(freq, sample_rate)
		audio = highpass_filter(audio, cutoff=cut_freq)
	else:				#Apply low pass
		freq = freq_to_apply(lowpass_freq_low, lowpass_freq_high)
		cut_freq = cut(freq, sample_rate)
		audio = lowpass_filter(audio, cutoff=cut_freq)

	return audio

#B.6 Delay:
#	Signal delayed with value chosen randomly between 200 and 500 ms in 50 ms interval
#	The delayed signal is added to the original signal with a volume factor of 0.5
def Delay(audio, sample_rate=22050, delay_min=200, delay_max=500, step=50, vol=0.5):
	ms = random.choice(np.arange(delay_min, delay_max, step))

	offset = int(ms * (sample_rate / 1000))
	beginning = torch.zeros(audio.shape[0], offset)
	end = audio[:,:-offset]
	delayed_signal = torch.cat((beginning, end), dim=1)
	delayed_signal = delayed_signal * vol
	audio = (audio + delayed_signal)/2

	return audio

#B.7 Pitch Shift:
#	The pitch of the signal is shifted up or down depending on a drawn number betweenn -5 and 5 semitones
#	Assumes 12-tone equal temperament tuning
def P_Shift(audio, sample_rate = 22050, n_samples=59049):
	return PitchShift(n_samples=n_samples, sample_rate=sample_rate)(audio)

#B.8 Reverb:
#	Applies Schroeder reverbation
#	For Reverb call Reverb(sample_rate)(audio)
def Rvrb(audio, sample_rate=22050):
    return Reverb(sample_rate)(audio)