import sys
from scipy.io import wavfile
import numpy as np
import math
from matplotlib import pyplot as plt
import pyaudio
import wave
import time
from scipy.fftpack import dct

from pydub import AudioSegment

def loadfile(filename, data_list):
	samplerate, y = wavfile.read(filename)
	##Taking only one channel.
	y = y[:,0]
	print y.shape

	l1 = []
	for a in data_list:
		temp1 = math.floor(samplerate*float(a[0]))
		temp2 = math.ceil(samplerate*float(a[1]))
		temp = [int(temp1), int(temp2), a[2]]
		l1.append(temp)
	l1 = np.array(l1)

	temp = l1[0]
	#print y[int(temp[0]):int(temp[1])]
	#print "##########"
	newWavFileAsList = []
	for i in range(0, len(l1)):
		elem = l1[i]
		startRead = int(elem[0])
		endRead = int(elem[1])
		if startRead >= y.shape[0]:
			startRead = y.shape[0]-1
		if endRead >= y.shape[0]:
			endRead = y.shape[0]-1
		a = np.asarray(y[startRead:endRead])
		newWavFileAsList.append([elem[2], a])
	#temp = newWavFileAsList[9]
	#wavfile.write("otest9.wav", fs, np.asarray(temp[1]))
	#print newWavFileAsList[0]
	return newWavFileAsList, samplerate

def transcript(filename):
	c = open(filename,'read')
	lines = c.readlines()
	lines = map(lambda s: s.strip(), lines)
	list = []
	for i in range(1,len(lines)):
		x = lines[i]
		l = x.split(' ')
		if l[0]!='':
			if l[3]!="[MISC]" and l[3]!="(Emotional":
				temp = [l[0], l[1], l[3].split(',')[0]]
				list.append(temp)
	return list

def plotFile(data):
	#data1 = data[:,0]
	plt.plot(data)
	plt.ylabel("Amplitude")
	plt.xlabel("Time")
	plt.show()
	return

def pre_emph(final, samplerate):
	emphasized_signal_final = []
	pre_emphasis = 0.97
	#y(t) = x(t) - a*x(t-1)
	for i in range(0, len(final)):
		temp = final[i]
		tag = temp[0]
		signal = temp[1]
		emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
		#print "here ", emphasized_signal
		emphasized_signal_final.append([tag, emphasized_signal])
	return emphasized_signal_final

def frame_signal(final, samplerate):
	framesize = 0.025
	framestride = 0.01
	final_frames = []
	NFFT = 512
	for temp in final:
		tag = temp[0]
		signal = temp[1]
		framelength = int(math.floor(framesize * samplerate))
		framestep = int(math.floor(framestride * samplerate))
		signallength = len(signal)
		num_frames = math.ceil(np.ceil(float(signallength - framelength) / framestep))

		pad_signallength = num_frames * framestep + framelength
		z = np.zeros((pad_signallength - signallength))
		pad_signal = np.append(signal, z)
		indices = np.tile(np.arange(0, framelength), (num_frames, 1)) + np.tile(np.arange(0, num_frames * framestep, framestep), (framelength, 1)).T
		frames = pad_signal[indices.astype(np.int32, copy=False)]
		frames *= np.hamming(framelength)
		mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
		pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
		#print "frames shape", pow_frames.shape
		final_frames.append([tag, pow_frames])
	return final_frames

def mel_filterbank(pow_frames, samplerate):
	nfilt = 40
	NFFT = 512
	
	low_melf = 0
	high_melf = (2595 * np.log10(1 + (samplerate / 2) / 700)) ##Convert hz to mel

	mel_points = np.linspace(low_melf, high_melf, nfilt + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

	bin = np.floor((NFFT + 1) * hz_points / samplerate)
	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

	for m in range(1, nfilt + 1):
		f_m_minus = int(bin[m - 1])   # left
		f_m = int(bin[m])             # center
		f_m_plus = int(bin[m + 1])    # right

	for k in range(f_m_minus, f_m):
		fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
	for k in range(f_m, f_m_plus):
		fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

	#print fbank.shape
	f_bank = []
	for temp in pow_frames:
		filter_banks = np.dot(np.asarray(temp[1]), np.asarray(fbank.T))
		filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
		filter_banks = 20 * np.log10(filter_banks)
		f_bank.append([temp[0], filter_banks])
	return f_bank

def decorrelate_feat(fbank, samplerate):
	print "enter"
	num_ceps = 13
	cep_lifter = 22
	n = 512
	mfcc_list = []
	for temp in fbank:
		tag = temp[0]
		filter_banks = temp[1]
		mfcc = dct(filter_banks, type=2, axis=1, norm="ortho")[:, 1 : (num_ceps + 1)]
		
		(nframes, ncoeff) = mfcc.shape
		n = np.arange(ncoeff)
		lift = 1 + (22 / 2) * np.sin(np.pi * n / cep_lifter)
		mfcc *= lift
		mfcc_list.append([tag, mfcc])
	return mfcc_list

if __name__=="__main__":
	print sys.argv[1]
	#Extracting data in the form [start_time, end_time, emotion_tag].
	data_list= transcript(sys.argv[2])
	#Plotting the wav file.
	#plotFile(sys.argv[1])
	#Extracting data according to [start_time, end_time, emotion_tag] from sound.wav file.
	final, samplerate = loadfile(sys.argv[1], data_list)
	#Pre_emphasis of the signal
	final_emp = pre_emph(final, samplerate)
	#Framing the signal and hammington window applied
	pow_frames = frame_signal(final_emp, samplerate)

	fbank = mel_filterbank(pow_frames, samplerate) 
	print "before"
	mfcc = decorrelate_feat(fbank, samplerate)
	mfcc_list = []
	for elem in mfcc:
		tag = elem[0]
		mfcc = elem[1]
		mfcc = map(lambda x:sum(x)/float(len(x)), zip(*mfcc))
		mfcc_list.append([tag, mfcc])
	print mfcc_list[0]
	thefile = open('test.csv', 'w')
	for item in mfcc_list:
		thefile.write("%s\n" % item)
	#Filter banks
