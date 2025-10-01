#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import signal
import copy


def randRange(x1, x2, integer):
    """Generate a random number in range [x1, x2].
    
    Args:
        x1 (float): Lower bound
        x2 (float): Upper bound
        integer (bool): Whether to return an integer
    
    Returns:
        float or int: Random number
    """
    if x1 > x2:
        x1, x2 = x2, x1
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    return int(y) if integer else float(y)

def normWav(x, always):
    """Normalize waveform to have max amplitude of 1.
    
    Args:
        x (np.ndarray): Input waveform
        always (bool): Whether to always normalize
    
    Returns:
        np.ndarray: Normalized waveform
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    max_amp = np.amax(np.abs(x))
    if max_amp == 0:
        return x
    
    if always or max_amp > 1:
        return x / max_amp
    return x



def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
    """Generate coefficients for notch filters.
    
    Args:
        nBands (int): Number of notch filters
        minF, maxF (float): Frequency range
        minBW, maxBW (float): Bandwidth range
        minCoeff, maxCoeff (int): Filter coefficient range
        minG, maxG (float): Gain range
        fs (float): Sampling frequency
    
    Returns:
        np.ndarray: Filter coefficients
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive")
    
    b = np.array([1.0])
    for i in range(nBands):
        # Generate random parameters
        fc = randRange(minF, maxF, False)
        bw = randRange(minBW, maxBW, False)
        c = randRange(minCoeff, maxCoeff, True)
        
        # Ensure odd number of coefficients
        if c % 2 == 0:
            c += 1
            
        # Calculate band edges
        f1 = max(1.0, fc - bw/2)  # Avoid 0 Hz
        f2 = min(fs/2 - 1.0, fc + bw/2)  # Avoid Nyquist
        
        # Generate and convolve filter
        try:
            notch_filter = signal.firwin(c, [float(f1), float(f2)], 
                                        window='hamming', fs=fs)
            b = np.convolve(notch_filter, b)
        except Exception as e:
            print(f"Warning: Failed to generate filter {i+1}/{nBands}: {str(e)}")
            continue
    
    # Apply gain
    G = randRange(minG, maxG, False)
    _, h = signal.freqz(b, 1, fs=fs)
    if np.amax(abs(h)) > 0:
        b = pow(10, G/20) * b / np.amax(abs(h))
    
    return b

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x, b):
    """Apply FIR filtering to the input signal.
    
    Args:
        x (np.ndarray): Input signal
        b (np.ndarray): Filter coefficients
    
    Returns:
        np.ndarray: Filtered signal
    """
    if not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("Input signal and filter coefficients must be numpy arrays")
    
    if len(x) == 0 or len(b) == 0:
        raise ValueError("Input signal and filter coefficients cannot be empty")
    
    try:
        N = b.shape[0] + 1
        xpad = np.pad(x, (0, N), 'constant')
        y = signal.lfilter(b, 1, xpad)
        y = y[int(N/2):int(y.shape[0]-N/2)]
        return y
    except Exception as e:
        raise RuntimeError(f"Error during FIR filtering: {str(e)}")

# Linear and non-linear convolutive noise
def LnL_convolutive_noise(x,N_f,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,minBiasLinNonLin,maxBiasLinNonLin,fs):
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y


# Impulsive signal dependent noise
def ISD_additive_noise(x, P, g_sd):
    """Add impulsive signal-dependent noise.
    
    Args:
        x (np.ndarray): Input signal
        P (float): Percentage of samples to affect
        g_sd (float): Gain factor for noise
    
    Returns:
        np.ndarray: Signal with added noise
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input signal must be a numpy array")
    
    if P < 0 or P > 100:
        raise ValueError("P must be between 0 and 100")
    
    if g_sd < 0:
        raise ValueError("g_sd must be non-negative")
    
    try:
        beta = randRange(0, P, False)
        y = copy.deepcopy(x)
        x_len = x.shape[0]
        n = max(1, int(x_len * (beta/100)))
        
        # Generate noise
        p = np.random.permutation(x_len)[:n]
        f_r = np.multiply(
            2 * np.random.rand(p.shape[0]) - 1,
            2 * np.random.rand(p.shape[0]) - 1
        )
        
        # Apply noise
        r = g_sd * x[p] * f_r
        y[p] = x[p] + r
        
        return normWav(y, False)
    except Exception as e:
        raise RuntimeError(f"Error during ISD noise addition: {str(e)}")


# Stationary signal independent noise

def SSI_additive_noise(x,SNRmin,SNRmax,nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs):
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x
