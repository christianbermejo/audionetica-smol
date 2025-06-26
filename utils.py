import numpy as np
import tempfile
import os
from scipy import signal

def normalize_audio(audio_data):
    """Normalize audio to -1.0 to 1.0 range"""
    max_val = np.max(np.abs(audio_data))
    if max_val == 0:
        return audio_data
    return audio_data / max_val

def resample_audio(audio_data, orig_sr, target_sr):
    """Resample audio to target sample rate"""
    return signal.resample(audio_data, int(len(audio_data) * target_sr / orig_sr))

def convert_to_mono(audio_data, channels):
    """Convert multi-channel audio to mono"""
    if channels > 1:
        return np.mean(audio_data.reshape(-1, channels), axis=1)
    return audio_data

def create_temp_wav_file(audio_data, sample_rate=16000):
    """Create a temporary WAV file from audio data"""
    import wave
    import struct
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_filename = temp_file.name
        
    with wave.open(temp_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(sample_rate)
        
        # Convert float to int16 if necessary
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
            
        wf.writeframes(audio_data.tobytes())
        
    return temp_filename
