import pyaudio
import numpy as np
import wave
import tempfile
import threading
from gtts import gTTS
import os
import time

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.p = pyaudio.PyAudio()
        self.recording = False
        self.audio_buffer = []
        
    def start_recording(self, duration=5):
        """Record audio from microphone for specified duration"""
        self.recording = True
        self.audio_buffer = []
        
        def record():
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            print(f"Recording for {duration} seconds...")
            
            for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
                if not self.recording:
                    break
                data = stream.read(self.chunk_size)
                self.audio_buffer.append(data)
                
            stream.stop_stream()
            stream.close()
            print("Recording finished")
            self.recording = False
            
        thread = threading.Thread(target=record)
        thread.start()
        return thread
    
    def stop_recording(self):
        """Stop the current recording"""
        self.recording = False
    
    def get_audio_data(self):
        """Get the recorded audio as numpy array"""
        if not self.audio_buffer:
            return None
            
        audio_data = b''.join(self.audio_buffer)
        return np.frombuffer(audio_data, dtype=np.int16)
    
    def save_audio(self, audio_data, filename="output.wav"):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        return filename
    
    def play_audio(self, audio_data):
        """Play audio data through speakers"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        # Save the audio data to the temporary file
        self.save_audio(audio_data, temp_filename)
        
        # Play the audio
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True
        )
        
        with wave.open(temp_filename, 'rb') as wf:
            data = wf.readframes(self.chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(self.chunk_size)
                
        stream.stop_stream()
        stream.close()
        os.remove(temp_filename)
    
    def play_audio_from_tts(self, text, lang='ko'):
        """Use gTTS as fallback if HuggingFace TTS fails"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(temp_filename)
        
        # Convert to WAV and play
        os.system(f"ffplay -autoexit -nodisp {temp_filename} > /dev/null 2>&1")
        time.sleep(0.5)  # Give time for audio to finish
        os.remove(temp_filename)
    
    def cleanup(self):
        """Clean up PyAudio resources"""
        self.p.terminate()
