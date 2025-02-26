
import io
import wave
import numpy as np


def process_audio_bytes(byte_io):
    byte_io = io.BytesIO(byte_io)
    with wave.open(byte_io, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        audio_frames = wav_file.readframes(n_frames)
    
    audio_array = np.frombuffer(audio_frames, dtype=np.int16)
    
    if n_channels > 1:
        audio_array = audio_array.reshape(-1, n_channels)
    
    return audio_array, framerate
