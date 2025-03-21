
import io
import librosa
import numpy as np


def process_audio_bytes(byte_data):
    if isinstance(byte_data, bytes):
        byte_io = io.BytesIO(byte_data)
    elif isinstance(byte_data, io.BytesIO):
        byte_io = byte_data
    else:
        raise TypeError("Expected bytes or BytesIO object")

    audio_array, sr = librosa.load(byte_io, sr=None)

    return audio_array, sr


def resample_audio_array(array, orig_sr, target_sr):
    if isinstance(array, list):
        array = np.array(array)

    if orig_sr == target_sr:
        return array

    array_resampled = librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr)
    return array_resampled


def split_user_audio(audio_array, sample_rate, target_sr, duration: int):
    audio_array = resample_audio_array(audio_array, sample_rate, target_sr)

    contents = []
    hop = int(target_sr*duration)
    step = 0
    for offset in range(0, len(audio_array), hop):
        offset_l = max(offset - step, 0)
        offset_r = min(offset + hop + step, len(audio_array))
        contents.append(
            {
                "type": "audio",
                "array": audio_array[offset_l:offset_r]
            }
        )
    return contents
