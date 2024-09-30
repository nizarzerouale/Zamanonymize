import librosa
import torch
def preprocess_audio(file_path, target_sr=16000):
    """
    Loads and resamples audio from the specified file.
    
    Parameters:
        file_path (str): Path to the audio file.
        target_sr (int): Target sampling rate. Defaults to 16000 Hz.
    
    Returns:
        resampled_audio (np.ndarray): Resampled audio data.
    """
    audio_input, sample_rate = librosa.load(file_path, sr=None)  # Keep original sample rate
    resampled_audio = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=target_sr)
    return resampled_audio

def transcribe_audio(model, processor, audio, target_sr=16000):
    """
    Transcribes the given audio using the Whisper model.
    
    Parameters:
        model: The Whisper model.
        processor: The processor used for preparing the input features.
        audio (np.ndarray): The resampled audio data.
        target_sr (int): The target sampling rate for the audio.
    
    Returns:
        transcription (str): The transcribed text from the audio.
    """
    input_features = processor(audio, sampling_rate=target_sr, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
