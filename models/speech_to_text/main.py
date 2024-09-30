import argparse
import torch
import os
from transcriber.model import load_model_and_processor
from transcriber.audio import preprocess_audio, transcribe_audio
from transcriber.utils import get_audio_files_from_directory

def transcribe_multiple_files(model, processor, audio_files, target_sr=16000):
    """
    Transcribes multiple audio files.
    
    Parameters:
        model: The Whisper model.
        processor: The processor used for preparing the input features.
        audio_files (list): List of paths to audio files.
        target_sr (int): The target sampling rate for the audio.
    
    Returns:
        results (dict): Dictionary mapping file names to their transcriptions.
    """
    results = {}
    
    for file_path in audio_files:
        print(f"Processing file: {file_path}")
        audio = preprocess_audio(file_path, target_sr=target_sr)
        transcription = transcribe_audio(model, processor, audio, target_sr=target_sr)
        results[file_path] = transcription
        print(f"Transcription for {file_path}: {transcription}")
    
    return results

def main():
    # Argument parser to accept directory or audio files as input
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper.")
    parser.add_argument('input_path', type=str, help="Path to the audio file or directory containing audio files.")
    args = parser.parse_args()
    
    # Load model and processor once
    model, processor = load_model_and_processor()

    # Check if input is a directory or a single file
    input_path = args.input_path
    audio_files = []
    
    if os.path.isdir(input_path):
        # Get all audio files from the directory
        audio_files = get_audio_files_from_directory(input_path)
    elif os.path.isfile(input_path):
        # Single file path provided
        audio_files = [input_path]
    else:
        print(f"Invalid input path: {input_path}")
        return

    # Transcribe all audio files
    transcriptions = transcribe_multiple_files(model, processor, audio_files)

    # Optionally, you can store the transcriptions in a file or print them out
    for file, transcription in transcriptions.items():
        print(f"File: {file}, Transcription: {transcription}")

if __name__ == "__main__":
    main()
