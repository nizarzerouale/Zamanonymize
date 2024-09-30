import os

def get_audio_files_from_directory(directory, extensions=[".wav", ".mp3"]):
    """
    Retrieves all audio files from a specified directory.
    
    Parameters:
        directory (str): The directory to search for audio files.
        extensions (list): List of valid audio file extensions.
    
    Returns:
        audio_files (list): List of paths to audio files.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    return audio_files
