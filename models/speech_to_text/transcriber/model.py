from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import pipeline

def load_model_and_processor(model_name="openai/whisper-base"):
    """
    Loads the Whisper model and processor.
    
    Parameters:
        model_name (str): The model to load. Defaults to 'openai/whisper-base'.
    
    Returns:
        model (WhisperForConditionalGeneration): Loaded Whisper model.
        processor (WhisperProcessor): Loaded processor for the model.
    """

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    
    return model, processor
