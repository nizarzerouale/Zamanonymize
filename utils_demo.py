import json
import os
import pickle as pkl
import re
import shutil
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


from pathlib import Path

# Core Application URL
SERVER_URL = "http://localhost:8000/"

# Maximum length for user queries
MAX_USER_QUERY_LEN = 128

# Base Directories
CURRENT_DIR = Path(__file__).parent
DEPLOYMENT_DIR = CURRENT_DIR / "deployment"
DATA_PATH = CURRENT_DIR / "files"

# Deployment Directories
CLIENT_DIR = DEPLOYMENT_DIR / "client_dir"
SERVER_DIR = DEPLOYMENT_DIR / "server_dir"
KEYS_DIR = DEPLOYMENT_DIR / ".fhe_keys"

# All Directories
ALL_DIRS = [KEYS_DIR, CLIENT_DIR, SERVER_DIR]

# Model and Data Files
LOGREG_MODEL_PATH = CURRENT_DIR / "models" / "cml_logreg.model"
ORIGINAL_FILE_PATH = DATA_PATH / "original_document.txt"
ANONYMIZED_FILE_PATH = DATA_PATH / "anonymized_document.txt"
MAPPING_UUID_PATH = DATA_PATH / "original_document_uuid_mapping.json"
MAPPING_ANONYMIZED_SENTENCES_PATH = DATA_PATH / "mapping_clear_to_anonymized.pkl"
MAPPING_ENCRYPTED_SENTENCES_PATH = DATA_PATH / "mapping_clear_to_encrypted.pkl"
MAPPING_DOC_EMBEDDING_PATH = DATA_PATH / "mapping_doc_embedding_path.pkl"

PROMPT_PATH = DATA_PATH / "chatgpt_prompt.txt"


# List of example queries for easy access
DEFAULT_QUERIES = {
    "Example Query 1": "What is the amount of the contract between David and Kate?",
    "Example Query 2": "How many people are engaged in the contract?",
    "Example Query 3": "Does Kate have an international bank account?",
}

# Load tokenizer and model
TOKENIZER = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
EMBEDDINGS_MODEL = AutoModel.from_pretrained("obi/deid_roberta_i2b2")

PUNCTUATION_LIST = list(string.punctuation)
PUNCTUATION_LIST.remove("%")
PUNCTUATION_LIST.remove("$")
PUNCTUATION_LIST = "".join(PUNCTUATION_LIST) + '°'


def clean_directory() -> None:
    """Clear direcgtories"""

    print("Cleaning...\n")
    for target_dir in ALL_DIRS:
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)


def get_batch_text_representation(texts, model, tokenizer, batch_size=1):
    """Get mean-pooled representations of given texts in batches."""
    mean_pooled_batch = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=False)
        last_hidden_states = outputs.last_hidden_state
        input_mask_expanded = (
            inputs["attention_mask"].unsqueeze(-1).expand(last_hidden_states.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        mean_pooled = sum_embeddings / sum_mask
        mean_pooled_batch.extend(mean_pooled.cpu().detach().numpy())
    return np.array(mean_pooled_batch)


def is_user_query_valid(user_query: str) -> bool:
    """
    Check if the `user_query` is None and not empty.
    Args:
        user_query (str): The input text to be checked.
    Returns:
        bool: True if the `user_query` is None or empty, False otherwise.
    """
    # If the query is not part of the default queries
    is_default_query = user_query in DEFAULT_QUERIES.values()

    # Check if the query exceeds the length limit
    is_exceeded_max_length = user_query is not None and len(user_query) <= MAX_USER_QUERY_LEN

    return not is_default_query and not is_exceeded_max_length


def compare_texts_ignoring_extra_spaces(original_text, modified_text):
    """Check if the modified_text is identical to the original_text except for additional spaces.

    Args:
        original_text (str): The original text for comparison.
        modified_text (str): The modified text to compare against the original.

    Returns:
        (bool): True if the modified_text is the same as the original_text except for
            additional spaces; False otherwise.
    """
    normalized_original = " ".join(original_text.split())
    normalized_modified = " ".join(modified_text.split())

    return normalized_original == normalized_modified


def is_strict_deletion_only(original_text, modified_text):

    # Define a regex pattern that matches a word character next to a punctuation
    # or a punctuation next to a word character, without a space between them.
    pattern = r"(?<=[\w])(?=[^\w\s])|(?<=[^\w\s])(?=[\w])"

    # Replace instances found by the pattern with a space
    original_text = re.sub(pattern, " ", original_text)
    modified_text = re.sub(pattern, " ", modified_text)

    # Tokenize the texts into words, considering also punctuation
    original_words = Counter(original_text.lower().split())
    modified_words = Counter(modified_text.lower().split())

    base_words = all(item in original_words.keys() for item in modified_words.keys())
    base_count = all(original_words[k] >= v for k, v in modified_words.items())

    return base_words and base_count


def read_txt(file_path):
    """Read text from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_txt(file_path, data):
    """Write text to a file."""
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


def write_pickle(file_path, data):
    """Save data to a pickle file."""
    with open(file_path, "wb") as f:
        pkl.dump(data, f)


def read_pickle(file_name):
    """Load data from a pickle file."""
    with open(file_name, "rb") as file:
        return pkl.load(file)


def read_json(file_name):
    """Load data from a json file."""
    with open(file_name, "r") as file:
        return json.load(file)


def write_json(file_name, data):
    """Save data to a json file."""
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, sort_keys=True)


def write_bytes(path, data):
    """Save binary data."""
    with path.open("wb") as f:
        f.write(data)


def read_bytes(path):
    """Load data from a binary file."""
    with path.open("rb") as f:
        return f.read()
