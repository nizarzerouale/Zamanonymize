import argparse
import re
import uuid

from transformers import AutoModel, AutoTokenizer
from concrete.ml.common.serialization.loaders import load
from utils_demo import *

def load_models():

    # Load the tokenizer and the embedding model
    try:
        tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
        embeddings_model = AutoModel.from_pretrained("obi/deid_roberta_i2b2")
    except:
        print("Error while loading Roberta")

    # Load the CML trained model
    with open(LOGREG_MODEL_PATH, "r") as model_file:
        cml_ner_model = load(file=model_file)

    return embeddings_model, tokenizer, cml_ner_model


def anonymize_with_cml(text, embeddings_model, tokenizer, cml_ner_model):
    token_pattern = r"(\b[\w\.\/\-@]+\b|[\s,.!?;:'\"-]+|\$\d+(?:\.\d+)?|\€\d+(?:\.\d+)?)"

    tokens = re.findall(token_pattern, text)
    uuid_map = {}
    processed_tokens = []

    for token in tokens:
        if token.strip() and re.match(r"\w+", token):  # If the token is a word
            x = get_batch_text_representation([token], embeddings_model, tokenizer)
            prediction_proba = cml_ner_model.predict_proba(x, fhe="disable")
            probability = prediction_proba[0][1]
            prediction = probability >= 0.77
            if prediction:
                if token not in uuid_map:
                    uuid_map[token] = str(uuid.uuid4())[:8]
                processed_tokens.append(uuid_map[token])
            else:
                processed_tokens.append(token)
        else:
            processed_tokens.append(token)  # Preserve punctuation and spaces as is

    anonymized_text = "".join(processed_tokens)
    return anonymized_text, uuid_map


def anonymize_text(text, verbose=False, save=False):

    # Load models
    if verbose:
        print("Loading models..")
    embeddings_model, tokenizer, cml_ner_model = load_models()

    if verbose:
        print(f"\nText to process:--------------------\n{text}\n--------------------\n")

    # Save the original text to its specified file
    if save:
        write_txt(ORIGINAL_FILE_PATH, text)

    # Anonymize the text
    anonymized_text, uuid_map = anonymize_with_cml(text, embeddings_model, tokenizer, cml_ner_model)

    # Save the anonymized text to its specified file
    if save:
        mapping = {o: (i, a) for i, (o, a) in enumerate(zip(text.split("\n\n"), anonymized_text.split("\n\n")))}
        write_txt(ANONYMIZED_FILE_PATH, anonymized_text)
        write_pickle(MAPPING_SENTENCES_PATH, mapping)

    if verbose:
        print(f"\nAnonymized text:--------------------\n{anonymized_text}\n--------------------\n")

    # Save the UUID mapping to a JSON file
    if save:
        write_json(MAPPING_UUID_PATH, uuid_map)

    if verbose and save:
        print(f"Original text saved to    :{ORIGINAL_FILE_PATH}")
        print(f"Anonymized text saved to  :{ANONYMIZED_FILE_PATH}")
        print(f"UUID mapping saved to     :{MAPPING_UUID_PATH}")
        print(f"Sentence mapping saved to :{MAPPING_SENTENCES_PATH}")

    return anonymized_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize named entities in a text file and save the mapping to a JSON file."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="files/original_document.txt",
        help="The path to the file to be processed.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="This provides additional details about the program's execution.",
    )
    parser.add_argument("--save", type=bool, default=True, help="Save the files.")

    args = parser.parse_args()

    text = read_txt(args.file_path)

    anonymize_text(text, verbose=args.verbose, save=args.save)
