import json
import re
import time
import uuid
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from utils_demo import *

from concrete.ml.common.serialization.loaders import load
from concrete.ml.deployment import FHEModelClient, FHEModelServer

TOLERANCE_PROBA = 0.77

CURRENT_DIR = Path(__file__).parent

DEPLOYMENT_DIR = CURRENT_DIR / "deployment"
KEYS_DIR = DEPLOYMENT_DIR / ".fhe_keys"


class FHEAnonymizer:
    def __init__(self):

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("obi/deid_roberta_i2b2")
        self.embeddings_model = AutoModel.from_pretrained("obi/deid_roberta_i2b2")

        self.punctuation_list = PUNCTUATION_LIST
        self.uuid_map = read_json(MAPPING_UUID_PATH)

        self.client = FHEModelClient(DEPLOYMENT_DIR, key_dir=KEYS_DIR)
        self.server = FHEModelServer(DEPLOYMENT_DIR)

    def generate_key(self):

        clean_directory()

        # Creates the private and evaluation keys on the client side
        self.client.generate_private_and_evaluation_keys()

        # Get the serialized evaluation keys
        self.evaluation_key = self.client.get_serialized_evaluation_keys()
        assert isinstance(self.evaluation_key, bytes)

        evaluation_key_path = KEYS_DIR / "evaluation_key"

        with evaluation_key_path.open("wb") as f:
            f.write(self.evaluation_key)

    def encrypt_query(self, text: str):
        # Pattern to identify words and non-words (including punctuation, spaces, etc.)
        tokens = re.findall(r"(\b[\w\.\/\-@]+\b|[\s,.!?;:'\"-]+)", text)
        encrypted_tokens = []

        for token in tokens:
            if bool(re.match(r"^\s+$", token)):
                continue
            # Directly append non-word tokens or whitespace to processed_tokens

            # Prediction for each word
            emb_x = get_batch_text_representation([token], self.embeddings_model, self.tokenizer)
            encrypted_x = self.client.quantize_encrypt_serialize(emb_x)
            assert isinstance(encrypted_x, bytes)

            encrypted_tokens.append(encrypted_x)

        write_pickle(KEYS_DIR / f"encrypted_quantized_query", encrypted_tokens)

    def run_server(self):

        encrypted_tokens = read_pickle(KEYS_DIR / f"encrypted_quantized_query")

        encrypted_output, timing = [], []
        for enc_x in encrypted_tokens:
            start_time = time.time()
            enc_y = self.server.run(enc_x, self.evaluation_key)
            timing.append((time.time() - start_time) / 60.0)
            encrypted_output.append(enc_y)

        write_pickle(KEYS_DIR / f"encrypted_output", encrypted_output)
        write_pickle(KEYS_DIR / f"encrypted_timing", timing)

        return encrypted_output, timing

    def decrypt_output(self, text):

        encrypted_output = read_pickle(KEYS_DIR / f"encrypted_output")

        tokens = re.findall(r"(\b[\w\.\/\-@]+\b|[\s,.!?;:'\"-]+)", text)
        decrypted_output, identified_words_with_prob = [], []

        i = 0
        for token in tokens:
            # Directly append non-word tokens or whitespace to processed_tokens
            if bool(re.match(r"^\s+$", token)):
                continue
            else:
                encrypted_token = encrypted_output[i]
                prediction_proba = self.client.deserialize_decrypt_dequantize(encrypted_token)
                probability = prediction_proba[0][1]
                i += 1

                if probability >= TOLERANCE_PROBA:
                    identified_words_with_prob.append((token, probability))

                    # Use the existing UUID if available, otherwise generate a new one
                    tmp_uuid = self.uuid_map.get(token, str(uuid.uuid4())[:8])
                    decrypted_output.append(tmp_uuid)
                    self.uuid_map[token] = tmp_uuid
                else:
                    decrypted_output.append(token)

            # Update the UUID map with query.
            with open(MAPPING_UUID_PATH, "w") as file:
                json.dump(self.uuid_map, file)

        write_pickle(KEYS_DIR / f"reconstructed_sentence", " ".join(decrypted_output))
        write_pickle(KEYS_DIR / f"identified_words_with_prob", identified_words_with_prob)


    def run_server_and_decrypt_output(self, text):
        self.run_server()
        self.decrypt_output(text)
