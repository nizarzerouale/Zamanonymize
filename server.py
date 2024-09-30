"""Server that will listen for GET and POST requests from the client."""

import base64
import time
from typing import List

import numpy
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from utils_demo import *
from utils_demo import SERVER_DIR

from concrete.ml.deployment import FHEModelServer

# Load the FHE server
FHE_SERVER = FHEModelServer(DEPLOYMENT_DIR)

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route
@app.get("/")
def root():
    """
    Root endpoint of the health prediction API.
    Returns:
        dict: The welcome message.
    """
    return {"message": "Welcome to your encrypted anonymization use-case with FHE!"}


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    files: List[UploadFile] = File(),
):
    """Send the inputs to the server."""

    # Save the files using the above paths
    write_bytes(SERVER_DIR / f"{user_id}_valuation_key", files[0].file.read())
    write_bytes(SERVER_DIR / f"{user_id}_encrypted_input", files[1].file.read())
    write_bytes(SERVER_DIR / f"{user_id}_encrypted_len_input", files[2].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
):
    """Inference in FHE."""

    evaluation_key_path = SERVER_DIR / f"{user_id}_valuation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_encrypted_input"
    encrypted_input_len_path = SERVER_DIR / f"{user_id}_encrypted_len_input"

    # Read the files (Evaluation key + Encrypted symptoms) using the above paths
    with encrypted_input_path.open("rb") as encrypted_output_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file, encrypted_input_len_path.open("rb") as lenght:
        evaluation_key = evaluation_key_file.read()
        encrypted_tokens = encrypted_output_file.read()
        length = int.from_bytes(lenght.read(), "big")

    timing, encrypted_output = [], []
    for i in range(0, len(encrypted_tokens), length):
        enc_x = encrypted_tokens[i : i + length]
        start_time = time.time()
        enc_y = FHE_SERVER.run(enc_x, evaluation_key)
        timing.append(round(time.time() - start_time, 2))
        encrypted_output.append(enc_y)

    # Write the files
    write_bytes(SERVER_DIR / f"{user_id}_encrypted_output", b"".join(encrypted_output))
    write_bytes(
        SERVER_DIR / f"{user_id}_encrypted_output_len", len(encrypted_output[0]).to_bytes(10, "big")
    )

    return JSONResponse(content=numpy.mean(timing))


@app.post("/get_output")
def get_output(user_id: str = Form()):
    """Retrieve the encrypted output from the server."""

    # Path where the encrypted output is saved
    encrypted_output_path = SERVER_DIR / f"{user_id}_encrypted_output"
    encrypted_output_len_path = SERVER_DIR / f"{user_id}_encrypted_output_len"

    # Read the file using the above path
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    # Read the file using the above path
    with encrypted_output_len_path.open("rb") as f:
        length = f.read()

    time.sleep(1)

    # Encode the binary data to a format suitable for JSON serialization
    content = {
        "encrypted_output": base64.b64encode(encrypted_output).decode("utf-8"),
        "length": base64.b64encode(length).decode("utf-8"),
    }

    # Send the encrypted output
    return JSONResponse(content)
