# Zamanonymize

Zamanonymize is an advanced text anonymization tool that uses Fully Homomorphic Encryption (FHE) and Natural Language Processing (NLP) techniques to detect and anonymize Personally Identifiable Information (PII) in transcribed audio files.

## Features

- Upload and process transcribed audio text files
- Detect PII using a pre-trained Concrete ML model and NLP techniques
- Anonymize detected PII by replacing it with unique identifiers
- Preserve the structure and readability of the original text
- Secure anonymization process using FHE techniques
- User-friendly interface built with Streamlit

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Zamanonymize2.git
   cd Zamanonymize2
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the web interface to:
   - Upload a transcribed audio text file
   - Anonymize the text
   - Review and save the anonymized text and PII mapping

## Project Structure

```
Zamanonymize2/
├── app.py
├── models/
│   ├── cml_logreg.model
│   └── test_ner_model.py
├── utils/
│   ├── anonymization.py
│   ├── data_handling.py
│   └── inference.py
├── transcriptions/
├── data/
├── requirements.txt
└── README.md
```

- `app.py`: Main Streamlit application file
- `models/`: Directory containing the pre-trained Concrete ML model
- `utils/`: Utility functions for anonymization, data handling, and inference
- `transcriptions/`: Directory to store anonymized transcriptions
- `data/`: Directory to store PII mappings and other data files

## How It Works

1. The app loads pre-trained models for embedding generation and PII detection.
2. Users upload a transcribed audio text file through the Streamlit interface.
3. The text is processed token by token:
   - Each token is embedded using a transformer model.
   - The Concrete ML model predicts whether the token contains PII.
   - Additional regex patterns are used to catch common PII formats.
4. Detected PII is replaced with unique identifiers (UUIDs).
5. The anonymized text and PII mapping are presented to the user and can be saved for future reference.

## Contributing

Contributions to Zamanonymize are welcome! Please feel free to submit a Pull Request.