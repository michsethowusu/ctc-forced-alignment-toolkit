# CTC Forced Alignment Toolkit with Omnilingual ASR

A production‑ready forced alignment tool that uses the **Omnilingual ASR 300M CTC model** to align audio with transcripts at **character, word, and sentence** levels.  
Supports **1600+ languages** and runs efficiently on CPU (with optional GPU support).

---

## ✨ Features

- 🎯 **Multiple alignment granularities** – Characters, words, **and sentences** (no extra effort).
- 🌍 **1600+ languages** – The model covers virtually any language.
- ⚡ **Fast** – Optimized ONNX runtime, INT8 quantization.
- 📦 **Easy to use** – Single command, JSON + SRT output.
- 🔧 **Reproducible** – Exact Python 3.10 + specific wheel versions (no dependency hell).

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ctc-forced-alignment-toolkit.git
cd ctc-forced-alignment-toolkit
```

### 2. Create a Python 3.10 virtual environment (recommended)
```bash
python3.10 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> The `requirements.txt` pins exact wheels for CPU (Torch 1.13.1, k2, sherpa_onnx, etc.)  
> All wheels are fetched from PyTorch and Hugging Face.

### 4. (Optional) GPU support
If you have an NVIDIA GPU, change the provider in `utils.py`:
```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 💡 Usage

### Basic command
```bash
python align.py <audio.wav> <transcript.txt> --output out.json
```

Or provide the transcript as a string:
```bash
python align.py audio.wav "Hello world, this is a test." --output out.json
```

### Options
| Argument | Description |
|----------|-------------|
| `audio` | Path to WAV file (16 kHz mono recommended, but any sample rate works – auto‑resampled). |
| `transcript` | Either a text file or a quoted string containing the transcript. |
| `--output`, `-o` | Output JSON file (default: `alignment.json`). |
| `--srt` | Generate word‑level SRT subtitles. |
| `--sentence-srt` | Generate sentence‑level SRT subtitles (one subtitle per sentence). |
| `--sample-rate` | Target sample rate (default: 16000). |
| `--blank-id` | CTC blank token ID (default: 0 – correct for this model). |

### Example
```bash
python align.py example/audio.wav example/transcript.txt --output result.json --sentence-srt
```

---

## 📄 Output Format

The output JSON contains three alignment levels:

```json
{
  "audio_file": "example/audio.wav",
  "transcript": "Hello world. How are you?",
  "frame_period_sec": 0.08,
  "char_alignment": [
    {"char": "H", "start": 0.12, "end": 0.24, "score": 0.98},
    {"char": "e", "start": 0.24, "end": 0.36, "score": 0.97},
    ...
  ],
  "word_alignment": [
    {"word": "Hello", "start": 0.12, "end": 0.68},
    {"word": "world.", "start": 0.69, "end": 0.98},
    {"word": "How", "start": 0.99, "end": 1.15},
    {"word": "are", "start": 1.16, "end": 1.34},
    {"word": "you?", "start": 1.35, "end": 1.60}
  ],
  "sentence_alignment": [
    {"sentence": "Hello world.", "start": 0.12, "end": 0.98},
    {"sentence": "How are you?", "start": 0.99, "end": 1.60}
  ]
}
```

- **`char_alignment`** – Each character with start/end time and confidence score.  
- **`word_alignment`** – Words merged from characters (spaces in transcript define word boundaries).  
- **`sentence_alignment`** – Sentences detected by punctuation (`.`, `!`, `?`) in the transcript. No extra user input required.

If you use `--srt`, you get a word‑level subtitle file.  
With `--sentence-srt`, you get a sentence‑level subtitle file.

---

## 🧠 How It Works

1. **Model** – `csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12`  
   - Zipformer encoder with CTC head, INT8 quantized for speed.  
   - Character‑based tokenizer (`tokens.txt`).
2. **Alignment** – `torchaudio.functional.forced_align` runs the Viterbi algorithm on the CTC emissions.
3. **Sentence splitting** – Uses the original transcript’s punctuation (`. ! ?`) to group words into sentences.

---

## 📦 Requirements

- Python 3.10 (exact version because of pre‑built wheels)
- Dependencies are listed in `requirements.txt` (pinned for reproducibility)

---

## 🤝 Acknowledgments

- Model and ONNX runtime integration by [csukuangfj](https://github.com/csukuangfj) (Sherpa ONNX project).
- CTC forced alignment uses PyTorch’s `torchaudio` implementation.

---

## 📜 License

Apache 2.0 – same as the original Omnilingual model.

---

## ❓ FAQ

**Q: Does the model support my language?**  
A: The model was trained on 1600+ languages. If your transcript is in a language with a Latin, Cyrillic, Arabic, or CJK script, it will almost certainly work.

**Q: What if my transcript has no punctuation?**  
A: Then `sentence_alignment` will contain a single sentence covering the whole transcript.

**Q: Can I align very long audio (hours)?**  
A: The model can process up to ~40 seconds per chunk. For longer files, you should split the audio and transcript into chunks (the toolkit does not do this automatically yet – contributions welcome).

**Q: How accurate are the timestamps?**  
A: CTC forced alignment is frame‑level (typically 80ms per frame). The timestamps are accurate within a few hundred milliseconds.

---

## 🛠️ Development

Run tests:
```bash
python -m unittest discover tests
```

Contributions are welcome! Please open an issue or pull request.
```

