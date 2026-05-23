# CTC Forced Alignment — omnilingual sherpa-onnx

Aligns an audio file with a given transcript and outputs character/subword, word,
and sentence-level timestamps using the
[csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12](https://huggingface.co/csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12)
CTC model (1 600 languages).

## Install

```bash
pip install torch torchaudio onnxruntime huggingface_hub "numpy<2"
pip install sherpa-onnx          # or use the versioned wheel in requirements_8_.txt
```

## Usage

```bash
# Align a WAV file with a literal transcript, write JSON + word SRT
python align.py audio.wav "Hello world" --srt

# Use a transcript file, write JSON + sentence SRT
python align.py audio.wav transcript.txt --output out.json --sentence-srt

# Use a different model variant (e.g. full-precision)
python align.py audio.wav "text" \
  --model-repo csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12
```

## Output

`alignment.json` contains:

```json
{
  "audio_file": "audio.wav",
  "transcript": "Hello world",
  "duration_sec": 3.12,
  "n_frames": 97,
  "frame_period_sec": 0.032164,
  "sherpa_hypothesis": "HELLO WORLD",
  "token_alignment": [
    {"token": "▁HE",  "start": 0.1, "end": 0.22, "score": -0.03},
    ...
  ],
  "word_alignment": [
    {"word": "Hello", "start": 0.1, "end": 0.38},
    {"word": "world", "start": 0.42, "end": 0.71}
  ],
  "sentence_alignment": [...]
}
```

## Design notes

### Why `sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc()`?

The model uses a custom internal preprocessing pipeline (feature extraction,
language conditioning) built into the sherpa-onnx C++ library.  Feeding raw
audio directly into the ONNX session via a generic `ort.InferenceSession` will
produce wrong results because the feature front-end is bypassed.  We use the
sherpa recogniser for a sanity-check decode pass, then call the ONNX encoder
directly (via `onnxruntime`) *only* to obtain the raw CTC logit matrix for
`torchaudio.functional.forced_align()`.

### Token format

`tokens.txt` uses the format `<token> <integer-id>` (space-separated, **not**
just the token on each line).  Tokens are SentencePiece subword units where
`▁` marks word-initial position (e.g. `▁THE`, `ING`, `ATION`).  Text must be
matched greedily against this vocabulary (upper-cased), not character-by-character.

### Log-probs vs softmax

`torchaudio.functional.forced_align()` expects **log-probabilities**.  We apply
`torch.log_softmax` to the raw logits from the ONNX encoder.
