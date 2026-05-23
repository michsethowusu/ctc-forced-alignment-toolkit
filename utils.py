"""
utils.py — helpers for CTC forced alignment with the omnilingual sherpa-onnx model.
"""
import re
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return (samples_float32, sample_rate).
    Accepts mono 16-bit PCM natively, falls back to torchaudio.
    Returns samples in range [-1, 1].
    """
    path = str(file_path)
    try:
        with wave.open(path) as f:
            if f.getnchannels() == 1 and f.getsampwidth() == 2:
                raw = f.readframes(f.getnframes())
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                sr = f.getframerate()
                if sr != target_sr:
                    samples = _resample_np(samples, sr, target_sr)
                return samples, target_sr
    except Exception:
        pass

    # Fallback: torchaudio
    waveform, orig_sr = torchaudio.load(path)
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy().astype(np.float32), target_sr


def _resample_np(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    t = torch.from_numpy(samples).unsqueeze(0)
    t = torchaudio.functional.resample(t, orig_sr, target_sr)
    return t.squeeze().numpy()


# ---------------------------------------------------------------------------
# Token I/O
# ---------------------------------------------------------------------------

def read_tokens(tokens_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Parse tokens.txt with format: <token> <integer-id>
    """
    token2id: Dict[str, int] = {}
    id2token: Dict[int, str] = {}

    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.rsplit(" ", 1)  # token may contain spaces, so split from right
            if len(parts) != 2:
                continue
            token, id_str = parts
            try:
                idx = int(id_str)
            except ValueError:
                continue
            token2id[token] = idx
            id2token[idx] = token

    return token2id, id2token


# ---------------------------------------------------------------------------
# Text → token IDs (CHARACTER‑LEVEL mapping)
# ---------------------------------------------------------------------------

def text_to_token_ids(text: str, token2id: Dict[str, int]) -> List[int]:
    """
    Convert text to CTC token IDs character by character.
    The model vocabulary is expected to contain single characters
    (lowercase Latin letters, digits, Twi-specific letters like ɛ, ɔ, etc.).

    Spaces are skipped – CTC blank will separate words automatically.
    Any character not found in token2id is skipped with a warning.
    """
    ids = []
    for ch in text:
        if ch == " ":
            continue
        if ch in token2id:
            ids.append(token2id[ch])
        else:
            # Try lowercase/uppercase variants only if needed (optional)
            alt = ch.lower() if ch.isupper() else ch.upper()
            if alt in token2id:
                ids.append(token2id[alt])
            else:
                print(f"Warning: no token for {ch!r}, skipping")
    return ids


# ---------------------------------------------------------------------------
# CTC emissions
# ---------------------------------------------------------------------------

def get_emissions_via_sherpa(
    recognizer,
    samples: np.ndarray,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Use the sherpa-onnx recogniser to decode and then extract raw emissions.
    """
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    hypothesis = stream.result.text
    print(f"Sherpa transcription (sanity check): {hypothesis!r}")

    model_path = _find_model_path(recognizer)
    if not model_path:
        raise RuntimeError("Could not find ONNX model path from recogniser config.")
    return get_emissions_direct(model_path, samples, sample_rate)


def _find_model_path(recognizer) -> str:
    """Extract the ONNX model path from the sherpa recogniser config."""
    try:
        mf = recognizer.config.model_config.offline_model_config.model_files
        for attr in ("model", "nemo_ctc", "wenet_ctc", "tdnn", "zipformer2_ctc"):
            val = getattr(mf, attr, None)
            if val and Path(val).exists():
                return val
    except AttributeError:
        pass
    return ""


def get_emissions_direct(
    model_path: str,
    samples: np.ndarray,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Run the ONNX encoder directly and return log-probabilities (T, vocab)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2

    sess = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

    audio_in = np.expand_dims(samples, axis=0).astype(np.float32)  # (1, N)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: audio_in})

    logits = outputs[0]
    if logits.ndim == 3:
        logits = logits[0]  # (1, T, vocab) -> (T, vocab)

    log_probs = torch.log_softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)
    return log_probs


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def forced_align_emissions(
    log_probs: torch.Tensor,
    target_ids: List[int],
    blank_id: int = 0,   # typically <s> or <blk> — check your tokens.txt, often 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around torchaudio.functional.forced_align().
    """
    target = torch.tensor(target_ids, dtype=torch.int32)
    with torch.no_grad():
        alignment, scores = torchaudio.functional.forced_align(
            log_probs.unsqueeze(0),
            target.unsqueeze(0),
            blank=blank_id,
        )
    return alignment.squeeze(0), scores.squeeze(0)


# ---------------------------------------------------------------------------
# Interval building
# ---------------------------------------------------------------------------

def build_char_intervals(
    alignment: torch.Tensor,
    scores: torch.Tensor,
    id2token: Dict[int, str],
    frame_period: float,
    blank_id: int = 0,
) -> List[dict]:
    """Convert frame-level alignment to token intervals."""
    intervals = []
    prev_id = -1
    cur_start = 0.0

    aln = alignment.tolist()
    scr = scores.tolist()
    n = len(aln)

    for i, token_id in enumerate(aln):
        if token_id == blank_id:
            prev_id = token_id
            continue
        if token_id != prev_id:
            cur_start = i * frame_period
        next_id = aln[i + 1] if i + 1 < n else -1
        if next_id != token_id:
            intervals.append({
                "token": id2token.get(token_id, f"<{token_id}>"),
                "start": round(cur_start, 4),
                "end": round((i + 1) * frame_period, 4),
                "score": round(scr[i], 4),
            })
        prev_id = token_id

    return intervals


def merge_to_words(token_intervals: List[dict], transcript: str) -> List[dict]:
    """Merge token intervals into word intervals using spaces in transcript."""
    if not token_intervals:
        return []
    words = transcript.split()
    # Simple grouping: each token corresponds to one character;
    # we need to group back into words by re-joining characters.
    # However, since our tokenization removes spaces, we've lost the boundaries.
    # Better to use the original transcript word lengths to re-group.
    char_tokens = [t["token"] for t in token_intervals]
    full_str = "".join(char_tokens)
    result = []
    pos = 0
    for word in words:
        word_len = len(word)
        chunk = token_intervals[pos:pos + word_len]
        if chunk:
            result.append({
                "word": word,
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
            })
        pos += word_len
    return result


def merge_words_into_sentences(word_intervals: List[dict], transcript: str) -> List[dict]:
    """Group word intervals into sentences (split on .!?)."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript.strip()) if s.strip()]
    result = []
    word_idx = 0
    for sent in sentences:
        count = len(sent.split())
        chunk = word_intervals[word_idx: word_idx + count]
        if chunk:
            result.append({"sentence": sent, "start": chunk[0]["start"], "end": chunk[-1]["end"]})
        word_idx += count
    return result


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def _fmt_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(word_intervals: List[dict], srt_path) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, w in enumerate(word_intervals, 1):
            f.write(f"{i}\n{_fmt_srt(w['start'])} --> {_fmt_srt(w['end'])}\n{w['word']}\n\n")


def write_sentence_srt(sentence_intervals: List[dict], srt_path) -> None:
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentence_intervals, 1):
            f.write(f"{i}\n{_fmt_srt(s['start'])} --> {_fmt_srt(s['end'])}\n{s['sentence']}\n\n")
