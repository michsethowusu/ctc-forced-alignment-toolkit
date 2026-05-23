"""
utils.py — helpers for CTC forced alignment with the omnilingual sherpa-onnx model.

Key design decisions (learned from model.py / the reference demo):
  - Audio is loaded via the wave module (16-bit PCM, mono) exactly as the
    reference does in read_wave(), then normalised to [-1, 1].
  - The recogniser is built with
        sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc()
    which handles all internal feature-extraction; we must NOT feed raw
    audio directly into the ONNX session naively.
  - Tokens file format:  "<token> <integer-id>"  (space-separated).
  - Tokens are subword units (e.g. "▁THE", "ING", "ATION") — text must be
    matched against them greedily, not character-by-character.
  - torchaudio.functional.forced_align() expects log-probs (shape B,T,C);
    we log-softmax the raw logits before passing them.
"""

import re
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Audio loading — mirrors read_wave() in the reference model.py
# ---------------------------------------------------------------------------

def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file and return (samples_float32, sample_rate).

    Accepts mono 16-bit PCM natively.  Falls back to torchaudio for other
    formats (stereo, non-16kHz, mp3, flac, etc.).

    Returns
    -------
    samples : np.ndarray  shape (N,)  dtype float32  range [-1, 1]
    sample_rate : int  (always target_sr after resampling)
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
    Parse tokens.txt with format:  <token> <integer-id>

    Example lines:
        <blk> 0
        <sos/eos> 1
        S 3
        ▁THE 5
        ING 14

    Special tokens like '<blk>', '<sos/eos>', '<unk>' are included so that
    id2token is complete for building char intervals.
    """
    token2id: Dict[str, int] = {}
    id2token: Dict[int, str] = {}

    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # Split on the LAST space to handle any token containing spaces
            parts = line.rsplit(" ", 1)
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
# Text → token IDs  (subword greedy matching)
# ---------------------------------------------------------------------------

def text_to_token_ids(text: str, token2id: Dict[str, int]) -> List[int]:
    """
    Convert *text* to a list of token IDs using greedy longest-match against
    the subword vocabulary.

    The tokens use SentencePiece '▁' to mark word-initial subwords.
    Steps:
      1. Upper-case the text (tokens are upper-case in this vocabulary).
      2. At each word boundary, prefer tokens with '▁' prefix.
      3. Greedily scan for the longest matching token.

    Characters/substrings with no matching token are skipped with a warning.
    """
    text_up = text.upper()
    ids: List[int] = []
    i = 0
    n = len(text_up)
    at_word_start = True

    while i < n:
        ch = text_up[i]
        if ch == " ":
            at_word_start = True
            i += 1
            continue

        matched = False
        # Try with ▁ prefix first (at word boundary), then without
        prefixes = ["▁"] if at_word_start else []
        prefixes.append("")

        for prefix in prefixes:
            # Longest-match: try from the max possible length down to 1
            for end in range(min(i + 30, n), i, -1):
                candidate = prefix + text_up[i:end]
                if candidate in token2id:
                    ids.append(token2id[candidate])
                    i = end
                    at_word_start = False
                    matched = True
                    break
            if matched:
                break

        if not matched:
            print(f"Warning: no token for '{text_up[i]}' at position {i}, skipping")
            i += 1
            at_word_start = False

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
    Use the sherpa-onnx recogniser to decode the audio, then re-run the raw
    ONNX encoder to obtain log-probability emissions for forced alignment.

    The recogniser's internal config exposes the ONNX model path, which we
    use to run onnxruntime directly.

    Returns
    -------
    log_probs : torch.Tensor  shape (T, vocab_size)
    """
    # Sanity-check decode via the full sherpa pipeline
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    hypothesis = stream.result.text
    print(f"Sherpa transcription (sanity check): {hypothesis!r}")

    # Locate the ONNX model file from the recogniser config
    model_path = _find_model_path(recognizer)
    if not model_path:
        raise RuntimeError(
            "Could not determine the ONNX model path from the recogniser config. "
            "Pass model_path explicitly to get_emissions_direct() instead."
        )

    return get_emissions_direct(model_path, samples, sample_rate)


def _find_model_path(recognizer) -> str:
    """
    Attempt to extract the ONNX model file path from the recogniser config.
    The sherpa-onnx config tree varies by model family; we try common paths.
    """
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
    """
    Run the ONNX encoder directly via onnxruntime and return log-probs.

    The omnilingual model expects raw float32 audio as its single input
    (shape 1 x N_samples).  This matches the pattern used in the reference
    decode path before sherpa's feature-extraction front-end was added.

    Returns
    -------
    log_probs : torch.Tensor  shape (T, vocab_size)
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 2
    opts.intra_op_num_threads = 2

    sess = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )

    audio_in = np.expand_dims(samples, axis=0).astype(np.float32)  # (1, N)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: audio_in})

    logits = outputs[0]
    if logits.ndim == 3:
        logits = logits[0]  # (1, T, vocab) → (T, vocab)

    log_probs = torch.log_softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    )
    return log_probs


# ---------------------------------------------------------------------------
# Forced alignment
# ---------------------------------------------------------------------------

def forced_align_emissions(
    log_probs: torch.Tensor,
    target_ids: List[int],
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around torchaudio.functional.forced_align().

    Parameters
    ----------
    log_probs  : (T, vocab)  log-probabilities
    target_ids : list[int]   subword token IDs to align
    blank_id   : CTC blank index (0 for this model)

    Returns
    -------
    alignment : (T,) int tensor — token id per frame (blank between tokens)
    scores    : (T,) float tensor — per-frame alignment scores
    """
    target = torch.tensor(target_ids, dtype=torch.int32)
    # forced_align expects (B, T, C) and (B, S)
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
    """
    Convert frame-level alignment to subword-token intervals.

    Each entry: {"token": str, "start": float, "end": float, "score": float}
    """
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
    """
    Merge subword-token intervals into word intervals.

    Tokens prefixed with '▁' mark the start of a new word.
    """
    if not token_intervals:
        return []

    orig_words = transcript.split()
    word_groups: List[List[dict]] = []
    current: List[dict] = []

    for tok in token_intervals:
        if tok["token"].startswith("▁") and current:
            word_groups.append(current)
            current = [tok]
        else:
            current.append(tok)
    if current:
        word_groups.append(current)

    result = []
    for idx, group in enumerate(word_groups):
        word_text = orig_words[idx] if idx < len(orig_words) else (
            "".join(t["token"].lstrip("▁") for t in group)
        )
        result.append({
            "word": word_text,
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })
    return result


def merge_words_into_sentences(word_intervals: List[dict], transcript: str) -> List[dict]:
    """Group word intervals into sentences delimited by .!? punctuation."""
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
