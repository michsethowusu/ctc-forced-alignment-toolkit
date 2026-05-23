import re

import numpy as np
import onnxruntime as ort
import torch
import torchaudio


def load_audio(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to target sample rate, return mono float32 array."""
    waveform, orig_sr = torchaudio.load(file_path)
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze().numpy().astype(np.float32)


def read_tokens(tokens_path: str):
    """Read tokens.txt, return (token2id dict, id2token dict)."""
    token2id = {}
    id2token = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            idx = len(token2id)
            token2id[token] = idx
            id2token[idx] = token
    return token2id, id2token


def text_to_token_ids(text: str, token2id: dict) -> list:
    """Convert text to list of token IDs (character-level)."""
    ids = []
    for ch in text:
        if ch in token2id:
            ids.append(token2id[ch])
        else:
            print(f"Warning: character '{ch}' not in token set, skipping")
    return ids


def get_emissions_from_onnx(model_path: str, audio: np.ndarray, sample_rate: int = 16000):
    """Run ONNX CTC model to get emission probabilities."""
    providers = ['CPUExecutionProvider']  # Change to ['CUDAExecutionProvider'] for GPU
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    audio_input = np.expand_dims(audio, axis=0).astype(np.float32)
    outputs = sess.run(None, {input_name: audio_input})
    logits = outputs[0]
    if logits.ndim == 3:
        logits = logits[0]
    emissions = torch.softmax(torch.tensor(logits), dim=-1)
    return emissions


def forced_align_emissions(emissions: torch.Tensor, target_ids: list, blank_id: int = 0):
    """Perform forced alignment using torchaudio."""
    target = torch.tensor(target_ids, dtype=torch.int32, device=emissions.device)
    with torch.no_grad():
        alignment, scores = torchaudio.functional.forced_align(
            emissions, target, blank=blank_id
        )
    return alignment, scores


def merge_to_words(char_intervals: list, transcript: str) -> list:
    """Merge character intervals into word intervals."""
    words = transcript.split()
    word_intervals = []
    char_idx = 0
    for word in words:
        word_start = None
        word_end = None
        word_chars = list(word)
        for _ in word_chars:
            if char_idx >= len(char_intervals):
                break
            char_info = char_intervals[char_idx]
            if word_start is None:
                word_start = char_info["start"]
            word_end = char_info["end"]
            char_idx += 1
        if word_start is not None:
            word_intervals.append({
                "word": word,
                "start": word_start,
                "end": word_end
            })
        # Skip spaces (they are not in char_intervals)
    return word_intervals


def merge_words_into_sentences(word_intervals: list, transcript: str) -> list:
    """Group word intervals into sentences based on punctuation (.!?) in transcript."""
    # Split transcript into sentences (keeping punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_intervals = []
    word_idx = 0
    for sent in sentences:
        sent_start = None
        sent_end = None
        sent_word_count = len(sent.split())
        for _ in range(sent_word_count):
            if word_idx >= len(word_intervals):
                break
            word = word_intervals[word_idx]
            if sent_start is None:
                sent_start = word["start"]
            sent_end = word["end"]
            word_idx += 1
        if sent_start is not None:
            sentence_intervals.append({
                "sentence": sent,
                "start": sent_start,
                "end": sent_end
            })
    return sentence_intervals


def write_srt(word_intervals: list, srt_path: str):
    """Write word-level SRT subtitles."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, w in enumerate(word_intervals, start=1):
            start_str = _format_srt_time(w["start"])
            end_str = _format_srt_time(w["end"])
            f.write(f"{i}\n{start_str} --> {end_str}\n{w['word']}\n\n")


def write_sentence_srt(sentence_intervals: list, srt_path: str):
    """Write sentence-level SRT subtitles."""
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, sent in enumerate(sentence_intervals, start=1):
            start_str = _format_srt_time(sent["start"])
            end_str = _format_srt_time(sent["end"])
            f.write(f"{i}\n{start_str} --> {end_str}\n{sent['sentence']}\n\n")


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')
