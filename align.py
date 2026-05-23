#!/usr/bin/env python3
"""
CTC Forced Alignment Tool using Omnilingual ASR 300M model.
Aligns an audio file with a given transcript and outputs character, word,
and sentence timestamps.
"""

import argparse
import json
import sys
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from sherpa_onnx import OfflineRecognizer

from utils import (
    load_audio,
    read_tokens,
    text_to_token_ids,
    get_emissions_from_onnx,
    forced_align_emissions,
    merge_to_words,
    merge_words_into_sentences,
    write_srt,
    write_sentence_srt,
)


MODEL_REPO = "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
TOKENS_FILE = "tokens.txt"
MODEL_FILE = "model.int8.onnx"


def main():
    parser = argparse.ArgumentParser(description="Forced alignment with CTC model")
    parser.add_argument("audio", type=str, help="Path to input audio file (WAV)")
    parser.add_argument("transcript", type=str, help="Transcript text or file containing transcript")
    parser.add_argument("--output", "-o", type=str, default="alignment.json",
                        help="Output JSON file (default: alignment.json)")
    parser.add_argument("--srt", action="store_true", help="Generate word-level SRT subtitles")
    parser.add_argument("--sentence-srt", action="store_true", help="Generate sentence-level SRT subtitles")
    parser.add_argument("--language", "-l", type=str, default=None,
                        help="Optional language hint (e.g., 'en', 'zh')")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--blank-id", type=int, default=0, help="Blank token ID (default: 0)")
    args = parser.parse_args()

    # 1. Load transcript
    if Path(args.transcript).is_file():
        with open(args.transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()
    else:
        transcript = args.transcript.strip()

    if not transcript:
        print("Error: Empty transcript", file=sys.stderr)
        sys.exit(1)

    # 2. Download model & tokens
    print("Downloading/loading model from Hugging Face...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    tokens_path = hf_hub_download(repo_id=MODEL_REPO, filename=TOKENS_FILE)

    # 3. Load token mapping
    token2id, id2token = read_tokens(tokens_path)
    print(f"Loaded {len(token2id)} tokens")

    # 4. Convert transcript to token IDs (character-level)
    target_ids = text_to_token_ids(transcript, token2id)
    if not target_ids:
        print("Error: No valid tokens in transcript", file=sys.stderr)
        sys.exit(1)

    # 5. Load audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio, target_sr=args.sample_rate)

    # 6. Run CTC model to get emissions
    print("Computing CTC emissions...")
    emissions = get_emissions_from_onnx(model_path, audio, sample_rate=args.sample_rate)

    # 7. Perform forced alignment
    print("Aligning...")
    alignment, scores = forced_align_emissions(emissions, target_ids, blank_id=args.blank_id)

    # 8. Frame period (time per emission frame)
    frame_period = audio.shape[0] / args.sample_rate / emissions.shape[0]
    print(f"Frame period: {frame_period:.3f} seconds")

    # 9. Build character-level intervals
    char_intervals = []
    current_start = 0.0
    prev_id = None
    for i, token_id in enumerate(alignment):
        if token_id != prev_id and token_id != args.blank_id:
            current_start = i * frame_period
        if i + 1 < len(alignment) and alignment[i+1] != token_id:
            if token_id != args.blank_id:
                end = (i + 1) * frame_period
                char_intervals.append({
                    "char": id2token[token_id],
                    "start": current_start,
                    "end": end,
                    "score": scores[i].item()
                })
        prev_id = token_id

    # 10. Merge characters into words
    word_intervals = merge_to_words(char_intervals, transcript)

    # 11. Merge words into sentences (using punctuation in transcript)
    sentence_intervals = merge_words_into_sentences(word_intervals, transcript)

    # 12. Prepare output JSON
    output_data = {
        "audio_file": args.audio,
        "transcript": transcript,
        "frame_period_sec": frame_period,
        "char_alignment": char_intervals,
        "word_alignment": word_intervals,
        "sentence_alignment": sentence_intervals,
    }

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON alignment to {args.output}")

    # 13. Optional subtitle outputs
    if args.srt:
        srt_path = Path(args.output).with_suffix(".word.srt")
        write_srt(word_intervals, srt_path)
        print(f"Saved word-level SRT to {srt_path}")

    if args.sentence_srt:
        sent_srt_path = Path(args.output).with_suffix(".sentence.srt")
        write_sentence_srt(sentence_intervals, sent_srt_path)
        print(f"Saved sentence-level SRT to {sent_srt_path}")


if __name__ == "__main__":
    main()
