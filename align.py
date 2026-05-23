#!/usr/bin/env python3
"""
CTC Forced Alignment Tool — omnilingual sherpa-onnx model.

Usage
-----
    python align.py audio.wav "transcript text" --srt
    python align.py audio.wav transcript.txt --output out.json --sentence-srt

The model is downloaded automatically from Hugging Face on first run.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
import sherpa_onnx

from utils import (
    load_audio,
    read_tokens,
    text_to_token_ids,
    get_emissions_direct,
    forced_align_emissions,
    build_char_intervals,
    merge_to_words,
    merge_words_into_sentences,
    write_srt,
    write_sentence_srt,
)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

MODEL_REPO    = "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"
MODEL_FILE    = "model.int8.onnx"
TOKENS_FILE   = "tokens.txt"
BLANK_ID      = 0
SAMPLE_RATE   = 16000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_recognizer(model_path: str, tokens_path: str) -> sherpa_onnx.OfflineRecognizer:
    """
    Instantiate the recogniser exactly as the reference model.py does:
        sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(tokens, model)
    """
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        tokens=tokens_path,
        model=model_path,
        num_threads=2,
        debug=False,
    )


def sanity_decode(recognizer: sherpa_onnx.OfflineRecognizer,
                  samples: np.ndarray,
                  sample_rate: int) -> str:
    """Run a full decode pass and return the transcript (for sanity check)."""
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced alignment with CTC omnilingual model")
    parser.add_argument("audio",      help="Path to input audio file (WAV recommended)")
    parser.add_argument("transcript", help="Transcript text, or path to a .txt file")
    parser.add_argument("--output", "-o", default="alignment.json",
                        help="Output JSON file (default: alignment.json)")
    parser.add_argument("--srt",          action="store_true",
                        help="Generate word-level SRT subtitles")
    parser.add_argument("--sentence-srt", action="store_true",
                        help="Generate sentence-level SRT subtitles")
    parser.add_argument("--model-repo", default=MODEL_REPO,
                        help="Hugging Face repo ID for the model")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--blank-id",    type=int, default=BLANK_ID)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load transcript
    # ------------------------------------------------------------------
    t_path = Path(args.transcript)
    if t_path.is_file():
        transcript = t_path.read_text(encoding="utf-8").strip()
    else:
        transcript = args.transcript.strip()

    if not transcript:
        print("Error: empty transcript", file=sys.stderr)
        sys.exit(1)

    print(f"Transcript: {transcript!r}")

    # ------------------------------------------------------------------
    # 2. Download model & tokens from Hugging Face
    # ------------------------------------------------------------------
    print(f"Downloading model from {args.model_repo} …")
    model_path  = hf_hub_download(repo_id=args.model_repo, filename=MODEL_FILE)
    tokens_path = hf_hub_download(repo_id=args.model_repo, filename=TOKENS_FILE)
    print(f"  model  → {model_path}")
    print(f"  tokens → {tokens_path}")

    # ------------------------------------------------------------------
    # 3. Load token vocabulary
    # ------------------------------------------------------------------
    token2id, id2token = read_tokens(tokens_path)
    print(f"Vocabulary size: {len(token2id)}")

    # ------------------------------------------------------------------
    # 4. Convert transcript to subword token IDs
    # ------------------------------------------------------------------
    target_ids = text_to_token_ids(transcript, token2id)
    if not target_ids:
        print("Error: transcript produced no valid tokens", file=sys.stderr)
        sys.exit(1)
    print(f"Target token sequence ({len(target_ids)} tokens): "
          f"{[id2token[i] for i in target_ids]}")

    # ------------------------------------------------------------------
    # 5. Load audio
    # ------------------------------------------------------------------
    print(f"Loading audio: {args.audio}")
    samples, sr = load_audio(args.audio, target_sr=args.sample_rate)
    duration = len(samples) / sr
    print(f"  {len(samples)} samples @ {sr} Hz  ({duration:.2f} s)")

    # ------------------------------------------------------------------
    # 6. Sanity-check decode via the full sherpa pipeline
    # ------------------------------------------------------------------
    print("Building recogniser …")
    recognizer = build_recognizer(model_path, tokens_path)
    hyp = sanity_decode(recognizer, samples, sr)
    print(f"Sherpa transcription (sanity check): {hyp!r}")

    # ------------------------------------------------------------------
    # 7. Get CTC log-probs directly from the ONNX encoder
    # ------------------------------------------------------------------
    print("Computing CTC log-probabilities …")
    log_probs = get_emissions_direct(model_path, samples, sr)
    n_frames = log_probs.shape[0]
    frame_period = duration / n_frames
    print(f"  {n_frames} frames, frame period = {frame_period:.4f} s")

    # ------------------------------------------------------------------
    # 8. Forced alignment
    # ------------------------------------------------------------------
    print("Running forced alignment …")
    alignment, scores = forced_align_emissions(log_probs, target_ids, blank_id=args.blank_id)

    # ------------------------------------------------------------------
    # 9. Build intervals
    # ------------------------------------------------------------------
    char_intervals = build_char_intervals(
        alignment, scores, id2token, frame_period, blank_id=args.blank_id
    )
    word_intervals = merge_to_words(char_intervals, transcript)
    sentence_intervals = merge_words_into_sentences(word_intervals, transcript)

    # ------------------------------------------------------------------
    # 10. Save JSON
    # ------------------------------------------------------------------
    output_data = {
        "audio_file":        args.audio,
        "transcript":        transcript,
        "duration_sec":      round(duration, 4),
        "n_frames":          n_frames,
        "frame_period_sec":  round(frame_period, 6),
        "sherpa_hypothesis": hyp,
        "token_alignment":   char_intervals,
        "word_alignment":    word_intervals,
        "sentence_alignment": sentence_intervals,
    }

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved alignment JSON → {out_path}")

    # ------------------------------------------------------------------
    # 11. Optional SRT outputs
    # ------------------------------------------------------------------
    if args.srt:
        srt_path = out_path.with_suffix(".word.srt")
        write_srt(word_intervals, srt_path)
        print(f"Saved word-level SRT → {srt_path}")

    if args.sentence_srt:
        sent_srt = out_path.with_suffix(".sentence.srt")
        write_sentence_srt(sentence_intervals, sent_srt)
        print(f"Saved sentence-level SRT → {sent_srt}")

    # Quick summary
    print("\n--- Word alignment ---")
    for w in word_intervals:
        print(f"  {w['start']:6.3f}s – {w['end']:6.3f}s  {w['word']}")


if __name__ == "__main__":
    main()
