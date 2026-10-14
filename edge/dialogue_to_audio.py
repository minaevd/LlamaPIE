"""Synthesize a dialogue file into a WAV with macOS `say`, for testing live_demo.py.

The dialogue uses the dataset format ("User: ... |SILENCE > Speaker 1: ..."). The user
and the other speakers get different voices, and each run of k "|SILENCE >" markers
becomes a pause long enough for the live pipeline to detect k silent 0.5 s chunks.
"Agent:" turns in the file are skipped. Speaker turns are written next to the WAV
(conversation.speakers.txt), and live_demo.py uses them to label the transcript.

    python dialogue_to_audio.py ../samples/00001/dialogue.txt conversation.wav
"""
import argparse
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 16000
TOKEN = re.compile(r"(\|SILENCE >|User:|Agent:|Speaker ?\d+:)")


def say(text, voice, rate):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn.wav"
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(path),
                        f"--data-format=LEI16@{SAMPLE_RATE}", text], check=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)  # say pads the header with a FLLR chunk
            return wavfile.read(path)[1]


def label(tag):
    """'Speaker1:' -> 'speaker 1', 'User:' -> 'user'"""
    return re.sub(r"speaker ?(\d+)", r"speaker \1", tag[:-1].lower())


def other_voice(tag, voices):
    n = re.search(r"\d+", tag)
    return voices[(int(n.group()) - 1 if n else 0) % len(voices)]


def pause(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dialogue", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--user-voice", default="Samantha")
    parser.add_argument("--other-voice", default="Daniel",
                        help="voice for Speaker 1; a comma list gives Speaker 2, 3, ... their own voices")
    parser.add_argument("--rate", type=int, default=180, help="words per minute")
    args = parser.parse_args()

    audio = [pause(0.5)]
    turns = []
    speaker = None
    silences = 0
    turn_change = False
    for part in TOKEN.split(args.dialogue.read_text()):
        part = part.strip()
        if not part:
            continue
        if part == "|SILENCE >":
            silences += 1
            continue
        if part.endswith(":"):
            turn_change = turn_change or part != speaker
            speaker = part
            continue
        if speaker == "Agent:" or speaker is None:
            continue
        # a turn change needs >= 1 s of silence so the live pipeline cuts the segment there
        gap = max(0.5 * silences + 0.5 if silences else 0.0, 1.0 if turn_change else 0.0)
        if gap:
            audio.append(pause(gap))
        if turn_change:
            turns.append((sum(len(a) for a in audio) / SAMPLE_RATE, label(speaker)))
        silences = 0
        turn_change = False
        voice = args.user_voice if speaker == "User:" else other_voice(speaker, args.other_voice.split(","))
        audio.append(say(part, voice, args.rate))
    audio.append(pause(0.5 * silences + 1.0))

    audio = np.concatenate(audio)
    wavfile.write(args.output, SAMPLE_RATE, audio)
    speakers_path = args.output.with_suffix(".speakers.txt")
    speakers_path.write_text("".join(f"{t:.2f} {name}\n" for t, name in turns))
    print(f"Wrote {args.output} ({len(audio) / SAMPLE_RATE:.0f} s) and {speakers_path}")


if __name__ == "__main__":
    main()
