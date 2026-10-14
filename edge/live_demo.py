"""Live LlamaPIE: microphone -> Whisper -> proactive whispers.

Audio is cut into 0.5 s chunks. Speech chunks are buffered and transcribed with
Whisper when the speaker pauses; every silent chunk becomes one " |SILENCE >"
marker, which is where the classifier decides whether the agent should speak.

Speaker labels ("user" = the wearer, "speaker 1", ...) come from one of:
  auto      voice recognition against an enrolled voice profile (enroll once first)
  manual    press Enter to switch between "user" and "speaker 1"
  timeline  the .speakers.txt written by dialogue_to_audio.py (replayed files only)

    python live_demo.py --enroll                          # record ~20 s of your voice once
    python live_demo.py --memory ../samples/00001/memory.txt
    python live_demo.py --audio-file conversation.wav     # replay a recording
"""
import argparse
import collections
import math
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # silence the warning when audio libs fork

import mlx_whisper
from proactive_agent import BIG_MODEL, SILENCE, SMALL_MODEL, ProactiveAgent
from quantize_models import SMALL_8BIT, generator_path
from speaker_id import PROFILE, SpeakerTracker, VoiceEmbedder, enroll, load_profile

SAMPLE_RATE = 16000
CHUNK_SEC = 0.5  # one silence marker per silent chunk, as in the training data
CHUNK = int(SAMPLE_RATE * CHUNK_SEC)
FRAME = 320  # 20 ms frames for the energy detector

COLOR = sys.stdout.isatty()


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


class EnergyVAD:
    """Speech/silence decision per chunk, with an adaptive noise floor.

    The floor is a low percentile of the last 30 s of chunk levels, so it follows
    the room up as well as down (e.g. a fan turning on, or mic auto-gain).
    """

    def __init__(self, threshold=None, window_sec=30):
        self.fixed = threshold
        self.levels = collections.deque(maxlen=int(window_sec / CHUNK_SEC))
        self.last_level = 0.0
        self.threshold = threshold or 0.003

    def level(self, chunk):
        frames = chunk[: len(chunk) // FRAME * FRAME].reshape(-1, FRAME)
        return float(np.percentile(np.sqrt((frames ** 2).mean(axis=1)), 80))

    def is_speech(self, chunk):
        self.last_level = self.level(chunk)
        self.levels.append(self.last_level)
        if not self.fixed:
            self.threshold = max(3 * float(np.percentile(self.levels, 10)), 0.003)
        return self.last_level > self.threshold


class Speakers:
    """Current speaker label, switched with Enter or by a timeline of turns."""

    LABELS = ["user", "speaker 1"]

    def __init__(self, first, timeline=None):
        self.current = first
        self.timeline = timeline or []  # [(start_seconds, label)]
        self.last_written = None
        self.lock = threading.Lock()

    def toggle(self):
        with self.lock:
            self.current = self.LABELS[1 - self.LABELS.index(self.current)] if self.current in self.LABELS else "user"
            return self.current

    def at(self, seconds):
        """Speaker at a point in the audio (timeline if given, else the live label)."""
        with self.lock:
            label = self.current
        for start, name in self.timeline:
            if start <= seconds:
                label = name
        return label

    def prefix(self, label):
        """Label to write before the next transcribed text, if the turn changed."""
        if label == self.last_written:
            return ""
        self.last_written = label
        return label + ": "

    def interrupt(self):
        """The agent spoke, so the next speech starts a new turn."""
        self.last_written = None


def load_timeline(path):
    timeline = []
    for line in path.read_text().splitlines():
        start, name = line.split(" ", 1)
        timeline.append((float(start), name))
    return timeline


def listen_for_enter(speakers):
    for _ in sys.stdin:
        print(paint(f"[now speaking: {speakers.toggle()}]", "2"), flush=True)


def mic_source(chunks, device):
    import sounddevice as sd

    def callback(indata, frames, time_info, status):
        chunks.put(indata[:, 0].copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=CHUNK, device=device, callback=callback)
    stream.start()
    return stream


def read_wav(path):
    """Mono float32 16 kHz audio from a WAV file."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # e.g. macOS `say` adds a FLLR chunk
        rate, audio = wavfile.read(path)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return audio


def speech_only(audio):
    """Drop the silent 0.5 s chunks of a recording."""
    chunks = [audio[i:i + CHUNK] for i in range(0, len(audio) - CHUNK + 1, CHUNK)]
    vad = EnergyVAD()
    levels = [vad.level(c) for c in chunks]
    # the quietest chunks may still be speech if the recording has few pauses
    threshold = max(min(3 * float(np.percentile(levels, 10)), 0.3 * float(np.percentile(levels, 90))), 0.003)
    kept = [c for c, level in zip(chunks, levels) if level > threshold]
    return np.concatenate(kept) if kept else np.zeros(0, dtype=np.float32)


def run_enrollment(args):
    if args.enroll_from:
        audio = read_wav(args.enroll_from)
    else:
        import sounddevice as sd

        print(paint(f"Recording {args.enroll:.0f} s. Talk naturally in your normal voice "
                    "(read something aloud, or describe your day)...", "1"), flush=True)
        audio = sd.rec(int(args.enroll * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                       dtype="float32", device=args.device)
        sd.wait()
        audio = audio[:, 0]
    speech = speech_only(audio)
    if len(speech) < 5 * SAMPLE_RATE:
        sys.exit(f"Only {len(speech) / SAMPLE_RATE:.1f} s of speech detected; need at least 5 s. Try again.")
    _, threshold, scores = enroll(VoiceEmbedder(), speech, args.profile)
    print(f"Saved voice profile from {len(speech) / SAMPLE_RATE:.0f} s of speech to {args.profile}")
    print(f"Your 3 s slices match it at {min(scores):.2f}-{max(scores):.2f}; "
          f"phrases scoring {threshold:.2f} or more will count as you.")


def file_source(chunks, path, realtime):
    audio = read_wav(path)

    def run():
        for start in range(0, len(audio), CHUNK):
            chunk = audio[start:start + CHUNK]
            chunks.put(np.pad(chunk, (0, CHUNK - len(chunk))))
            if realtime:
                time.sleep(CHUNK_SEC)
        chunks.put(None)

    threading.Thread(target=run, daemon=True).start()


def local_or(path, repo):
    """Smaller local copy from quantize_models.py if it exists, else the original model."""
    return str(path) if path.exists() else repo


def transcribe(audio, model):
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=model, language="en", temperature=0.0,
                                    condition_on_previous_text=False, verbose=None)
    # drop segments Whisper itself thinks are not speech (noise often transcribes as "Thank you.")
    texts = [s["text"].strip() for s in result["segments"]
             if s["no_speech_prob"] < 0.6 and s["avg_logprob"] > -1.0]
    return " ".join(t for t in texts if t)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory", type=Path, help="text file with facts about the user")
    parser.add_argument("--audio-file", type=Path, help="replay a WAV file instead of the microphone")
    parser.add_argument("--fast", action="store_true", help="with --audio-file, do not wait in real time")
    parser.add_argument("--device", help="input device name or index (see `python -m sounddevice`)")
    parser.add_argument("--speakers", choices=["auto", "manual", "timeline"],
                        help="how to label speakers (default: timeline for a file with .speakers.txt, "
                             "else auto if a voice profile exists, else manual)")
    parser.add_argument("--enroll", type=float, nargs="?", const=20.0, metavar="SECONDS",
                        help="record your voice from the mic (default 20 s), save the profile and exit")
    parser.add_argument("--enroll-from", type=Path, metavar="WAV", help="enroll from a recording instead")
    parser.add_argument("--profile", type=Path, default=PROFILE, help="voice profile file")
    parser.add_argument("--speaker-threshold", type=float,
                        help="voice similarity (0-1) needed to count a phrase as the user "
                             "(default: calibrated at enrollment)")
    parser.add_argument("--first-speaker", choices=Speakers.LABELS, default="user", help="for manual mode")
    parser.add_argument("--whisper-model", default="mlx-community/whisper-small.en-mlx",
                        help="e.g. mlx-community/whisper-large-v3-turbo for better accuracy (~1.6 GB)")
    parser.add_argument("--small-model", default=local_or(SMALL_8BIT, SMALL_MODEL), help="classifier (MLX)")
    parser.add_argument("--big-model", default=local_or(generator_path(6), BIG_MODEL), help="generator (MLX)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="P(speak) the agent needs at a pause; lower (e.g. 0.3) = more proactive")
    parser.add_argument("--debug", action="store_true", help="print loudness and P(speak) at every pause")
    parser.add_argument("--vad-threshold", type=float, help="fixed speech RMS level instead of adaptive")
    parser.add_argument("--max-segment", type=float, default=8.0,
                        help="transcribe after this many seconds of continuous speech")
    parser.add_argument("--max-silences", type=int, default=6,
                        help="silence markers to add per pause; a long pause adds no more")
    parser.add_argument("--save-dialogue", type=Path, help="write the dialogue (dataset format) here on exit")
    args = parser.parse_args()

    if args.enroll or args.enroll_from:
        run_enrollment(args)
        return

    timeline_path = args.audio_file.with_suffix(".speakers.txt") if args.audio_file else None
    if args.speakers is None:
        if timeline_path and timeline_path.exists():
            args.speakers = "timeline"
        else:
            args.speakers = "auto" if args.profile.exists() else "manual"
    if args.speakers == "auto" and not args.profile.exists():
        sys.exit(f"No voice profile at {args.profile}: run `python live_demo.py --enroll` first.")
    if args.speakers == "timeline" and not (timeline_path and timeline_path.exists()):
        sys.exit("--speakers timeline needs --audio-file with a matching .speakers.txt")

    print(f"Loading {args.small_model} and {args.big_model}...", flush=True)
    if args.small_model == SMALL_MODEL or args.big_model == BIG_MODEL:
        print(paint("Tip: run quantize_models.py once for smaller, faster models (~3 GB less memory).", "2"))
    agent = ProactiveAgent(args.small_model, args.big_model, threshold=args.threshold)
    memory = args.memory.read_text() if args.memory else ""
    agent.reset(memory)
    transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), args.whisper_model)  # load Whisper now

    timeline = None
    tracker = None
    if args.speakers == "timeline":
        timeline = load_timeline(timeline_path)
        print(f"Speaker turns from {timeline_path}")
    elif args.speakers == "auto":
        profile, threshold = load_profile(args.profile)
        threshold = args.speaker_threshold or threshold
        tracker = SpeakerTracker(VoiceEmbedder(), profile, user_threshold=threshold)
        print(f"Recognizing your voice from {args.profile} (match threshold {threshold:.2f})")
    speakers = Speakers(args.first_speaker, timeline)
    vad = EnergyVAD(args.vad_threshold)
    chunks = queue.Queue()
    dialogue = []

    if args.audio_file:
        file_source(chunks, args.audio_file, realtime=not args.fast)
        stream = None
    else:
        stream = mic_source(chunks, args.device)
        if args.speakers == "manual":
            if sys.stdin.isatty():
                threading.Thread(target=listen_for_enter, args=(speakers,), daemon=True).start()
            print(paint(f"Listening. Speaking: {speakers.current}. "
                        "Press Enter to switch speaker, Ctrl+C to stop.", "1"), flush=True)
        else:
            print(paint("Listening. Ctrl+C to stop.", "1"), flush=True)

    def flush(pending, voiced, label):
        text = transcribe(np.concatenate(pending), args.whisper_model)
        if not text:
            return
        if tracker:
            label = tracker.label(np.concatenate(voiced))  # speech chunks only, no padding
            if args.debug:
                score = f"{tracker.last_score:.2f}"
                print(paint(f"  [voice match with user: {score}]", "2"), flush=True)
        line = speakers.prefix(label) + text
        dialogue.append(line)
        print(line, flush=True)
        agent.feed_words(line)

    pending = []
    voiced = []
    label = None
    preroll = None
    silent_run = 0
    elapsed = 0.0
    try:
        while True:
            chunk = chunks.get()
            if chunk is None:
                if pending:
                    flush(pending, voiced, label)
                break
            elapsed += CHUNK_SEC
            if not args.fast and chunks.qsize() * CHUNK_SEC > 3:
                print(paint(f"[falling behind by {chunks.qsize() * CHUNK_SEC:.1f}s]", "2"), flush=True)

            if vad.is_speech(chunk):
                if not pending:
                    label = speakers.at(elapsed)  # chunk ends at `elapsed`, so speech has started
                    if preroll is not None:
                        pending.append(preroll)  # keep the onset of the first word
                pending.append(chunk)
                voiced.append(chunk)
                silent_run = 0
                if len(pending) * CHUNK_SEC >= args.max_segment:
                    flush(pending, voiced, label)
                    pending, voiced = [], []
                continue

            if pending:
                pending.append(chunk)
                flush(pending, voiced, label)
                pending, voiced = [], []
            preroll = chunk
            silent_run += 1
            if silent_run > args.max_silences:
                continue
            dialogue.append(SILENCE)
            whisper = agent.feed_silence()
            if args.debug:
                print(paint(f"  [pause  level {vad.last_level:.4f} < {vad.threshold:.4f}  "
                            f"P(speak) {agent.last_prob:.2f}]", "2"), flush=True)
            if whisper:
                dialogue.append("Agent: " + whisper)
                speakers.interrupt()
                print(paint(f"  >>> AGENT: {whisper}", "1;32"), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.stop()
        if args.save_dialogue:
            args.save_dialogue.write_text(" ".join(dialogue) + "\n")
            print(f"Dialogue saved to {args.save_dialogue}")


if __name__ == "__main__":
    main()
