"""Tell the wearer ("user") apart from other speakers by voice.

Enrollment stores a voiceprint of the user: a WeSpeaker ResNet34 speaker
embedding (VoxCeleb-trained, 26 MB ONNX). Each transcribed segment is compared
to it by cosine similarity: a segment is the user if it is close enough to the
profile and closer to it than to any other voice heard so far. Other voices are
clustered online into "speaker 1", "speaker 2", ...
"""
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

SAMPLE_RATE = 16000
EMBED_MODEL = ("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx")
PROFILE = Path(__file__).parent / "voice_profile.npz"


class VoiceEmbedder:
    def __init__(self):
        self.session = ort.InferenceSession(hf_hub_download(*EMBED_MODEL), providers=["CPUExecutionProvider"])
        self.opts = knf.FbankOptions()
        self.opts.frame_opts.dither = 0.0
        self.opts.frame_opts.samp_freq = SAMPLE_RATE
        self.opts.mel_opts.num_bins = 80

    def fbank(self, audio):
        fbank = knf.OnlineFbank(self.opts)
        fbank.accept_waveform(SAMPLE_RATE, (audio * 32768).tolist())  # WeSpeaker expects int16 scale
        fbank.input_finished()
        feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)])
        return feats - feats.mean(axis=0)  # cepstral mean normalization

    def __call__(self, audio):
        """Unit-length voiceprint of a float32 16 kHz clip."""
        emb = self.session.run(None, {"feats": self.fbank(audio)[None].astype(np.float32)})[0][0]
        return emb / np.linalg.norm(emb)


class SpeakerTracker:
    """Labels segments as "user" or "speaker N"."""

    def __init__(self, embedder, profile, user_threshold=0.5, other_threshold=0.5,
                 min_sec=1.0, new_speaker_sec=2.0, short_margin=0.2):
        self.embed = embedder
        self.user = profile
        self.user_threshold = user_threshold
        self.other_threshold = other_threshold
        self.min_samples = int(min_sec * SAMPLE_RATE)
        self.new_speaker_samples = int(new_speaker_sec * SAMPLE_RATE)
        self.short_margin = short_margin
        self.others = []  # running sums of voiceprints, one per other speaker
        self.last = "speaker 1"
        self.last_score = None

    def label(self, audio):
        if len(audio) < self.min_samples:
            return self._label_short(audio)
        emb = self.embed(audio)
        long_enough = len(audio) >= self.new_speaker_samples
        self.last_score = float(emb @ self.user)
        scores = [float(emb @ (s / np.linalg.norm(s))) for s in self.others]
        best = max(scores, default=-1.0)

        if self.last_score >= self.user_threshold and self.last_score > best:
            self.last = "user"
        elif best >= self.other_threshold or (scores and not long_enough):
            # a known other voice, or too short to start a new one: take the closest known voice
            if not long_enough and self.last_score > best:
                self.last = "user"
                return self.last
            k = int(np.argmax(scores))
            if long_enough:
                self.others[k] = self.others[k] + emb
            self.last = f"speaker {k + 1}"
        else:
            self.others.append(emb)
            self.last = f"speaker {len(self.others)}"
        return self.last

    def _label_short(self, audio):
        """Clips under min_sec ("Yes.", "Okay.") give noisy voiceprints: switch speaker
        only on a clear margin between known voices, otherwise the turn continues."""
        emb = self.embed(audio)
        self.last_score = float(emb @ self.user)
        voices = [("user", self.last_score)] + [
            (f"speaker {k + 1}", float(emb @ (s / np.linalg.norm(s)))) for k, s in enumerate(self.others)]
        voices.sort(key=lambda v: -v[1])
        if len(voices) > 1 and voices[0][1] - voices[1][1] >= self.short_margin:
            self.last = voices[0][0]
        return self.last


def load_profile(path=PROFILE):
    """(voiceprint, calibrated user threshold)"""
    data = np.load(path)
    return data["profile"], float(data["threshold"])


def enroll(embedder, speech, path=PROFILE, window_sec=3.0):
    """Save the user's voiceprint from speech-only audio (silence removed).

    The threshold is calibrated from how well short slices of the enrollment speech
    match the whole: a bit below the user's typical score, so that similar-sounding
    voices fall under it.
    """
    profile = embedder(speech)
    win, hop = int(window_sec * SAMPLE_RATE), int(window_sec * SAMPLE_RATE / 2)
    scores = [float(embedder(speech[i:i + win]) @ profile) for i in range(0, len(speech) - win + 1, hop)]
    threshold = float(np.clip(np.percentile(scores, 10) - 0.15, 0.35, 0.65))
    np.savez(path, profile=profile, threshold=threshold)
    return profile, threshold, scores
