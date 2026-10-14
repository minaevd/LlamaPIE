"""Streaming LlamaPIE pipeline on MLX.

The small model classifies every incoming token and is consulted at each
" |SILENCE >" marker (0.5 s of silence). When it fires, the large model
generates a short whisper from the conversation so far. Both models keep a
KV cache of the whole conversation, so each new word costs only its own tokens.
"""
from typing import Optional

import mlx.core as mx
from mlx_lm import generate, inject_init_prompt, load
from mlx_lm.models import cache

SMALL_MODEL = "tuochao/Llama-3.2-1B-Proactive-Classifier-Aug-mlx-fp16"
BIG_MODEL = "tuochao/Llama-3.1-8B-Proactive-Gen-Positive-mlx-8bit"

SYSTEM_PROMPT = (
    "You are a proactive AI agent designed to actively help humans by reminding and "
    "assisting them in following dialogue, by whispering short, concise phrases "
    "(1-3 words) to its user."
)
SILENCE = "|SILENCE >"
ASSISTANT_TAIL = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def adapt_to_ASR(dialogue):
    # match the ASR-style text the classifier was augmented with
    dialogue = dialogue.lower()
    dialogue = dialogue.replace("|silence >", SILENCE)
    for p in [".", "!", "?", ";", ","]:
        dialogue = dialogue.replace(p, "")
    return dialogue


class ProactiveAgent:
    def __init__(self, small_model=SMALL_MODEL, big_model=BIG_MODEL, max_whisper_tokens=16, threshold=0.5):
        self.model_small, self.tokenizer_small = load(small_model)
        self.model_big, self.tokenizer_big = load(big_model)
        # keep both models wired for the whole session; mlx_lm otherwise wires and
        # unwires them around every call, which stalls for seconds at a time
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.max_whisper_tokens = max_whisper_tokens
        self.threshold = threshold  # P(speak) needed at a silence marker; lower = more proactive
        self.last_prob = 0.0  # P(speak) at the most recent silence marker
        self.mask_token = self.tokenizer_small.encode(" >", add_special_tokens=False)[0]
        self.reset()

    def reset(self, memory=""):
        """Start a new conversation, optionally with facts about the user."""
        self.cache_small = cache.make_prompt_cache(self.model_small)
        self.cache_big = cache.make_prompt_cache(self.model_big)
        self.started = False

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"You have the following memory of facts for the user:\n{memory}"},
            {"role": "user", "content": ""},
        ]
        conv_text = self.tokenizer_big.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        assert conv_text.endswith(ASSISTANT_TAIL)
        head = conv_text[: -len(ASSISTANT_TAIL)]
        self.tail_ids = mx.array(self.tokenizer_big.encode(ASSISTANT_TAIL, add_special_tokens=False))

        # the large model sees system prompt + memory + the open user turn
        head_ids = self.tokenizer_big.encode(head, add_special_tokens=True)
        inject_init_prompt(prompt=mx.array(head_ids), model=self.model_big, prompt_cache=self.cache_big)
        # the small model only sees the dialogue, starting from BOS
        bos_ids = self.tokenizer_small.encode(self.tokenizer_small.bos_token, add_special_tokens=False)
        self._classify(bos_ids)

    def _classify(self, ids):
        """Per-token P(speak) from the small model, all tokens in one forward pass."""
        logits = self.model_small(mx.array(ids)[None], cache=self.cache_small)
        return mx.softmax(logits[0].astype(mx.float32), axis=-1)[:, 1].tolist()

    def _feed(self, text):
        """Append text to both models; return True if the classifier fires on a silence marker."""
        if self.started:
            text = " " + text
        self.started = True

        small_ids = self.tokenizer_small.encode(text, add_special_tokens=False)
        probs = self._classify(small_ids)
        big_ids = self.tokenizer_big.encode(text, add_special_tokens=False)
        inject_init_prompt(prompt=mx.array(big_ids), model=self.model_big, prompt_cache=self.cache_big)

        # only respond at the silence marker
        at_marker = [p for t, p in zip(small_ids, probs) if t == self.mask_token]
        if not at_marker:
            return False
        self.last_prob = at_marker[-1]
        return self.last_prob >= self.threshold

    def feed_words(self, text):
        """Add transcribed speech, e.g. "user: i think so" or "speaker 1: how are you"."""
        text = adapt_to_ASR(text).strip()
        if text:
            self._feed(text)

    def feed_silence(self) -> Optional[str]:
        """Add 0.5 s of silence; return a whisper if the agent decides to speak."""
        if not self._feed(SILENCE):
            return None

        # generate on top of the conversation cache, then roll the cache back
        offset = self.cache_big[0].offset
        whisper = generate(
            self.model_big,
            self.tokenizer_big,
            prompt=self.tail_ids,
            prompt_cache=self.cache_big,
            max_tokens=self.max_whisper_tokens,
        ).strip()
        cache.trim_prompt_cache(self.cache_big, self.cache_big[0].offset - offset)
        if not whisper:
            return None

        # both models are whisper-aware: the agent's turn becomes part of the dialogue
        self._feed("Agent: " + whisper)
        return whisper
