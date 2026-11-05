"""LLM access for the CP4 agents: one client, one hash-checked cache.

Every call goes through LLMClient.complete(). The request (model, system, prompt,
max_tokens) is hashed; the response is stored in data/llm_cache/<key>.json with
its own sha256 and the tokens of that one call. In replay mode a missing entry is
an error, and an entry whose stored text no longer matches its hash is rejected,
so a cached run cannot change silently. The agents only ever send schemas, field
names and summary statistics.

Changes from the TA version (CP4 audit W7, W16):
- No temperature. Claude Sonnet 5.5 rejects non-default sampling values with a
  400, so the TA client could not make a single call. Runs are reproducible
  through the cache, not through sampling settings.
- Each cache entry stores that call's tokens (the TA stored the running total),
  so the cost of a run can be recomputed from the cache alone.
- One price table, from the Claude API model table (cached 2026-10-06). The TA
  hard-coded $3/$15 per million tokens, the old Sonnet 4.6 price; Sonnet 5.5 is
  $2/$10.
- A refusal or a response cut off at max_tokens raises LLMError and is not cached.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from theorylab.catalog import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "llm_cache"
DEFAULT_MODEL = "claude-sonnet-5-5"
# US$ per million (input, output) tokens, Claude API first-party rates.
PRICES_USD_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class CacheError(RuntimeError):
    """A cache entry is missing in replay mode, or fails its hash check."""


class LLMError(RuntimeError):
    """The model refused or its answer was cut off; nothing is cached."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def price(model: str) -> tuple:
    if model not in PRICES_USD_PER_MTOK:
        raise KeyError(f"no price for model {model!r}; add it to PRICES_USD_PER_MTOK")
    return PRICES_USD_PER_MTOK[model]


@dataclass
class Usage:
    calls: int = 0              # live API calls made by this client
    cache_hits: int = 0
    input_tokens: int = 0       # tokens of the live calls
    output_tokens: int = 0
    cached_input_tokens: int = 0   # tokens the replayed entries cost when they were made
    cached_output_tokens: int = 0

    def cost_usd(self, model: str, include_cached: bool = True) -> float:
        """Cost of the calls this client served; with include_cached, as if none were cached."""
        p_in, p_out = price(model)
        tin = self.input_tokens + (self.cached_input_tokens if include_cached else 0)
        tout = self.output_tokens + (self.cached_output_tokens if include_cached else 0)
        return tin / 1e6 * p_in + tout / 1e6 * p_out


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    cache_dir: Path = CACHE_DIR
    replay_only: bool = False          # True: never call the API, cache entries must exist
    max_tokens: int = 8000
    usage: Usage = field(default_factory=Usage)
    _api: object = None

    def key(self, system: str, prompt: str) -> str:
        return _sha(json.dumps({"model": self.model, "system": system, "prompt": prompt,
                                "max_tokens": self.max_tokens}, sort_keys=True))

    def _path(self, key: str) -> Path:
        return Path(self.cache_dir) / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if _sha(entry["text"]) != entry["response_sha256"]:
            raise CacheError(f"{path.name}: stored response does not match its sha256")
        if entry["request_sha256"] != key:
            raise CacheError(f"{path.name}: request hash does not match the file name")
        return entry

    def _write_cache(self, key: str, system: str, prompt: str, reply: dict) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        entry = {"request_sha256": key, "model": self.model, "max_tokens": self.max_tokens,
                 "system": system, "prompt": prompt, "text": reply["text"],
                 "response_sha256": _sha(reply["text"]), "stop_reason": reply["stop_reason"],
                 "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"]}
        self._path(key).write_text(json.dumps(entry, indent=2), encoding="utf-8")

    def _call_api(self, system: str, prompt: str) -> dict:
        if self._api is None:
            import anthropic  # imported late so offline replay needs no SDK or key
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise CacheError("ANTHROPIC_API_KEY is not set and the request is not cached")
            self._api = anthropic.Anthropic()
        msg = self._api.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": prompt}])
        reply = {"text": "".join(b.text for b in msg.content if getattr(b, "type", "") == "text"),
                 "stop_reason": msg.stop_reason,
                 "input_tokens": int(msg.usage.input_tokens),
                 "output_tokens": int(msg.usage.output_tokens)}
        self.usage.calls += 1
        self.usage.input_tokens += reply["input_tokens"]
        self.usage.output_tokens += reply["output_tokens"]
        if reply["stop_reason"] in ("refusal", "max_tokens"):
            raise LLMError(f"model stopped with {reply['stop_reason']!r}; response not cached")
        return reply

    def complete(self, system: str, prompt: str) -> str:
        key = self.key(system, prompt)
        cached = self._read_cache(key)
        if cached is not None:
            self.usage.cache_hits += 1
            self.usage.cached_input_tokens += int(cached.get("input_tokens", 0))
            self.usage.cached_output_tokens += int(cached.get("output_tokens", 0))
            return cached["text"]
        if self.replay_only:
            raise CacheError(f"no cache entry {key[:12]} for this request (replay-only mode)")
        reply = self._call_api(system, prompt)
        self._write_cache(key, system, prompt, reply)
        return reply["text"]

    def verify_cache(self) -> dict:
        """Hash-check every entry; returns {'ok': n, 'bad': [file names]}."""
        ok, bad = 0, []
        for path in sorted(Path(self.cache_dir).glob("*.json")):
            try:
                self._read_cache(path.stem)
                ok += 1
            except (CacheError, KeyError, json.JSONDecodeError):
                bad.append(path.name)
        return {"ok": ok, "bad": bad}


def extract_json(text: str) -> dict:
    """The first top-level JSON object in a response (models sometimes fence it)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the response")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in the response")
