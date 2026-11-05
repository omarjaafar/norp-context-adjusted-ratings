"""LLM access for the CP4 agents: one client, one hash-checked cache.

Every call goes through LLMClient.complete(). The request (model, system, prompt,
temperature) is hashed; the response is stored in data/llm_cache/<key>.json with
its own sha256. In replay mode a missing entry is an error, and an entry whose
stored text no longer matches its hash is rejected, so a cached run cannot change
silently. The agents only ever send schemas, field names and summary statistics.
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


class CacheError(RuntimeError):
    """A cache entry is missing in replay mode, or fails its hash check."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, in_per_mtok: float = 3.0, out_per_mtok: float = 15.0) -> float:
        return self.input_tokens / 1e6 * in_per_mtok + self.output_tokens / 1e6 * out_per_mtok


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    cache_dir: Path = CACHE_DIR
    replay_only: bool = False          # True: never call the API, cache entries must exist
    temperature: float = 0.0
    max_tokens: int = 4000
    usage: Usage = field(default_factory=Usage)
    _api: object = None

    def key(self, system: str, prompt: str) -> str:
        return _sha(json.dumps({"model": self.model, "system": system, "prompt": prompt,
                                "temperature": self.temperature}, sort_keys=True))

    def _path(self, key: str) -> Path:
        return Path(self.cache_dir) / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if _sha(entry["text"]) != entry["response_sha256"]:
            raise CacheError(f"{path.name}: stored response does not match its sha256")
        if entry["request_sha256"] != key:
            raise CacheError(f"{path.name}: request hash does not match the file name")
        return entry["text"]

    def _write_cache(self, key: str, system: str, prompt: str, text: str) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        entry = {"request_sha256": key, "model": self.model, "temperature": self.temperature,
                 "system": system, "prompt": prompt, "text": text, "response_sha256": _sha(text),
                 "input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens}
        self._path(key).write_text(json.dumps(entry, indent=2), encoding="utf-8")

    def _call_api(self, system: str, prompt: str) -> str:
        if self._api is None:
            import anthropic  # imported late so offline replay needs no SDK or key
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise CacheError("ANTHROPIC_API_KEY is not set and the request is not cached")
            self._api = anthropic.Anthropic()
        msg = self._api.messages.create(model=self.model, max_tokens=self.max_tokens,
                                        temperature=self.temperature, system=system,
                                        messages=[{"role": "user", "content": prompt}])
        self.usage.input_tokens += int(msg.usage.input_tokens)
        self.usage.output_tokens += int(msg.usage.output_tokens)
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    def complete(self, system: str, prompt: str) -> str:
        key = self.key(system, prompt)
        cached = self._read_cache(key)
        if cached is not None:
            self.usage.cache_hits += 1
            return cached
        if self.replay_only:
            raise CacheError(f"no cache entry {key[:12]} for this request (replay-only mode)")
        text = self._call_api(system, prompt)
        self.usage.calls += 1
        self._write_cache(key, system, prompt, text)
        return text

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
