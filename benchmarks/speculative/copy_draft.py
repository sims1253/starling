"""Offline, token-exact copy-draft acceptance probe for issue #312.

The proposer sees only source tokens and the already emitted output prefix.
``simulate`` uses recorded greedy target IDs as an oracle for acceptance; it
does not execute a model or estimate wall time or energy. This keeps proposal
quality separate from the still pending native verifier (#311).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


class CopyDrafter:
    def __init__(self, source: Sequence[int], *, max_skip: int = 32) -> None:
        self.source = list(source)
        self.position = 0
        self.max_skip = max_skip
        self.misses = 0

    def observe(self, emitted: int) -> None:
        """Advance past source tokens omitted by the target when it realigns."""
        end = min(len(self.source), self.position + self.max_skip + 1)
        for i in range(self.position, end):
            if self.source[i] == emitted:
                self.position = i + 1
                self.misses = 0
                return
        self.misses += 1

    @staticmethod
    def _lookup(prefix: Sequence[int], k: int) -> list[int]:
        """Use the most recent matching output n-gram when copying stalls."""
        for n in range(min(4, len(prefix)), 0, -1):
            needle = prefix[-n:]
            for i in range(len(prefix) - n - 1, -1, -1):
                if prefix[i:i + n] == needle:
                    return list(prefix[i + n:i + n + k])
        return []

    def propose(self, prefix: Sequence[int], k: int) -> list[int]:
        if k < 1:
            raise ValueError("k must be positive")
        if self.misses >= 3 or self.position >= len(self.source):
            return self._lookup(prefix, k)
        return self.source[self.position:self.position + k]


@dataclass(frozen=True)
class Result:
    output_tokens: int
    verify_passes: int
    drafted_tokens: int
    accepted_tokens: int
    full_accept_passes: int
    passes_without_draft: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "accepted_per_pass": self.accepted_tokens / self.verify_passes,
            "output_tokens_per_pass": self.output_tokens / self.verify_passes,
            "draft_acceptance_rate": (
                self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0
            ),
        }


def simulate(source: Sequence[int], target: Sequence[int], *, max_k: int = 2,
             max_skip: int = 32) -> Result:
    """Replay a target's greedy ID stream without showing future IDs to drafter."""
    if not target or max_k < 1:
        raise ValueError("target must be nonempty and max_k positive")
    drafter = CopyDrafter(source, max_skip=max_skip)
    output: list[int] = []
    passes = drafted = accepted_total = full = empty = 0
    k = min(2, max_k)
    while len(output) < len(target):
        draft = drafter.propose(output, k)
        passes += 1
        drafted += len(draft)
        if not draft:
            empty += 1
        accepted = 0
        remaining = len(target) - len(output)
        for token in draft[:remaining]:
            if token != target[len(output) + accepted]:
                break
            accepted += 1
        accepted_total += accepted
        # Only the verified part of a draft can be accepted on the last pass.
        verified = len(draft[:remaining])
        if draft and accepted == verified:
            full += 1
            k = min(max_k, k + 1)
        elif draft:
            k = max(1, k // 2)
        # A verification pass emits the accepted prefix and, if available,
        # the target's first non-draft token (the verifier's bonus token).
        count = min(remaining, accepted + 1)
        for token in target[len(output):len(output) + count]:
            output.append(token)
            drafter.observe(token)
    if output != list(target):
        for index, (got, want) in enumerate(zip(output, target)):
            if got != want:
                raise ValueError(f"replay diverged at index {index}: got {got}, want {want}")
        raise ValueError(f"replay length mismatch: got {len(output)}, want {len(target)}")
    return Result(len(output), passes, drafted, accepted_total, full, empty)
