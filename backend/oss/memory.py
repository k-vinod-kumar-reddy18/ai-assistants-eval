"""
ConversationMemory — rolling window of the last N turns.
Swap the in-memory dict for Redis in production:
  import redis; r = redis.Redis(); r.lpush(session_id, json.dumps(turn))
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    max_turns: int = 10
    _history: deque = field(default_factory=deque)

    def __post_init__(self):
        self._history = deque(maxlen=self.max_turns * 2)  # user + assistant per turn

    def add_turn(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return list(self._history)

    def clear(self):
        self._history.clear()

    def token_estimate(self) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        return sum(len(m["content"]) // 4 for m in self._history)
