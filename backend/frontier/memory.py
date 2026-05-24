"""
ConversationMemory for frontier assistant.
Supports auto-summarisation to compress context when tokens grow large.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    max_turns: int = 50
    _history: deque = field(default_factory=deque)
    _summary: str = ""

    def __post_init__(self):
        self._history = deque(maxlen=self.max_turns * 2)

    def add_turn(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        history = list(self._history)
        if self._summary:
            # Prepend a synthetic summary turn
            return [
                {"role": "user", "content": f"[CONVERSATION SUMMARY]\n{self._summary}"},
                {"role": "assistant", "content": "Understood. I have the context from our earlier conversation."},
                *history,
            ]
        return history

    def clear(self):
        self._history.clear()
        self._summary = ""

    def token_estimate(self) -> int:
        return sum(len(m["content"]) // 4 for m in self._history)

    async def summarise(self, assistant) -> None:
        """Compress current history into a summary and clear turns."""
        history = list(self._history)
        self._summary = await assistant.summarise(history)
        self._history.clear()
