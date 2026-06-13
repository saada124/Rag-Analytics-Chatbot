import threading
from collections import deque, OrderedDict

from config import MEMORY_SIZE


class ConversationMemory:
    """Thread-safe rolling conversation history for a single session."""

    def __init__(self, max_turns: int = MEMORY_SIZE):
        # Each turn is a (user, assistant) pair -> store 2 messages per turn.
        self._messages = deque(maxlen=max(1, max_turns) * 2)
        self._lock = threading.Lock()

    def add_user(self, text: str) -> None:
        with self._lock:
            self._messages.append(("User", (text or "").strip()))

    def add_assistant(self, text: str) -> None:
        with self._lock:
            self._messages.append(("Assistant", (text or "").strip()))

    def get_history(self) -> str:
        with self._lock:
            return "\n".join(f"{role}: {content}" for role, content in self._messages)

    def clear(self) -> None:
        with self._lock:
            self._messages.clear()


class SessionManager:
    """Creates and stores one ConversationMemory per session id (LRU bounded)."""

    def __init__(self, max_sessions: int = 1000):
        self._sessions: "OrderedDict[str, ConversationMemory]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def get_memory(self, session_id: str = None) -> ConversationMemory:
        sid = session_id or "default"
        with self._lock:
            mem = self._sessions.get(sid)
            if mem is not None:
                self._sessions.move_to_end(sid)
                return mem

            mem = ConversationMemory()
            self._sessions[sid] = mem
            if len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return mem

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id or "default", None)

session_manager = SessionManager()
