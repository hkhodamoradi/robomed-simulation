# speech_announcer.py
import shutil
import subprocess
import threading
import time


class SpeechAnnouncer:
    def __init__(self, enabled=True, rate=165, gap_sec=0.4):
        self.enabled = enabled
        self.rate = rate
        self.gap_sec = gap_sec
        self._last_text = None
        self._last_time = 0.0
        self._lock = threading.Lock()
        self._engine = shutil.which("espeak")

        if self.enabled and self._engine is None:
            print("[speech] espeak not found, speech disabled")
            self.enabled = False

    def say(self, text: str, force: bool = False):
        """Non-blocking speech. Deduplicates repeated announcements."""
        if not self.enabled or not text:
            return

        now = time.time()
        with self._lock:
            if not force:
                if text == self._last_text and (now - self._last_time) < 2.0:
                    return
                if (now - self._last_time) < self.gap_sec:
                    return
            self._last_text = text
            self._last_time = now

        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text: str):
        try:
            subprocess.Popen(
                [self._engine, "-s", str(self.rate), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print(f"[speech] failed: {e}")