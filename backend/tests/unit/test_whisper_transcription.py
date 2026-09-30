import threading
import time
from types import SimpleNamespace

from src.media import transcription


class RecordingModel:
    """Stands in for a Whisper model and records overlapping transcribe calls."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self._guard = threading.Lock()

    def transcribe(self, audio_path, **kwargs):
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self._guard:
            self.active -= 1
        return {"text": "", "segments": []}


def run_concurrently(target, count=4):
    threads = [threading.Thread(target=target, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_concurrent_jobs_transcribe_one_at_a_time_on_one_model(monkeypatch, tmp_path):
    model = RecordingModel()
    loads = []

    def load_model(name):
        loads.append(name)
        time.sleep(0.05)  # widen the window for a duplicate load
        return model

    monkeypatch.setattr(transcription, "_WHISPER_AVAILABLE", True)
    monkeypatch.setattr(transcription, "_WHISPER_MODEL_CACHE", {})
    monkeypatch.setattr(transcription, "_whisper", SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(transcription, "_prepare_audio_for_transcription", lambda path: path)

    run_concurrently(
        lambda i: transcription.transcribe_with_whisper(tmp_path / f"{i}.mp4", "turbo")
    )

    assert model.peak == 1
    assert loads == ["turbo"]
