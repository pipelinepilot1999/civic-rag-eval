"""Regression test for the retry bug that killed a 41,072-variant annotation run.

ConnectionResetError is an OSError but NOT a urllib.error.URLError. Catching
only URLError looked correct, passed every short test, and then died 85% of the
way through a 45-minute job.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import annotate as annotate_module  # noqa: E402


def test_connection_reset_is_retried(monkeypatch):
    calls = {"n": 0}

    def flaky(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionResetError(54, "Connection reset by peer")
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return b"[]"
        return Response()

    monkeypatch.setattr(annotate_module.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(annotate_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(annotate_module.json, "load", lambda _: [])

    assert annotate_module._post(["chr7:g.1A>T"]) == []
    assert calls["n"] == 3, "ConnectionResetError must be retried, not raised"


def test_retry_gives_up_eventually(monkeypatch):
    def always_fails(request, timeout=None):
        raise ConnectionResetError(54, "Connection reset by peer")

    monkeypatch.setattr(annotate_module.urllib.request, "urlopen", always_fails)
    monkeypatch.setattr(annotate_module.time, "sleep", lambda _: None)
    with pytest.raises(ConnectionResetError):
        annotate_module._post(["chr7:g.1A>T"])
