"""Tests for mjswanApp.

L1 — pure Python, no MuJoCo/ONNX required (safe for pre-commit).
"""

import socket as _socket_module
import socketserver as _socketserver_module
import sys
import threading as _threading_module
import webbrowser as _webbrowser_module
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mjswan.app import _detect_colab, mjswanApp


class _MockTCPServer:
    """No-op stand-in for socketserver.TCPServer."""

    allow_reuse_address = True

    def __init__(self, address, handler):
        self.address = address

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def serve_forever(self):
        pass


@pytest.fixture(autouse=True)
def _patch_tcp_server(monkeypatch):
    monkeypatch.setattr(_socketserver_module, "TCPServer", _MockTCPServer)


@pytest.fixture(autouse=True)
def _patch_socket(monkeypatch):
    # Makes _find_available_port always succeed without touching real ports.
    monkeypatch.setattr(_socket_module, "socket", MagicMock())


@pytest.fixture
def app_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dist"
    d.mkdir()
    return d


class TestDetectColab:
    def test_returns_false_outside_colab(self):
        assert _detect_colab() is False

    def test_returns_true_when_google_colab_importable(self, monkeypatch):
        mock_google_colab = MagicMock()
        monkeypatch.setitem(sys.modules, "google.colab", mock_google_colab)
        assert _detect_colab() is True

    def test_returns_false_after_colab_removed(self, monkeypatch):
        mock_google_colab = MagicMock()
        monkeypatch.setitem(sys.modules, "google.colab", mock_google_colab)
        assert _detect_colab() is True
        monkeypatch.delitem(sys.modules, "google.colab")
        assert _detect_colab() is False


class TestMjswanAppValidation:
    def test_raises_if_app_dir_missing(self, tmp_path):
        with pytest.raises(RuntimeError, match="does not exist"):
            mjswanApp(tmp_path / "nonexistent").launch()


class TestMjswanAppNormalMode:
    def test_opens_browser_by_default(self, app_dir, monkeypatch):
        mock_open = MagicMock()
        monkeypatch.setattr(_webbrowser_module, "open", mock_open)
        mjswanApp(app_dir).launch()
        mock_open.assert_called_once()

    def test_skips_browser_when_open_browser_false(self, app_dir, monkeypatch):
        mock_open = MagicMock()
        monkeypatch.setattr(_webbrowser_module, "open", mock_open)
        mjswanApp(app_dir).launch(open_browser=False)
        mock_open.assert_not_called()

    def test_browser_url_uses_host_and_port(self, app_dir, monkeypatch):
        mock_open = MagicMock()
        monkeypatch.setattr(_webbrowser_module, "open", mock_open)
        mjswanApp(app_dir).launch(host="127.0.0.1", port=9000)
        mock_open.assert_called_once_with("http://127.0.0.1:9000")


class TestMjswanAppColabMode:
    @pytest.fixture
    def mock_colab_output(self, monkeypatch):
        mock_output = MagicMock()
        mock_google_colab = MagicMock()
        mock_google_colab.output = mock_output
        mock_google = MagicMock()
        mock_google.colab = mock_google_colab
        monkeypatch.setitem(sys.modules, "google", mock_google)
        monkeypatch.setitem(sys.modules, "google.colab", mock_google_colab)
        return mock_output

    @pytest.fixture
    def mock_thread(self, monkeypatch):
        thread_instance = MagicMock()
        thread_cls = MagicMock(return_value=thread_instance)
        monkeypatch.setattr(_threading_module, "Thread", thread_cls)
        return thread_cls, thread_instance

    def test_does_not_open_browser(
        self, app_dir, monkeypatch, mock_colab_output, mock_thread
    ):
        mock_open = MagicMock()
        monkeypatch.setattr(_webbrowser_module, "open", mock_open)
        mjswanApp(app_dir).launch()
        mock_open.assert_not_called()

    def test_starts_daemon_thread(self, app_dir, mock_colab_output, mock_thread):
        thread_cls, thread_instance = mock_thread
        mjswanApp(app_dir).launch()
        thread_cls.assert_called_once()
        _, kwargs = thread_cls.call_args
        assert kwargs.get("daemon") is True
        thread_instance.start.assert_called_once()

    def test_calls_serve_kernel_port_as_iframe(
        self, app_dir, mock_colab_output, mock_thread
    ):
        mjswanApp(app_dir).launch(port=8080)
        mock_colab_output.serve_kernel_port_as_iframe.assert_called_once_with(
            8080, height="600"
        )

    def test_respects_height_parameter(self, app_dir, mock_colab_output, mock_thread):
        mjswanApp(app_dir).launch(port=8080, height=800)
        mock_colab_output.serve_kernel_port_as_iframe.assert_called_once_with(
            8080, height="800"
        )

    def test_returns_immediately(self, app_dir, mock_colab_output, mock_thread):
        result = mjswanApp(app_dir).launch()
        assert result is None
