"""mjswanApp class for exporting and running applications.

This module defines the mjswanApp class which represents a built application
that can be saved to disk or launched in a web browser.
"""

from __future__ import annotations

from pathlib import Path


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


class mjswanApp:
    """A built mjswan application ready to be launched.

    This class encapsulates the built application and provides methods
    for launching it in a web browser.
    """

    def __init__(self, app_dir: Path) -> None:
        self._app_dir = app_dir

    def launch(
        self,
        *,
        host: str = "localhost",
        port: int = 8080,
        open_browser: bool = True,
        height: int = 600,
    ) -> None:
        """Launch the application in a local web server.

        Automatically detects Google Colab and displays the viewer as an
        inline iframe. Outside Colab, starts a blocking server and optionally
        opens a browser tab.

        Args:
            host: Host to bind the server to (ignored in Colab).
            port: Port to run the server on.
            open_browser: Whether to automatically open a browser (ignored in Colab).
            height: Height of the Colab iframe in pixels (ignored outside Colab).
        """
        if not self._app_dir.exists():
            raise RuntimeError(f"Application directory {self._app_dir} does not exist.")

        import http.server
        import socket
        import socketserver

        directory = str(self._app_dir)

        class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
            """HTTP handler with Cross-Origin Isolation headers for SharedArrayBuffer."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def end_headers(self):
                # Required for SharedArrayBuffer (used by MuJoCo WASM threading)
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                super().end_headers()

        handler = CrossOriginIsolatedHandler

        def _find_available_port(
            bind_host: str, start_port: int, max_tries: int = 1000
        ) -> int:
            port_try = start_port
            tries = 0
            while tries < max_tries and port_try <= 65535:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind((bind_host, port_try))
                        return port_try
                    except OSError:
                        port_try += 1
                        tries += 1
            raise RuntimeError(f"No available port found starting at {start_port}")

        class _ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        if _detect_colab():
            bind_host = ""
            chosen_port = _find_available_port(bind_host, port)
            if chosen_port != port:
                print(f"Port {port} unavailable — using port {chosen_port} instead.")
            port = chosen_port

            import threading

            def _serve():
                with _ReusableTCPServer((bind_host, port), handler) as httpd:
                    httpd.serve_forever()

            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            print(f"Server running on port {port}")

            from google.colab import output  # type: ignore[import]

            output.serve_kernel_port_as_iframe(port, height=str(height))
            return

        import webbrowser

        chosen_port = _find_available_port(host, port)
        if chosen_port != port:
            print(f"Port {port} unavailable — using port {chosen_port} instead.")
        port = chosen_port

        print(f"Starting server at http://{host}:{port}")
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")

        try:
            with _ReusableTCPServer((host, port), handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


__all__ = ["mjswanApp"]
