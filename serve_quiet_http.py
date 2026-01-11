#!/usr/bin/env python3
"""serve_quiet_http.py

Run: python3 serve_quiet_http.py --dir ~/www --port 8080

Cel:
- Zamiennik dla: python3 -m http.server 8080
- Tłumi typowe BrokenPipeError/ConnectionResetError gdy przeglądarka zrywa połączenie
  (np. odświeżanie obrazka co 1s) — to normalne, ale standardowy http.server wypisuje traceback.
"""

from __future__ import annotations

import argparse
import os
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class QuietHandler(SimpleHTTPRequestHandler):
    # Nie spamuj na stdout standardowymi logami GET (opcjonalnie)
    def log_message(self, format: str, *args) -> None:  # noqa: A003
        if getattr(self.server, "quiet", False):
            return
        super().log_message(format, *args)

    def copyfile(self, source, outputfile) -> None:
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            # Klient (browser) przerwał pobieranie — normalne przy częstym odświeżaniu.
            return

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, socket.error):
            return

    def log_error(self, format: str, *args) -> None:  # noqa: A003
        # Nie wypisuj tracebacków dla typowych zerwań połączenia.
        if getattr(self.server, "quiet_errors", False):
            return
        super().log_error(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path.cwd(), help="Katalog do serwowania")
    parser.add_argument("--host", default="0.0.0.0", help="Adres bind (domyślnie 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (domyślnie 8080)")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Wyłącz standardowe logi GET (zostają tylko błędy krytyczne)",
    )
    parser.add_argument(
        "--quiet-errors",
        action="store_true",
        help="Tłum błędy/tracebacki (poza krytycznymi), w tym BrokenPipe",
    )
    args = parser.parse_args()

    directory = args.dir.expanduser().resolve()
    if not directory.exists() or not directory.is_dir():
        raise SystemExit(f"Brak katalogu: {directory}")

    os.chdir(directory)

    httpd = ThreadingHTTPServer((args.host, args.port), QuietHandler)
    httpd.quiet = bool(args.quiet)
    httpd.quiet_errors = bool(args.quiet_errors)

    print(f"[INFO] Serving: {directory}")
    print(f"[INFO] URL: http://{args.host}:{args.port}/")
    print("[INFO] Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
