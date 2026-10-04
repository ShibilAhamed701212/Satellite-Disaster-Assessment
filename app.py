#!/usr/bin/env python3
"""
Satellite Disaster Assessment & Analysis Dashboard.

Launch entrypoint from repository root:
    python app.py
"""

import os
import sys

# Configure UTF-8 stdout/stderr for Windows console compatibility
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, "reconfigure", None)
        if callable(_reconf):
            try:
                _reconf(encoding="utf-8")
            except Exception:
                pass

# Ensure DL-SatelliteImagery is in python path
_dl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL-SatelliteImagery")
if _dl_path not in sys.path:
    sys.path.insert(0, _dl_path)

try:
    from disaster_gradio_app import create_app
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("disaster_gradio_app", os.path.join(_dl_path, "disaster_gradio_app.py"))
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["disaster_gradio_app"] = _mod
        _spec.loader.exec_module(_mod)
        create_app = getattr(_mod, "create_app")
    else:
        raise
except OSError as e:
    if "DLL" in str(e) or "c10" in str(e) or "1114" in str(e):
        print("\n[Environment Error] PyTorch dynamic library failed to load in current Python environment.")
        print(f"Details: {e}")
        print("\nPlease run this application using the project's dedicated virtual environment:")
        print("  PowerShell (Windows):  .\\.venv\\Scripts\\python.exe app.py")
        print("  Or activate venv:      .\\.venv\\Scripts\\Activate.ps1; python app.py\n")
        sys.exit(1)
    raise


def main():
    app = create_app()
    import socket

    def get_open_port(default_port=7860):
        for port in range(default_port, default_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        return None

    selected_port = get_open_port(7860) or 7860
    print(f"🛰️ Launching Satellite Disaster Assessment Dashboard on port {selected_port}...")
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=selected_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
