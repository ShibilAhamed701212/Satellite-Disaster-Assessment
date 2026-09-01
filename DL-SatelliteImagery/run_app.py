"""Launcher for the Satellite Disaster Assessment Gradio app.

Pre-imports torch before the main app to avoid Windows DLL initialization
errors (WinError 1114 on c10.dll) that occur when torch is imported as
part of a script file's module resolution.
"""
import torch  # noqa: F401 — must be imported before the app module

from disaster_gradio_app import create_app  # noqa: E402

if __name__ == "__main__":
    import socket

    def get_open_port(default_port=7860):
        for port in range(default_port, default_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return port
        return default_port

    port = get_open_port(7860)
    print(f"Starting Satellite Disaster Assessment on http://127.0.0.1:{port}")
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )
