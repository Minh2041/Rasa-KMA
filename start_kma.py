
import subprocess
import time
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RASA = ROOT / ".venv" / "Scripts" / "python.exe"
RAG = ROOT / ".venv-rag" / "Scripts" / "python.exe"

processes = []
logs = []

def start_background(command, filename):
    logfile = open(ROOT / filename, "a", encoding="utf-8")
    logs.append(logfile)

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=logfile,
        stderr=subprocess.STDOUT
    )
    processes.append(process)
    return process

def wait_for_port(port, process, seconds=90):
    for _ in range(seconds * 2):
        if process.poll() is not None:
            raise RuntimeError(f"Dich vu cong {port} da dung")

        try:
            with socket.create_connection(
                ("127.0.0.1", port), timeout=1
            ):
                return
        except OSError:
            time.sleep(0.5)

    raise RuntimeError(f"Cong {port} khong san sang")

try:
    with socket.create_connection(
        ("127.0.0.1", 11434), timeout=3
    ):
        print("Ollama dang hoat dong.")

    print("Khoi dong RAGLight API...")
    rag_process = start_background([
        str(RAG), "-m", "uvicorn",
        "rag_api:app",
        "--host", "127.0.0.1",
        "--port", "8001"
    ], "rag_api.log")
    wait_for_port(8001, rag_process)

    print("Khoi dong Rasa Action Server...")
    action_process = start_background([
        str(RASA), "-m", "rasa_sdk",
        "--actions", "actions",
        "--port", "5055"
    ], "rasa_actions.log")
    wait_for_port(5055, action_process)

    print("Khoi dong Rasa Shell...")
    subprocess.run([
        str(RASA), "-m", "rasa",
        "shell", "--endpoints", "endpoints.yml"
    ], cwd=ROOT)

except OSError as exc:
    print("Loi: Hay kiem tra Ollama, Python va cong.", exc)
except RuntimeError as exc:
    print("Loi khoi dong:", exc)

finally:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    for logfile in logs:
        logfile.close()

    print("Da dung cac dich vu do script khoi dong.")

