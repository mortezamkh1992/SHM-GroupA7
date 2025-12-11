import os, sys, time, subprocess

RUNTIME_LIMIT_SEC = 3600
SLEEP_POLL_SEC = 5

CWD = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(CWD, "VAE.py")
LOCK = os.path.join(CWD, ".vae_lock")
PYTHON = sys.executable

while True:
    # ensure no stale lock blocks the new run
    if os.path.exists(LOCK):
        try:
            os.remove(LOCK)
        except OSError:
            pass

    print("Starting VAE.py")
    proc = subprocess.Popen([PYTHON, SCRIPT], cwd=CWD)

    start = time.time()
    while True:
        ret = proc.poll()
        if ret is not None:
            print(f"VAE.py exited with code {ret}. Restarting in 5 s...")
            time.sleep(5)
            break

        if time.time() - start >= RUNTIME_LIMIT_SEC:
            print("Time limit reached, restarting VAE.py...")
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print("Terminate timed out, killing process...")
                proc.kill()
                proc.wait()
            time.sleep(3)
            break

        time.sleep(SLEEP_POLL_SEC)