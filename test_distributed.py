# test_distributed.py
import subprocess
import time
import os

def main():
    # Number of workers and how we partition MNIST (60,000 training samples)
    N_WORKERS = 4
    total_samples = 4000
    chunk_size = total_samples // N_WORKERS

    # 1) Start the parameter server
    print("[Test] Starting param_server.py ...")
    server_proc = subprocess.Popen(["python", "param_server.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Give the server a moment to start listening
    time.sleep(2)

    # 2) Launch each worker
    worker_procs = []
    for w_id in range(N_WORKERS):
        start_idx = w_id * chunk_size
        # Last worker might get the remainder if total_samples not divisible
        end_idx = (w_id + 1) * chunk_size if w_id < N_WORKERS - 1 else total_samples

        print(f"[Test] Launching worker {w_id} with data range [{start_idx}, {end_idx})")
        worker_cmd = ["python", "worker.py", str(start_idx), str(end_idx)]
        worker_proc = subprocess.Popen(worker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        worker_procs.append(worker_proc)

    # 3) Collect output from each worker
    for i, wproc in enumerate(worker_procs):
        out, _ = wproc.communicate()  # Wait for worker to finish
        print(f"=== Worker {i} output ===")
        print(out)

    # 4) Optionally, we can kill or wait on the server once all workers are done
    # The server might still be running if it's in an infinite accept loop.
    # We'll terminate it for this test.
    print("[Test] All workers completed. Terminating server.")
    #server_proc.terminate()

    # Print server output
    server_out, _ = server_proc.communicate()
    print("=== Server output ===")
    print(server_out)

if __name__ == "__main__":
    main()
