import socket
import pickle
import torch
import torch.nn as nn
from models import SimpleCNN
import struct
import threading

HOST = '127.0.0.1'
PORT = 65432

def recvall(conn, num_bytes):
    data = b''
    while len(data) < num_bytes:
        chunk = conn.recv(num_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def apply_gradients(model, grads, lr =0.01):
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param -= lr * grad

def handle_client(conn, addr, model, model_lock, lr = 0.01):
    print(f"[THREAD] Connected by worker at {addr}")

    try:
        while True:
            raw_len = recvall(conn, 4)
            if not raw_len:
                print("No data received; closing connection.")
                conn.close()
                return

            data_len = struct.unpack('>I', raw_len)[0]
            data = recvall(conn, data_len)
            if not data:
                print("No command data received; closing connection.")
                conn.close()
                return
            message = pickle.loads(data)
            command, payload = message
            
            with model_lock:
                if command == "GET_WEIGHTS":
                    weights = [p.data for p in model.parameters()]
                    serialized = pickle.dumps(weights)

                    # Send length first:
                    data_len = struct.pack('>I', len(serialized))
                    conn.sendall(data_len)

                    # Then send the actual data
                    conn.sendall(serialized)
                elif command == "SEND_GRADIENTS":
                    grads = payload
                    print(grads)
                    apply_gradients(model, grads)
                    #Send the weights back
                    weights = [p.data for p in model.parameters()]

                    serialized = pickle.dumps(weights)
                    data_len = struct.pack('>I', len(serialized))  # 4-byte unsigned int in big-endian
                    conn.sendall(data_len)
                    conn.sendall(serialized)
                elif command == "DONE":
                    print(f"[THREAD] Worker at {addr} signaled DONE.")
                    break
                else:
                    print(f"[THREAD] Unknown command: {command}")
                    break
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
    finally:
        conn.close()
        print(f"[THREAD] Disconnected from {addr}")


def main():
    model = SimpleCNN()
    model.train()

    model_lock = threading.Lock()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.bind((HOST, PORT))
    s.listen()
    print(f"Parameter Server listening on {HOST}:{PORT}")


    while True:
        conn, addr = s.accept()

        new_thread = threading.Thread(target=handle_client, args=(conn, addr, model, model_lock), daemon=True)
        new_thread.start()

if __name__ == "__main__":
    main()
