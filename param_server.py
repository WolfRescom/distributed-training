import socket
import pickle
import torch
import torch.nn as nn
from models import SimpleCNN
import struct
import threading

HOST = '127.0.0.1'
PORT = 65432
N_WORKERS = 4

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

class SyncParamServer:
    def __init__(self, n_workers):
        self.model = SimpleCNN()
        self.model.train()
        self.lr = 0.01

        self.n_workers = n_workers
        self.current_round = 0
        self.accumulated_grads = {}

        self.lock = threading.Lock()
        self.round_cond = threading.Condition(self.lock)

        self.round_submissions = {}

    
    def handle_get_weights(self, conn):
        weights = [p.data for p in self.model.parameters()]
        self.send_command(conn, weights)

    def handle_send_gradients(self, conn, round_number, worker_grads):
        with self.lock:
            if round_number not in self.accumulated_grads:
                self.accumulated_grads[round_number] = []
                self.round_submissions[round_number] = 0

            self.accumulated_grads[round_number].append(worker_grads)
            self.round_submissions[round_number] += 1

            if self.round_submissions[round_number] == self.n_workers:
                print(f"** All workers reached round {round_number}, averaging gradients. **")
                self.average_and_apply(round_number)

                self.round_cond.notify_all()
            else:
                while self.round_submissions[round_number] < self.n_workers:
                    self.round_cond.wait()
            
        updated_weights = [p.data for p in self.model.parameters()]
        self.send_command(conn, updated_weights)
    
    def average_and_apply(self, round_number):
        #List of gradients from all workers
        grads_list = self.accumulated_grads[round_number]

        sum_grads = []
        for i, param in enumerate(self.model.parameters()):
            sum_grads.append(torch.zeros_like(param.data))

        #Loops through each workers gradients and adds to the sum
        for gradients in grads_list:
            for i, g in enumerate(gradients):
                sum_grads += g
        
        for i in range(len(sum_grads)):
            sum_grads[i] /= self.n_workers

        apply_gradients(self.model, sum_grads, lr=self.lr)

    def send_command(self, conn, obj):
        message = pickle.dumps(obj)
        data_len = struct.pack('>I', len(message))
        conn.sendall(data_len)
        conn.sendall(message)
        

    def handle_client(self, conn, addr):
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
                
                if isinstance(message, tuple) and len(message) >= 1:
                    command = message[0]
                    if command == "GET_WEIGHTS":
                        self.handle_get_weights(conn)
                    elif command == "SEND_GRADIENTS":
                        round_number = message[1]
                        grads = message[2]
                        self.handle_send_gradients(conn, round_number, grads)
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
    server = SyncParamServer(n_workers=N_WORKERS)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    s.bind((HOST, PORT))
    s.listen()
    print(f"Parameter Server listening on {HOST}:{PORT}")


    while True:
        conn, addr = s.accept()

        new_thread = threading.Thread(target=server.handle_client, args=(conn, addr), daemon=True)
        new_thread.start()

if __name__ == "__main__":
    main()
