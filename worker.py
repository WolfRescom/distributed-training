import socket
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import SimpleCNN
import struct

HOST = '127.0.0.1'
PORT = 65432

def recvall(sock, num_bytes):
    """Helper function to receive exactly num_bytes from the socket."""
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            # Connection closed or lost in the middle
            return None
        data += chunk
    return data

def send_command(sock, command, payload):
    """
    Generic function to send a command + payload with a 4-byte length prefix.
    Returns any data received as a response (the server might or might not send it).
    """
    message = pickle.dumps((command, payload))
    data_len = struct.pack('>I', len(message))
    sock.sendall(data_len)
    sock.sendall(message)

def receive_pickled(sock):
    """Receive a length-prefixed pickle object from the server."""
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    resp_len = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, resp_len)
    if data is None:
        return None
    return pickle.loads(data)

def get_weights_from_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        #Send the GET_WEIGHTS command
        message = pickle.dumps(("GET_WEIGHTS", None))
        data_len = struct.pack('>I', len(message))
        sock.sendall(data_len)
        sock.sendall(message)

        #Receive the weights
        raw_len = recvall(sock, 4)
        data_len = struct.unpack('>I', raw_len)[0]  # convert bytes to int
        # --- 2) Read the actual serialized data of length data_len ---
        data = recvall(sock, data_len)
        if data is None:
            raise RuntimeError("Connection lost while receiving data.")
        weights = pickle.loads(data)
    return weights

def send_gradients_to_server(grads):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        #Send the SEND_GRADIENTS command
        message = pickle.dumps(("SEND_GRADIENTS", grads))
        data_len = struct.pack('>I', len(message))
        sock.sendall(data_len)
        sock.sendall(message)


        #Receive the weights
        raw_len = recvall(sock, 4)
        data_len = struct.unpack('>I', raw_len)[0]  # convert bytes to int
        # --- 2) Read the actual serialized data of length data_len ---
        data = recvall(sock, data_len)
        if data is None:
            raise RuntimeError("Connection lost while receiving data.")
        
        updated_weights  = pickle.loads(data)
    return updated_weights

def set_model_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(w)

def compute_gradients(model, data_loader, device='cpu'):
    #Single pass over data
    model.train()
    grads = []
    
    for param in model.parameters():
        param.grad = None  # reset gradients

    for (x, y) in data_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
    grads = [param.grad.clone() for param in model.parameters()]
    return grads

def main():
    # 1. Prepare local data (e.g., subset of MNIST)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    subset = torch.utils.data.Subset(train_dataset, range(0,1024)) #Each worker should have a different range of data
    data_loader = DataLoader(subset, batch_size=32, shuffle=True)

    model = SimpleCNN()

    num_epochs = 10

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        for epoch in range(num_epochs):
            print(f"[Worker] Starting epoch {epoch+1}/{num_epochs}...")
            send_command(sock, "GET_WEIGHTS", None)
            weights = receive_pickled(sock)
            if weights is None:
                raise RuntimeError("Failed to receive weights from server.")
            
            set_model_weights(model, weights)
            grads = compute_gradients(model, data_loader)

            send_command(sock, "SEND_GRADIENTS", grads)
            updated_weights = receive_pickled(sock)
            set_model_weights(model, updated_weights)

            print(f"[Worker] Epoch {epoch+1} complete. Model updated.")
        send_command(sock, "DONE", None)
    
    print("[Worker] Training complete. Connection closed.")

if __name__ == "__main__":
    main()
