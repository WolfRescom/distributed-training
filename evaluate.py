# evaluate.py

import socket
import pickle
import struct
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import SimpleCNN
import torch.nn.functional as F

HOST = '127.0.0.1'
PORT = 65432

def recvall(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def send_command(sock, command, payload):
    message = pickle.dumps((command, payload))
    data_len = struct.pack('>I', len(message))
    sock.sendall(data_len)
    sock.sendall(message)

def receive_pickled(sock):
    raw_len = recvall(sock, 4)
    if not raw_len:
        return None
    resp_len = struct.unpack('>I', raw_len)[0]
    data = recvall(sock, resp_len)
    if data is None:
        return None
    return pickle.loads(data)

def set_model_weights(model, weights):
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(w)

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs, y, reduction='sum')
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy

def main():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))

        # Get the final weights from server
        send_command(sock, "GET_WEIGHTS", None)
        final_weights = receive_pickled(sock)
        if final_weights is None:
            raise RuntimeError("Failed to receive final weights.")

        set_model_weights(model, final_weights)

        # Evaluate
        test_loss, test_acc = evaluate_model(model, test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        # Optionally send "DONE" if you want to close the connection gracefully
        send_command(sock, "DONE", None)

if __name__ == "__main__":
    main()
