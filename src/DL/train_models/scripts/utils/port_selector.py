import socket
import random

def get_free_port():
    while True:
        port = random.randint(0, 65535)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                # Если порт занят, попробуем снова
                continue

