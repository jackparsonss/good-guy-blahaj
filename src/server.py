#!/usr/bin/env python3
import socket
import numpy as np
import torch
import whisper
import struct

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 9031  # Port to listen on (non-privileged ports are > 1023)

model = whisper.load_model("base.en").to("cuda")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = recv_msg(conn)
            print(len(data))
            if not data:
                break
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            results = model.transcribe(audio_np, fp16=True)
            returned_data = results["text"].encode()
            print(returned_data)
            conn.sendall(returned_data)

