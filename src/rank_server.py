#!/usr/bin/env python3
import socket
import requests as rq
import json
import time
import re
import random

import numpy as np
import torch
import whisper
import struct

NUM_ROUNDS = 5

HOST = "127.0.0.1"
PORT = 9033

headers = { "Content-Type": "application/x-www-form-urlencoded" }

def query_model(data):
    response = rq.post("http://localhost:11434/api/generate", headers=headers, data=data)
    json_version = re.sub(r"\n(?=.)", ",", response.text)
    json_version = f"[{json_version}]"
    json_version = re.sub("\n", "", json_version)
    js = json.loads(json_version)
    words = "".join(word["response"] for word in js).strip()
    return words


def get_max(ranks):
    max_count = 0
    max_words = ""

    for w, i in ranks.items():
        if i > max_count:
            max_count = i
            max_words = w

    print(f"Max {max_count}: {max_words}")
    return max_words


def encode_to_backed(max_words):
    response = list()

    for word in max_words.split():
        response.append({ "original": word, "is_censored": random.random() < 0.3 })

    return json.dumps(response)


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


############################################################
# Main
############################################################
model = whisper.load_model("large").to("cuda")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        try:
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            while True:
                data = recv_msg(conn)

                if len(data) == 0:
                    break

                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                results = model.transcribe(audio_np, fp16=True)
                transcribed_text = results["text"]

                og_audio = data

                ##########################
                # Prompting
                ##########################
                data = {
                    "model": "openorca_hacked:v0.4",
                    "prompt": transcribed_text,
                }
                data = json.dumps(data)

                ranks = dict()
                for _ in range(NUM_ROUNDS):
                    words = query_model(data)

                    if ranks.get(words) is not None:
                        ranks[words] += 1
                    else:
                        ranks[words] = 1

                max_words = get_max(ranks)
                response = encode_to_backed(max_words)
                response = response.encode('utf-8')

                msg = (
                    struct.pack('>I', len(data))
                    + struct.pack('>I', len(response))
                    + data
                    + response
                )

                conn.sendall(msg)
        except BrokenPipeError:
            continue
        finally:
            conn.close()
