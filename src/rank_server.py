#!/usr/bin/env python3
import socket
import requests as rq
import json
import time
import re

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
        response.append({ "original": word, "is_censored": False })

    return json.dumps(response)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        try:
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            while True:
                data = conn.recv(32768).decode('utf-8')

                if data == "":
                    break

                print(f"Data: `{data}`")

                data = {
                    "model": "openorca_hacked:v0.4",
                    "prompt": data,
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
                conn.sendall(str.encode(response, 'utf-8'))
        except BrokenPipeError:
            continue
        finally:
            conn.close()
