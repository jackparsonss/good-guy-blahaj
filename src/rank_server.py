#!/usr/bin/env python3
import socket
import requests as rq
import json
import time
import re
import inflect
import difflib
import random

import numpy as np
import torch
import whisper
import struct
import threading

import torchaudio
import torchaudio.functional as F
import wave
import io
import pydub
from pydub import AudioSegment

NUM_ROUNDS = 5

HOST = "127.0.0.1"
PORT = 9032
PORT_WEB = 9033

WEB_RESPONSE = "[]".encode('utf-8')

headers = { "Content-Type": "application/x-www-form-urlencoded" }

def pluralize(w):
    if not inflect.singular_noun(w):
        return (w, inflect.plural(w))
    else:
        return (inflect.singular_noun(w), w)


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


def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    alignments, scores = F.forced_align(emission, targets, blank=0)
    alignments, scores = alignments[0], scores[0]
    scores = scores.exp()
    return alignments, scores


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def annotated_to_string(words_annote):
    s = list()

    for word in words_annote:
        if word[1]:
            s.append(word[0])
        else:
            s.append("#BEEP#")

    return " ".join(s)


def get_word_starts(words):
    words_index = list()

    running_sum = 0
    for word in words.split():
        words_index.append([running_sum, word, False])
        running_sum += len(word) + 1

    return words_index


def to_running_index(words_index):
    running_index = [-1 for _ in range(words_index[-1][0] + len(words_index[-1][1]))]

    for i in range(len(running_index)):
        for ii, word in enumerate(words_index):
            start = word[0]
            cease = word[0] + len(word[1]) - 1

            if start <= i <= cease:
                running_index[i] = ii

    return running_index

def word_starts_to_string(word_starts):
    s = list()

    for word in word_starts:
        if word[2]:
            s.append(word[1])
        else:
            s.append("#BEEP#")

    return " ".join(s)

def sequence_match(lhs, rhs, word_starts, word_run_idx):
    matcher = difflib.SequenceMatcher(None, lhs, rhs)

    for x in matcher.get_matching_blocks()[:-1]:
        i, j, n = x

        while lhs[i] == " ":
            i += 1
            j += 1
            n -= 1

        for x in range(n):
            x += i
            print(x, len(word_run_idx), lhs[i:i+n], rhs[j:j+n], word_starts)
            index = word_run_idx[x]

            if index != -1:
                word_starts[index][2] = True

    return word_starts


def censor_audio(audio, beep_wav, word_spans, word_annotes, ratio, sample_rate):
    prev_rm_word_idx = -1
    audio_buffer = audio[:0]

    new_audio = audio[:]

    for i in range(len(word_annotes)):
        if not word_annotes[i][1]:  # Don't include this word
            start = 1000 * int(word_spans[i][0].start * ratio) / sample_rate
            cease = 1000 * int(word_spans[i][-1].end * ratio) / sample_rate

            new_audio = new_audio[:start] + beep_wav[:cease-start] + new_audio[cease:]

    return new_audio


def web_backend_thread():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT_WEB))
        s.listen()

        while True:
            try:
                conn, addr = s.accept()

                while True:
                    data = conn.recv(1024)

                    if not data:
                        break
                    conn.sendall(WEB_RESPONSE)
            except BrokenPipeError:
                continue
            finally:
                conn.close()


############################################################
# Main
############################################################
testing_audio = None

##########################
# Startup Constants
##########################
model = whisper.load_model("medium.en").to("cuda")
inflect = inflect.engine()
beep_sound = AudioSegment.from_wav("beep1.wav")

bundle = torchaudio.pipelines.MMS_FA
align_model = bundle.get_model(with_star=False).to("cuda")
LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)

censor_words = list()
with open("bad-words.csv", "r") as f:
    for line in f.readlines():
        word = line.strip()
        censor_words.append(pluralize(word)[0])
        censor_words.append(pluralize(word)[1])

web_thread = threading.Thread(target=web_backend_thread)
web_thread.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    print("Socket inbound!")

    while True:
        try:
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            while True:
                og_audio = recv_msg(conn)

                if not og_audio:
                    break

                audio_np = np.frombuffer(og_audio, dtype=np.int16).astype(np.float32) / 32768.0
                results = model.transcribe(audio_np, fp16=True)

                og_transcribed_text = results["text"].strip().lower()
                og_transcribed_text = re.sub("[^a-z ]", "", og_transcribed_text)
                og_transcribed_text = " ".join(og_transcribed_text.split())

                ##########################
                # Regex Filtering
                ##########################
                words_annote = [[word, True] for word in og_transcribed_text.split()]

                for i in range(len(words_annote)):
                    for word in censor_words:
                        if re.match(word, words_annote[i][0], flags=re.IGNORECASE):
                            words_annote[i][1] = False

                # Alignment (used for audio alignment later)
                #removed_words = re.sub("#BEEP#", "", processed_words)
                #word_starts = get_word_starts(og_transcribed_text)
                #print(word_starts)
                #word_run_idx = to_running_index(word_starts)
                #print(word_run_idx)
                #word_starts = sequence_match(og_transcribed_text, removed_words, word_starts, word_run_idx)
                print(json.dumps(words_annote, indent=4))
                print(annotated_to_string(words_annote))


                ##########################
                # Prompting
                ##########################
                data = {
                    "model": "openorca_hacked:v0.7",
                    "prompt": og_transcribed_text,
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

                WEB_RESPONSE = response

                print("==== INPUT =============")
                print(og_transcribed_text)
                print("==== REGEX =============")
                print(annotated_to_string(words_annote))
                print("==== LLAMA =============")
                print(response.decode('utf-8'))
                print("========================")

                ##########################
                # Chop up
                ##########################
                waveform, _ = torchaudio.load(io.BytesIO(og_audio))
                with torch.inference_mode():
                    emission, _ = align_model(waveform.to("cuda"))

                tokenized_transcript = [DICTIONARY[c] for word in og_transcribed_text.split() for c in word]
                aligned_tokens, alignment_scores = align(emission, tokenized_transcript)
                token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
                word_spans = unflatten(token_spans, [len(word) for word in og_transcribed_text.split()])

                ratio = waveform.size(1) / emission.size(1)
                audio = AudioSegment.from_wav(io.BytesIO(og_audio))

                new_audio = censor_audio(audio, beep_sound, word_spans, words_annote, ratio, bundle.sample_rate)
                if testing_audio is None:
                    testing_audio = new_audio
                else:
                    testing_audio += new_audio

                ##########################
                # Respond
                ##########################
                msg = (
                    struct.pack('>I', len(og_audio))
                    + og_audio
                )

                conn.sendall(msg)
        except BrokenPipeError:
            continue
        finally:
            conn.close()
            testing_audio.export("out.wav", format="wav")
