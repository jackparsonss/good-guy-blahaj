import speech_recognition as sr
import socket
import struct
import io
import pygame
import threading
import queue
from time import sleep

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

def write_thread():
    while True:
        if not dataqueue.empty():
            audio = dataqueue.get()
            # data to write, so let's do that
            print("sending data")
            print(len(audio.get_wav_data()))
            msg = struct.pack('>I', len(audio.get_wav_data())) + audio.get_wav_data()
            s.sendall(msg)
            data = recv_msg(s)
            playqueue.put(data)
        sleep(0.1)

def play_song():
    while True:
        if not playqueue.empty():
            audio_bytes = playqueue.get()
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            pygame.mixer.init()
            sound.play()
            pygame.time.wait(int(sound.get_length() * 1000))
        sleep(0.1)

# connect to mic
mic_idx = None
for device_index, device_name in sr.Microphone.list_working_microphones().items():
    if device_name.find("Blue Snowball"):
        mic_idx = device_index
    
if mic_idx is None:
    raise ValueError("mic not found")

pygame.init()
mic = sr.Microphone(device_index=mic_idx, sample_rate=16000)
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 1000

dataqueue = queue.Queue()
writethread = threading.Thread(target=write_thread)
writethread.start()

playqueue = queue.Queue()
playthread = threading.Thread(target=play_song)
playthread.start()

# set up connection to the orca over socket
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        HOST = "orca.mami2.moe"
        PORT = 9032
        s.connect((HOST, PORT))
        
        def listen_callback(_, audio: sr.AudioData):
            dataqueue.put(audio)

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            
        # spin, listening to messages.
        print("Starting!")
        die_func = recognizer.listen_in_background(source=source, phrase_time_limit=4.0, callback=listen_callback)
        
        # don't worry about it, this is bad (makes the listen thread not die, but should be done by joining later or something...)
        while True:
            pass
        
except KeyboardInterrupt:
    pygame.quit()