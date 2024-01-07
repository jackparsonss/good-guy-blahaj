import speech_recognition as sr
import socket
import struct

# connect to mic
mic_idx = None
for device_index, device_name in sr.Microphone.list_working_microphones().items():
    if device_name.find("Blue Snowball"):
        mic_idx = device_index
    
if mic_idx is None:
    raise ValueError("mic not found")

# set up connection to the orca over socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    HOST = "orca.mami2.moe"
    PORT = 9031
    s.connect((HOST, PORT))

    mic = sr.Microphone(device_index=mic_idx, sample_rate=16000)
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 1000

    def listen_callback(_, audio: sr.AudioData):
        print("sending data")
        print(len(audio.get_raw_data()))
        msg = struct.pack('>I', len(audio.get_raw_data())) + audio.get_raw_data()
        s.sendall(msg)
        data = s.recv(4096)
        print("received", data)

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        
    # spin, listening to messages.
    print("Starting!")
    recognizer.listen_in_background(source=source, phrase_time_limit=1.0, callback=listen_callback)
    
    # don't worry about it, this is bad (makes the listen thread not die, but should be done by joining later or something...)
    while True:
        pass
