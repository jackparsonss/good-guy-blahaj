import speech_recognition as sr
import queue

mic_idx = None
for device_index, device_name in sr.Microphone.list_working_microphones().items():
    if device_name.find("Blue Snowball"):
        mic_idx = device_index
    
if mic_idx is None:
    raise ValueError("mic not found")

data_queue = queue.Queue()
mic = sr.Microphone(mic_idx)
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False
recognizer.energy_threshold = 1000

def listen_callback(_, audio: sr.AudioData):
    data_queue.put(audio)

with mic as source:
    recognizer.adjust_for_ambient_noise(source)
    
print("Starting!")
recognizer.listen_in_background(source=source, phrase_time_limit=1.0, callback=listen_callback)

while True:
    if not data_queue.empty():
        while not data_queue.empty():
            data = data_queue.get()
            text = recognizer.recognize_whisper(data, model="medium.en")
            print(text)
    
    
