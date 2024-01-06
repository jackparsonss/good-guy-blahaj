import speech_recognition as sr

mic_idx = None
for device_index, device_name in sr.Microphone.list_working_microphones().items():
    if device_name.find("Blue Snowball"):
        mic_idx = device_index
    
if mic_idx is None:
    raise ValueError("mic not found")

mic = sr.Microphone(mic_idx)
recognizer = sr.Recognizer()

with mic as source:
    recognizer.adjust_for_ambient_noise(source)
    print("TALK!")
    data = recognizer.listen(source, phrase_time_limit=1.0)
    print(data)
    print(recognizer.recognize_whisper(data, model="tiny.en"))