import pvcheetah
import pvrecorder
import queue

mic_idx = None
for device_index, device_name in enumerate(pvrecorder.PvRecorder.get_available_devices()):
    if device_name.find("Blue Snowball"):
        mic_idx = device_index
    
if mic_idx is None:
    raise ValueError("mic not found")

data_queue = queue.Queue()
handle = pvcheetah.create('', endpoint_duration_sec=0.5)
recorder = pvrecorder.PvRecorder(frame_length=handle.frame_length, device_index=mic_idx)
recorder.start()
print("Starting!")


while True:
    partial_transcript, is_endpoint = handle.process(recorder.read())
    if partial_transcript != '':
        print(partial_transcript)  
    if is_endpoint:
        print(handle.flush())
        # final_transcript = handle.flush()
        #break
        
