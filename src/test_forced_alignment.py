#!/usr/bin/env python3
import socket
import numpy as np
import torch
import whisper
import struct
import torchaudio
import torchaudio.functional as F
import io 
import wave
import pydub
from pydub import AudioSegment

def align(emission, tokens):
    targets = torch.tensor([tokens], dtype=torch.int32, device="cuda")
    alignments, scores = F.forced_align(emission, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    scores = scores.exp()  # convert back to probability
    return alignments, scores

def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret

waveform, _ = torchaudio.load("test_audio_16khz.wav")
transcript = "shawty got them apple bottom jeans boots with the fur".split()
bundle = torchaudio.pipelines.MMS_FA
model = bundle.get_model(with_star=False).to("cuda")
with torch.inference_mode():
    emission, _ = model(waveform.to("cuda"))

LABELS = bundle.get_labels(star=None)
DICTIONARY = bundle.get_dict(star=None)
for k, v in DICTIONARY.items():
    print(f"{k}: {v}")

tokenized_transcript = [DICTIONARY[c] for word in transcript for c in word]
aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
word_spans = unflatten(token_spans, [len(word) for word in transcript])
print(word_spans[0])



ratio = waveform.size(1)/emission.size(1)

audio = AudioSegment.from_wav("test_audio_16khz.wav")
for i in range(len(word_spans)):
    start_timestamp = 1000 * int(word_spans[i][0].start * ratio) / bundle.sample_rate
    end_timestamp = 1000 * int(word_spans[i][-1].end * ratio) / bundle.sample_rate
    print(start_timestamp, end_timestamp)
    new_audio = audio[start_timestamp:end_timestamp]
    new_audio.export(f'out{i}.wav', format="wav")











#muted_bytearray = bytearray_data[:start_offset] + b'\x00' * (end_offset - start_offset) + bytearray_data[end_offset:]

# outfile = wave.open("out.wav", "w")
# outfile.setnchannels(1)
# outfile.setsampwidth(2)
# outfile.setframerate(16000)
# outfile.writeframes(muted_bytearray)
# outfile.close()

