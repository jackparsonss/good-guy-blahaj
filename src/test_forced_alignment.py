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

tokenized_transcript = [DICTIONARY[c] for word in transcript for c in word]
aligned_tokens, alignment_scores = align(emission, tokenized_transcript)

token_spans = F.merge_tokens(aligned_tokens, alignment_scores)
word_spans = unflatten(token_spans, [len(word) for word in transcript])

#mute the 5th word which is "bottom"
bit_rate = 256000

start_timestamp = word_spans[4][0].start/100
end_timestamp = word_spans[4][-1].end/100

start_offset = int(start_timestamp * bit_rate)
end_offset = int(end_timestamp * bit_rate)
print(start_offset, end_offset)

with wave.open("test_audio_16khz.wav", "rb") as wav_file:
    num_frames = wav_file.getnframes()
    audio_data = wav_file.readframes(num_frames)

bytearray_data = bytearray(audio_data)



