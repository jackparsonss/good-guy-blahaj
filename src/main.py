import requests as rq
import json
import time
import re
#from textblob import TextBlob
import inflect

inflect = inflect.engine()

with open("bad-words.csv", "r") as f:
    censor_words = [line.strip() for line in f.readlines()]


##############################################
# External inputs
##############################################
input_words = "I got some pairs of orange oranges for my pear. Go jump off a cliff! You beast."
processed_words = str(input_words)

##############################################
# Regex pre-processing
##############################################
input_words = input_words.lower()

def pluralize(w):
    if not inflect.singular_noun(w):
        return (w, inflect.plural(w))
    else:
        return (inflect.singular_noun(w), w)

censor_words2 = list()
for word in censor_words:
    a = pluralize(word)
    censor_words2.append(a[0])
    censor_words2.append(a[1])

for word in censor_words2:
    processed_words = re.sub(fr"\b{word}\b", "#BEEP#", processed_words, flags=re.IGNORECASE)

# Remove consecutive beeps
processed_words = re.sub("(\s*#BEEP#\s*)+", " #BEEP# ", processed_words)

##############################################
# Request
##############################################
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {
    "model": "openorca_hacked:v0.1",
    "prompt": processed_words,
}

data = json.dumps(data)

start = time.time()
response = rq.post('http://localhost:11434/api/generate', headers=headers, data=data)
cease = time.time()

print(f"Took: {cease-start}s")

##############################################
# Post-processing
##############################################
json_version = re.sub(r"\n(?=.)", ",", response.text)
json_version = f"[{json_version}]"
json_version = re.sub(f"\n", "", json_version)

js = json.loads(json_version)

words = "".join(word['response'] for word in js).strip()

print("==== INPUT ============")
print(input_words)
print("==== REGEX ============")
print(processed_words)
print("==== LLAMA ============")
print(words)
print("=======================")
