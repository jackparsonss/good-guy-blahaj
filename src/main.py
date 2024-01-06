import requests as rq
import json
import time
import re

##############################################
# External inputs
##############################################
input_words = "I got some pairs of oranges for my pear."

censor_words = [
    "oranges",
    "pears",
    "pear",
]

##############################################
# Request
##############################################

s_censor_words = "".join(f" - {word}\n" for word in censor_words)

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

prompt = f"""
Please replace all the words in the following list with "#BEEP#" in the text at the bottom. You are not allowed to output any additional explanation. Be direct to the point. Only output the censored sentence, nothing more. You cannot add any notes after!

{s_censor_words}
{input_words}
"""

print(prompt)

data = {
    "model": "mistral:instruct",
    "prompt": prompt,
}

data = json.dumps(data)

start = time.time()
response = rq.post("http://localhost:11434/api/generate", headers=headers, data=data)
cease = time.time()

print(f"Took: {cease-start}s")

##############################################
# Post-processing
##############################################

json_version = re.sub(r"\n(?=.)", ",", response.text)
json_version = f"[{json_version}]"
json_version = re.sub("\n", "", json_version)

js = json.loads(json_version)

words = "".join(word["response"] for word in js).strip()

print(words)
