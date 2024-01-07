#!/usr/bin/env python3
import difflib


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


words1 = "my horse went to graze"
words2 = "my went to"

words1_word_to_index = get_word_starts(words1)
words1_running_index = to_running_index(words1_word_to_index)

print(words1_running_index)
print(words1_word_to_index)
print(words1)

for i in range(len(words1)):
    is_printed = False

    for x in words1_word_to_index:
        if x[0] == i:
            is_printed = True
            print("^", end="")
            break

    if not is_printed:
        print(" ", end="")

print()

matcher = difflib.SequenceMatcher(None, words1, words2)

for x in matcher.get_matching_blocks()[:-1]:
    i, j, n = x

    while words1[i] == " ":
        i += 1
        j += 1
        n -= 1

    for x in range(n):
        x += i
        index = words1_running_index[x]

        if index != -1:
            words1_word_to_index[index][2] = True

word_string = word_starts_to_string(words1_word_to_index)
print(word_string)
