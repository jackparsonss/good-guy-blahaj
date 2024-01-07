# good-guy-blahaj

## Inspiration
Discord has become a very popular social media platform for large communities to interact online. The platform is especially popular with younger children and teens that join large open discord servers for related to video games or content creators they enjoy. One issue with moderating discord is that there are large voice chats that are often used when playing video games with strangers and there are very few existing tools to moderate audio data to ensure people can safely use discord voice chats without being exposed to inappropriate content.

## What it does
Our moderation tool takes in a stream of audio and censors inappropriate languages such as swear words or sexual content. The goal is for this to be done in real time so that it doesn't disrupt from the experience of using Discord. This could also be used for content creators that play games with a voice chat between players as it would allow them to use voice chat and interact with fans without risking people saying inappropriate words over voice chat.

## How we built it
We created a server that can run 3 large language models that are the backbone of our project, OpenAI's Whisper, Meta's wav2vec2 and Mistral 7B. We use Whisper to transcribe audio files to text so that we can use both traditional nlp tools such as regex and input it to Mistral 7B to determine which words and phrases should be censored. We use wav2vec2 for Forced Alignment to match the words in the transcript that need to be censored to the time within the audio file that that word was spoken. We can then take the output from these 3 models to modify the audio stream to censor any inappropriate content.

We use the speech_recognition library on our client side computer to record audio and partition it into small segments without partitioning the audio while a word is being spoken and then continuously send audio data to the server.

Finally we also developed a web UI using Astro and go with a python backend that can display the censored or uncensored transcripts to allow those hard of hearing to use the built in transcription tools to generate subtitles which they can chose to censor.

We connect the client to the server using sockets to allow for continuous streams of data to get as close to real time transcription as possible.

## Challenges we ran into
One of our most fundamental challenges we had is that our speech-to-text models cannot run on continuous streams of audio; thus, we need to partition the audio into chunks without cutting off words to process on the server. This also makes real time processing difficult as there is a delay equal to the chunk size, this creates a trade of where larger chunks have a longer delay but smaller chunks have a higher risk of cutting off words and reduce the quality of the audio playback.

Another challenge with processing audio files in real time with affordable models is the lack of streaming support. Sliding-window inference is not available on self-hosted models and we had hoped to stay as open-source as possible.

Prompting Mistral 7B to detect inappropriate messages that are more complex than a single inappropriate word is considerably challenging. We asked the model to simply repeat the statements that it receives as input but with inappropriate content removed, however, the model would often attempt to replace the inappropriate words with other random words or mistake the audio input for a new prompt and stop following the initial instructions it was given.

## What's next for good-guy-blahaj
We would like to offer proper integration with systems that could use audio moderation tools like Discord and Slack as well as offering moderation for languages other than english. We would also like to look into training our own LLM for filtering inappropriate content more consistently and allowing to change what content is filtered based on user ages or preferences.

Additionally, exposing our text filtering as an API separate from the voice filtering would allow for more complete discord moderation, with potential to moderate voice and text channels.

## System Design
![image](https://github.com/jackparsonss/good-guy-blahaj/assets/62918090/ed527bac-9d99-4d2b-9dd5-bbbd81364225)

