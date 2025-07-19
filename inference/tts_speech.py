from gtts import gTTS
import os

def speak_caption(caption, filename="output.mp3"):
    tts = gTTS(text=caption, lang='en')
    tts.save(filename)
    os.system(f"start {filename}")  # Use 'afplay' on Mac or 'xdg-open' on Linux
