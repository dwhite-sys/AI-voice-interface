#--------------------------------------------------------------------------------------------------------------
#   Dependancies
#--------------------------------------------------------------------------------------------------------------

import pyaudio
import vosk; vosk.SetLogLevel(-1) # Disables the log because I don't like how it looks in the console.
import json
import ollama
import pyttsx3
import time

#--------------------------------------------------------------------------------------------------------------
#   Modules
#--------------------------------------------------------------------------------------------------------------

from simplify import clear

#--------------------------------------------------------------------------------------------------------------
#   Config
#--------------------------------------------------------------------------------------------------------------

LANGUAGE_MODEL = 'llama3.2:3B'                       # Feel free to change the AI models to more powerful ones,
VOICE_MODEL = 'vosk-model-en-us-daanzu-20200905'     # these are the defaults for performance reasons.
RECORD_TIMEOUT = 4

#--------------------------------------------------------------------------------------------------------------
#   Initialization + indication of initialization
#--------------------------------------------------------------------------------------------------------------

# Text to Speech
engine =  pyttsx3.init()
print('TTS initialized')
engine.say('Text to speech initialized')
engine.runAndWait()

# Language Model
initialization_message = ollama.generate(LANGUAGE_MODEL, 'Just say "Language model initialized"')['response']
print('LLM initialized')
engine.say(initialization_message)
engine.runAndWait()

# Voice Model
voice_model = vosk.Model(model_name=VOICE_MODEL)
recognitize = vosk.KaldiRecognizer(voice_model, 16000)
print('Speech Recognition Initialized')
engine.say('Speech Recognition Initialized')
engine.runAndWait()

#--------------------------------------------------------------------------------------------------------------
#   Start loop
#--------------------------------------------------------------------------------------------------------------

def start_conversation():
    "Begins a conversation."
    context = []
    engine.say('Conversation begin')
    engine.runAndWait()
    wait_for_input(context)

#--------------------------------------------------------------------------------------------------------------
#   Wait for Input
#--------------------------------------------------------------------------------------------------------------

def wait_for_input(context:list):
    '''
    *For internal use only, use start_conversation() instead.

    This takes in context and waits for new
    input. When it finds new input, it passes it on to record input.
    '''
    clear()
    audio = pyaudio.PyAudio()
    input_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    while True:
        data = input_stream.read(4096)                       # Reads data from a specified time period from the input stream.
        if recognitize.AcceptWaveform(data):                 # This puts the data through the speech recognition AI.
            result = json.loads(recognitize.Result())        # This is the result from the AI.
            text = result['text']                            # The actual text from the result.
            if text != '' and text != ' ':
                speech = text                                # This is what's sent to the recording function as the initial wording.
                audio.terminate()
                record_input(speech, context)
                break

#--------------------------------------------------------------------------------------------------------------
#   Record Input
#--------------------------------------------------------------------------------------------------------------

def record_input(speech:str, context:list):
    '''
    *For internal use only, use start_conversation() instead. 

    This takes in context and initial speech from wait_for_input, and records input.
    This input is then passed to generate_output.
    '''
    clear()
    start_timeout = time.time() # Compared against the timeout limit constant at the top of the script.
    audio = pyaudio.PyAudio()
    input_stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    while True:
        if time.time()-start_timeout >= RECORD_TIMEOUT:
            data = input_stream.read(4096)                   # Reads data from a specified time period from the input stream.
            if recognitize.AcceptWaveform(data):             # This puts the data through the speech recognition AI.
                result = json.loads(recognitize.Result())    # This is the result from the AI.
                text = result['text']                        # The actual text from the result.
                if text != '':
                    speech += text + '\n'                    # Adds to speech, which is what's passed to the AI.
                    start_timeout = time.time()
                    print(speech)                            # Prints the speech to the console for debugging.
        else:
            break
    audio.terminate()
    generate_response(speech, context)

#--------------------------------------------------------------------------------------------------------------
#   Generate Response
#--------------------------------------------------------------------------------------------------------------

def generate_response(speech:str, context:list):
    '''
    *For internal use only, use start_conversation() instead.

    This takes in context and initial speech from wait_for_input, and records input.
    This input is then passed to generate_output.
    '''
    clear()
    sentence = '' # This is used to keep track of the complete output of the ollama text stream, which is fed to the TTS.
    output = ollama.generate(LANGUAGE_MODEL, speech, context=context, stream=True)
    for chunk in output:
        sentence_enders = ['.', '!', '?', ',']    # Any symbols that pause a sentence
        markdowns = '*_'                          # Any formatting symbols that shouldn't be read out loud.
        text = chunk['response']
        sentence += text.strip(markdowns)         # This strips away the markdows.
        print(text, end='', flush=True) # Print to the console. Good for debugging and killing boredom between generation and TTS.
        if any(ender in text for ender in sentence_enders):
            engine.say(sentence)
            engine.runAndWait()
            sentence = ''
        try:
            context = chunk['context']
        except:
            pass
    # Pass the torch
    if sentence != '':
        engine.say(sentence)
        engine.runAndWait()
    time.sleep(0.5)
    wait_for_input(context)

#--------------------------------------------------------------------------------------------------------------
#   End of functions
#--------------------------------------------------------------------------------------------------------------