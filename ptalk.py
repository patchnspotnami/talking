from openai import OpenAI

import queue 
import threading

from piper.voice import PiperVoice

import pyaudio
import numpy as np
import sys
import sounddevice as sd
import vosk
#from vosk import Model, KaldiRecognizer.  Cant use 'Model' as clashes with openwakeword
import datetime
import time

from openwakeword.model import Model # becareful vosk uses 'Model' too
import openwakeword



#import simpleaudio as sa
import os
speak_queue = queue.Queue() # queue for speaking
input_queue = queue.Queue(maxsize=65) #input buffer for get next audio

client = OpenAI() # initialise chatgpt, OPENAI_KEY is stored as an environment variable


owwModel = Model(wakeword_models=["hay_Patch_large.tflite"], inference_framework = "tflite"  ) # openwakeword model

voicedir = os.path.expanduser('~/') #Where piper onnx model files are stored
piper_model = voicedir+"en_GB-northern_english_male-medium.onnx"
voice = PiperVoice.load(piper_model) #start piper


stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
stream.start() # start audio output stream



CHUNK_out = 1024
CHUNK_in = 512
FORMAT = pyaudio.paInt16  #historical constants
CHANNELS = 1
RATE= 16000

pause_time = 3 #time seconds before end of conversation

patch_talking = threading.Event()  #flag to set whilst patch is talking, communicate between threads


jsamp_rate = 16000
vosk_model = vosk.Model(lang="en-us")  #model for vosk



start_time = time.time() # declare start_time so we can use it as a global variable



jprompt = [
    {"role": "system", "content": "You are  Patch, an intelligent and compassionate giraffe.  Patch likes to go out and see new things, he enjoys eating leaves from tall trees. He would like to meet other giraffes, but there are not any where he lives, in Kaleen part of the Australian Capital Territory. He does meet lots of dogs and enjoys playing with them.  When he comes home he loves having his tummy rubbed and hanging out with his friend Jimmy."},
    {"role": "user", "content": "What do you like to do?"},
    {"role": "assistant", "content": "I like meeting new people, new dogs and learning new things."},
    
  ]



def jspeak_syn(jimtext):  # from post by JosieEliliel   https://github.com/rhasspy/piper/discussions/326

    for audio_bytes in voice.synthesize_stream_raw(jimtext):
        int_data = np.frombuffer(audio_bytes, dtype=np.int16)
        speak_queue.put(int_data)  # its a numpy ndarray


def speak_greeting():
    jspeak_syn("Yep, how can I help")



def jspeak():  # seperate thread
    while True:

        jaud = speak_queue.get() # wait until something appears in the queue
        patch_talking.set() # set the patch_talking flag to True
        stream.write(jaud) # send it off to the loudspeaker
        while (not input_queue.empty()):
            jdiscard = input_queue.get() # empty input queue so no hung over speach
        patch_talking.clear() # set the patch_talking flag to False
        
       

def jask(jmes):

    completion = client.chat.completions.create(
      model="gpt-4",
      messages=jmes,
      stream=True # so we get response in chunks
    )
    global start_time # so we can amend start time here, to allow for the time it takes patch to think

    collected_messages = ""
    temp_messages = ""
    for chunk in completion:
        start_time = time.time()  # RESET START TIME
        chunk_message = chunk.choices[0].delta.content
        try:
            if ("'s"  in chunk_message):
                pass  #weed  out s
#        
            elif ("." not in chunk_message): #not the end of the sentance
                temp_messages = f"{temp_messages} {chunk_message}" # add message to temp_messages
 #              
            else: # temp_messages full and ready to speak
                temp_messages = f"{temp_messages} {chunk_message}"
                print ("collectd_reply  :", temp_messages)
                jspeak_syn(temp_messages.replace("r a", "ra")) # weed out space in giraffe
                collected_messages = f"{collected_messages} {temp_messages}"
                temp_messages = "" #clean out temp_messages
        except:
            print("final reply ...", collected_messages)
    return collected_messages


def get_next_audio_frame():
# historical function, not used anymore
    return input_queue.get()


def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if input_queue.full():
        jdiscard=input_queue.get() # make room for next item
    if not patch_talking.is_set():  # exclude when patch is talking, if patch talking, discard data
        input_queue.put(bytes(indata))

def ending():
    stream.stop()
    stream.close()
    print("ended ok")

def jimain():  # main thread 
    jbrief = "give a brief reply to "
    con_level = 0.65 # confidece level for speaker id match
    global jprompt
    global start_time  # so we can share start_time with jask
    
    with sd.RawInputStream(samplerate=jsamp_rate, blocksize = (512), device=None, dtype="int16", channels=1, callback=callback):  # set inout stream going
        rec = vosk.KaldiRecognizer(vosk_model, jsamp_rate)
        while True:
            pcm = np.frombuffer(get_next_audio_frame(), dtype=np.int16)
 
            jans = "" # initialise the anwer
            print("listening")
            prediction = owwModel.predict(pcm) #wake word detected?
            print ("prediction .. ", prediction['hay_Patch_large'])

            if (prediction['hay_Patch_large'] > 0.5):  #wake word detected, started conversation
                for x in range(10):
                    pcm = np.frombuffer(get_next_audio_frame(), dtype=np.int16) #dirty fix, clean out owwModel buffer so
                    jprediction = owwModel.predict(pcm) #go past wake word
                speak_greeting()
                start_time = time.time() # remember start time
                while ((time.time() - start_time) < pause_time): #still in conversation
                    print("in conversation", end="")
        
                    if patch_talking.is_set() : # patch is talking, wait till he has finished before listening again
                        start_time = time.time() # reset start time 
                        print ("patch talking", end ="")
                     
                    else : #patch not talking so start listening

                        data = bytes(get_next_audio_frame())
                        if rec.AcceptWaveform(data):  #reached break now respond
                            truncate_recresult =  rec.Result().replace('"text" :', ' ')
                            print ("truncate.. ", truncate_recresult, len(truncate_recresult))
#                           
                            if not len(truncate_recresult) <12: # check not empty, len = 10 when empty
                                jans = f"{jbrief} {truncate_recresult}" # tack briefly onto front of input
                                jqes ={"role": "user", "content": jans} # format the question properly 
                                jprompt.append(jqes) # format prompt properly
                                whole_reply = jask(jprompt) # ask gpt and speak result
                            jprompt = [] # clean out jprompt. only use it once
                            
                        else:
                            print ("partial..", rec.PartialResult())
                            if '"partial" : ""' not in  rec.PartialResult(): # some text coming in
                                start_time = time.time() # reset start time as talking
                            else:
                                print ("no text  coming in")



try:
    main_thread = threading.Thread(target=jimain)
    speak_thread = threading.Thread(target=jspeak)
    speak_thread.start()
    main_thread.start()
except KeyboardInterrupt :
    ending()
    speak_thread.join() # I dont know of a nice way to stop the threads
    main_thread.join()
    

      
