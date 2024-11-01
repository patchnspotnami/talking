import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from naoqi_bridge_msgs.msg import AudioBuffer

import numpy as np
import soundfile as sf
import time
import assemblyai as aai

from scipy import signal

from openwakeword.model import Model
import openwakeword

from openai import OpenAI

import queue 
import threading

import os
aai_key = os.environ["AAI_KEY"]
aai.settings.api_key = aai_key
owwModel = Model(wakeword_models=["/home/jimmy/jimexp/ami_exp/openwakeword/Hey__Ameeee_l.tflite"], inference_framework = "tflite"  )

client = OpenAI() # initialise chatgpt, OPENAI_KEY is stored as an environment >
#speak_queue = queue.Queue() # queue for speaking
#ask_queue = queue.Queue(maxsize=65) #queue for asking ai

ami_talking = False
ami_woken = False
ami_woken_time = time.time()


class AmiNode(Node):
    def __init__(self):
        super().__init__('ami_node')
        self.publisher = self.create_publisher(String, '/speech', 10)
        self.timer = self.create_timer(0.5, self.amitalk_callback)
        self.count = 0
        self.estimate_time = time.time() # initialise estimate time
        self.subscription = self.create_subscription(
            AudioBuffer,
            '/audio',
            self.listener_callback,
            10)

        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=48000,  # audio stream  from Ami
            on_data=self.on_data,
            on_error=self.on_error,
            on_open=self.on_open,
            on_close=self.on_close,
            word_boost=["ROS", "robot"]
        )

        self.transcriber.connect()


        self.mic_index = 0  # Change this to select different microphones (0, 1, 2, or 3)
        self.speak_queue = queue.Queue()
#        self.transcriber.on_transcript(self.handle_transcript)

  
    def __del__(self):
        self.transcriber.close() 

     #     Create functions to handle events from the real-time transcriber.
    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        print("Session opened with ID:", session_opened.session_id)


    def on_error(self, error: aai.RealtimeError):
        print("jimmy Error:", error)


    def on_close(self):
        print("Session closed")
        self.transcriber.close() 

    def on_data(self, transcript: aai.RealtimeTranscript):
        global ami_woken_time
        if not transcript.text:
            print ("on data .. no transcript")
            return
        if ("hope that" in transcript.text):
            return
        print("resetting ami_wokken_time")
        ami_woken_time=time.time()
        if isinstance(transcript, aai.RealtimeFinalTranscript):  #end of transcript
            # Add new line after final transcript.
            print("partial .. ", transcript.text) #, end="\r\n")
            self.ask_ai(transcript.text) # put transcrit from ami into ask_queue
            
        else:
            print("finally .. ", transcript.text)#, end="\r")        




    def listener_callback(self, msg):
        global ami_woken
        global ami_woken_time
        elapsed_time = time.time() - ami_woken_time
        
        if (elapsed_time > 20):
            ami_woken = False
            print ("timed out")
        print ("callback .., ami is ", ami_woken,  "for .. ", elapsed_time)
 #       global ami_talking
        audio_array = np.frombuffer( msg.data, dtype=np.int16) 
     # Reshape the array to separate the channels
        num_channels = len(msg.channel_map)
        audio_array = audio_array.reshape(-1, num_channels)     

        # Extract data for the selected microphone
        single_mic_data = audio_array[:, self.mic_index] 
        if ami_woken :  # ami has been woken up
            mic_data_bytes=single_mic_data.tobytes()
            print("send off to transcriber")
            self.transcriber.stream(mic_data_bytes) # see on_data 
        else :  # ami not awake
            self.try_wake(single_mic_data)

    def try_wake(self, mic_data):
        global ami_woken
        global ami_woken_time
        resampled = signal.resample(mic_data, int(4096 * 16000 / 48000)) # was 48mhz but need 16mhz
 #       print("mic_data shape : ", mic_data.shape, "type : ", type(mic_data))
        prediction = owwModel.predict(resampled)
#        print("listening, for wake up")
        if (prediction['Hey__Ameeee_l'] > 0.1 ) :
            print("deteceted ! ")
            ami_woken = True # set wwoken up flag
            ami_woken_time = time.time()
            self.speak_queue.put("Yep.") # How can I help. What would you like to know")
        # need to clean out the oww buffer - dirty fix
            zero_array = np.zeros(1365, dtype=np.int16)
 #           print("zero array shape ", zero_array.shape)
            for x in range(10):
                prediction = owwModel.predict(zero_array) # fill the oww buffer>
  
    def amitalk_callback(self):
        global ami_woken
        if not self.speak_queue.empty():  # something to say
            msg = String()
            msg.data = self.speak_queue.get() # get data out of the queue
            self.publisher.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')
            estimate_time = len(msg.data)/20 # assume sspeak 10 characters per second
            time.sleep(estimate_time) # stop everything whilst ami talking
           


    def exp_talk(self):
        jimstring = "this is a string number "
        for x in range(3):
            self.msg.data = jimstring + str(x)
            print("message data for : ", x, "  ", self.msg.data)
  #          self.publisher_.publish(self.msg)


    def ask_ai(self, ques_tion) : #asking ai thread

        
        print("thats a tricky one!")
        self.speak_queue.put("Hmm, thats a tricky one")
  #      ai_prompt = ami_prompt + [{role:"user", content:ques_tion}]

        completion = client.chat.completions.create(
               model="gpt-4o-mini",
               messages= [{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content":ques_tion}
                          ],
                
               
               stream=False # so we get whole response
            )
        self.speak_queue.put(completion.choices[0].message.content)
        self.speak_queue.put("I hope that answers you")
           


def main(args=None):
    rclpy.init(args=args)

    amis_node= AmiNode()

    rclpy.spin(amis_node)

    amis_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
