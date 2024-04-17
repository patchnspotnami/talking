
import pyaudio
import numpy as np
from openwakeword.model import Model
import openwakeword

openwakeword.utils.download_models()
# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)


# Instantiate the model(s)


#owwModel = Model(wakeword_models=['/home/ami/.local/lib/python3.10/site-packages/openwakeword/resources/models/hey_patch.onnx'], inference_framework = "onnx"  )

owwModel = Model(wakeword_models=["/home/ami/jimexp/openwakeword/hay_Patch_medium.tflite"], inference_framework = "tflite"  )
#owwModel = Model() 

n_models = len(owwModel.models.keys())

while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(audio)

        # Column titles
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        # Print results table
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')
