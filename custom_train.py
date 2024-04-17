import openwakeword

openwakeword.train_custom_verifier(
    positive_reference_clips = ["jimpos1.wav", "jimpos2.wav", "jimpos3.wav"]
    negative_reference_clips = ["jimneg.wav", "negative_clip2.wav"]
    output_path = "path/to/directory/model.pkl"
    model_name = "hey_jarvis.onnx" # the target model path which matches the wake word/phrase of the collected positive examples
)
