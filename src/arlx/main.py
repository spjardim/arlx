from arlx.audio.sources.input_capture import InputAudioCapture
if __name__ == "__main__":

    capture = InputAudioCapture(device_name="default")
    capture.start()
    
    try:
        while True:
            audio = capture.get_latest_sample()
            print(audio)
    except KeyboardInterrupt:
        print("Stopping audio feature extraction...")
