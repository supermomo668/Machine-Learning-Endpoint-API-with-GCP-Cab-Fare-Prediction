from google.cloud import speech
#from google.cloud.speech import enums
#from google.cloud.speech import types


class SpeechToTextClient:
    def __init__(self):
        self.client = speech.SpeechClient()

        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # LINEAR16
            sample_rate_hertz=16000,
            language_code='en-US')

    def recognize(self, content_bytes):
        audio = speech.RecognitionAudio(content=content_bytes)
    
        response = self.client.recognize(config=self.config, audio=audio)
        print(response.results)
        for result in response.results:
            return result.alternatives[0].transcript
