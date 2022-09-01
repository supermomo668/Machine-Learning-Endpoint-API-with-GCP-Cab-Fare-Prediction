from google.cloud import texttospeech


class TextToSpeechClient:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

        self.voice_selection_params = texttospeech.VoiceSelectionParams(
            language_code='en-US',
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)

        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000)

    def synthesize_speech(self, text):
        synthesis_input = texttospeech.SynthesisInput(text=text)

        response = self.client.synthesize_speech(input=synthesis_input, 
                                                 voice=self.voice_selection_params, 
                                                 audio_config=self.audio_config)

        return response.audio_content
