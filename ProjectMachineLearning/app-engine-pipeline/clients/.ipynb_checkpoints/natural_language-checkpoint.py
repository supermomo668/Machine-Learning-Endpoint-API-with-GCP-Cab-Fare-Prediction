from google.cloud import language


class NaturalLanguageClient:
    def __init__(self):
        print("Lagnuage Client")
        self.client = language.LanguageServiceClient()

    def analyze_entities(self, text):
         # Available types: PLAIN_TEXT, HTML
        type_ = language.Document.Type.PLAIN_TEXT

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages
        language_ = "en"
        document = {"content": text, "type_": type_, "language": language_}

        # Available values: NONE, UTF8, UTF16, UTF32
        encoding_type = language.EncodingType.UTF8

        return self.client.analyze_entities(request = {'document': document, 'encoding_type': encoding_type})
