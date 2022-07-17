import requests
from relevanceai.operations_new.transform_base import TransformBase


class TranslateTransform(TransformBase):
    def __init__(
        self,
        fields,
        model_id="facebook/mbart-large-50-many-to-many-mmt",
    ):
        if model_id != "facebook/mbart-large-50-many-to-many-mmt":
            raise NotImplementedError("Translation model not found.")

        self.model_id = model_id
        self.fields = fields
        # Get the language dictionary
        self.lang_dict = requests.get(
            "https://gist.githubusercontent.com/boba-and-beer/b142b44e7c5120c5ce839e4b7f8f0247/raw/21b73269fecb67e871843086fe8ebe78d8b4c7a0/lang_isocode_mapping.json"
        ).json()
        self.lang_to_normal = requests.get(
            "https://gist.githubusercontent.com/boba-and-beer/65bf4a0810d6492050577f1d2373d394/raw/d3817a37401b0776b8ed4bda1bbcf89e8375c747/lang_iso_2_mapping.json"
        ).json()
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

        self.model = MBartForConditionalGeneration.from_pretrained(model_id)
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_id)

    def translate_to_english(self, text):
        from langdetect import detect_langs

        language = detect_langs(text)
        if language[0].lang == "en":
            return None, None
        supported_languages = []
        for lang in language:
            try:
                return self._pass_text_through_model(
                    text, lang.lang, supported_languages
                )
            except Exception as e:
                pass

        return None, None

    def _pass_text_through_model(self, text, language: str, supported_languages: list):
        self.tokenizer.src_lang = self.lang_dict[language]
        if language in self.lang_to_normal:
            supported_languages.append(self.lang_to_normal[language])
        else:
            supported_languages.append(language)

        encoded_hi = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_hi, forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"]
        )
        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return supported_languages, translation

    def translate_document(self, field, document):
        try:
            sup_lang, translation = self.translate_to_english(
                self.get_field(field, document)
            )
            self.set_field(
                "_translation_." + field,
                document,
                {"detectedLanguage": sup_lang, "translation": translation},
            )
        except:
            pass
        return document

    def bulk_translate_documents(self, documents, **kwargs):
        for field in self.fields:
            [self.translate_document(field, d, **kwargs) for d in documents]
        return documents

    transform = bulk_translate_documents  # type: ignore
