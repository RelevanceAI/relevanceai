import pandas as pd


class JsonShowerMixin:
    def _show_json(
        self, documents, image_fields, audio_fields, highlight_fields, text_fields, **kw
    ):
        from jsonshower import show_json

        if not text_fields:
            text_fields = pd.json_normalize(documents).columns.tolist()
        else:
            text_fields = text_fields
        return show_json(
            documents,
            image_fields=image_fields,
            audio_fields=audio_fields,
            highlight_fields=highlight_fields,
            text_fields=text_fields,
            **kw,
        )
