from relevanceai.base import Base


class Encoders(Base):
    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

    def textimage(self, text: str):
        """ 
        Encode text to make searchable with images

        Parameters
        ----------
        text: string
            Text to encode
        """
        return self.make_http_request(
            "/services/encoders/textimage", method="GET", parameters={"text": text}
        )

    def text(self, text: str):
        """ 
        Encode text

        Parameters
        ----------
        text: string
            Text to encode
        """
        return self.make_http_request(
            "/services/encoders/text", method="GET", parameters={"text": text}
        )

    def multi_text(self, text):
        """
        Encode multilingual text

        Parameters
        ----------
        text: string
            Text to encode
        """
        return self.make_http_request(
            "/services/encoders/multi_text", method="GET", parameters={"text": text}
        )

    def image(self, image):
        """
        Encode an image

        Parameters
        ----------
        image: string
            URL of image to encode
        """
        return self.make_http_request(
            "/services/encoders/image", method="POST", parameters={"image": image}
        )

    def imagetext(self, image):
        """
        Encode an image to make searchable with text
        
        Parameters
        ----------
        image: string
            URL of image to encode
        """
        return self.make_http_request(
            "/services/encoders/imagetext", method="GET", parameters={"image": image}
        )