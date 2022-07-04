import requests

from typing import List
from concurrent.futures import ThreadPoolExecutor

from relevanceai.operations_new.vectorize.models.base import VectorizeModelBase
from relevanceai.utils.decorators.vectors import catch_errors

try:
    import traceback
    import clip
    import torch
    import requests
    import cv2
    from PIL import Image
    from requests.exceptions import MissingSchema
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Run `pip install git+https://github.com/openai/CLIP.git`."
    )
except:
    import traceback

    traceback.print_exc()


class ClipImage2Vec(VectorizeModelBase):
    def __init__(self, url, vector_length, context_length=77):

        self.context_length = context_length
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.model, self.preprocess = clip.load(url, device=self.device)
        self.vector_length = vector_length
        self.url = url

    @property
    def model_name(self):
        return "clip"

    def read(self, image_url):
        """It takes an image url, and returns an image object

        Parameters
        ----------
        image_url
            The URL of the image to be downloaded.

        Returns
        -------
            The image is being returned.

        """
        try:
            return Image.open(requests.get(image_url, stream=True).raw)
        except MissingSchema:
            return Image.open(image_url)

    def preprocess_black_and_white_image(self, x):
        """It takes a black and white image, converts it to a tensor, adds two more channels to it, and
        then normalizes it

        Parameters
        ----------
        x
            the image

        Returns
        -------
            A tensor of shape (3, 224, 224)

        """
        x = self.preprocess.transforms[0](x)
        x = self.preprocess.transforms[1](x)
        x = self.preprocess.transforms[3](x)
        x = torch.stack((x, x, x), dim=1)
        x = self.preprocess.transforms[4](x)
        return x

    @catch_errors
    def encode_text(self, text: str):
        """If the device is cuda, then tokenize the text, send it to the device, encode it, detach it,
        convert it to numpy, convert it to a list, and return the first element of that list

        Parameters
        ----------
        text : str
            str

        Returns
        -------
            A list of floats.

        """
        if self.device == "cuda":
            text = clip.tokenize(text, context_length=self.context_length).to(
                self.device
            )
            return self.model.encode_text(text).cpu().detach().numpy().tolist()[0]
        elif self.device == "cpu":
            text = clip.tokenize(text, context_length=self.context_length).to(
                self.device
            )
            return self.model.encode_text(text).detach().numpy().tolist()[0]

    def encode_video(self, video_url: str):
        """It takes a video file, reads it frame by frame, converts each frame to a PIL image, preprocesses
        it, converts it to a tensor, and then encodes it using the model

        Parameters
        ----------
        video_url : str
            the path to the video file

        Returns
        -------
            A list of floats

        """
        cap = cv2.VideoCapture(video_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    def bulk_encode_text(self, texts: List[str]):
        """If the device is cuda, then tokenize the text, send it to the device, encode it, and return the
        result.

        If the device is cpu, then tokenize the text, send it to the device, encode it, and return the
        result.

        The only difference is the device.

        So, let's make a function that takes a device as an argument and returns a function that does
        the above.

        We'll call this function `make_encode_text_function`.

        The function it returns will be called `encode_text`.

        We'll use the `partial` function from the `functools` module to do this.

        The `partial` function takes a function and some arguments and returns a function that takes the
        remaining arguments.

        So, we'll pass it the `bulk_encode_text` function and the `device

        Parameters
        ----------
        texts : List[str]
            List[str]

        Returns
        -------
            A list of lists of floats.

        """
        if self.device == "cuda":
            tokenized_text = clip.tokenize(
                texts, context_length=self.context_length
            ).to(self.device)
            return (
                self.model.encode_text(tokenized_text).cpu().detach().numpy().tolist()
            )
        elif self.device == "cpu":
            tokenized_text = clip.tokenize(
                texts, context_length=self.context_length
            ).to(self.device)
            return self.model.encode_text(tokenized_text).detach().numpy().tolist()

    def preprocess_image(self, img: str):
        try:
            if self.is_greyscale(img):
                return (
                    self.preprocess_black_and_white_image(self.read(img))
                    .unsqueeze(0)
                    .to(self.device)
                )
            return self.preprocess(self.read(img)).unsqueeze(0).to(self.device)
        except:
            traceback.print_exc()
            return torch.empty((1, 3, 224, 224), dtype=torch.int32, device=self.device)

    def parallel_preprocess_image(self, images: List[str]):
        with ThreadPoolExecutor(max_workers=5) as executor:
            future = executor.map(self.preprocess_image, images)
        return list(future)

    @catch_errors
    def encode_image(self, image_url: str):
        """If the device is cpu, then preprocess the image, add a dimension to the image, and then return
        the encoded image.

        If the device is cuda, then preprocess the image, add a dimension to the image if it's 3D, and
        then return the encoded image

        Parameters
        ----------
        image_url : str
            The URL of the image to be encoded.

        Returns
        -------
            A list of floats

        """
        if self.device == "cpu":
            image = self.preprocess_image(image_url)
            if image.dim() == 3:
                image = image.unsqueeze(0).to(self.device)
            return self.model.encode_image(image).detach().numpy().tolist()[0]
        elif self.device == "cuda":
            image = self.preprocess_image(image_url)
            if image.ndim == 3:
                image = image.unsqueeze(0).to(self.device)
            elif image.ndim == 4:
                image = image.to(self.device)
            return self.model.encode_image(image).cpu().detach().numpy().tolist()[0]

    def bulk_encode_image(self, images: List[str]):
        """Batch Processing for CLIP image encoding"""
        # Parallel process the encoding
        future = self.parallel_preprocess_image(images)
        results = self.model.encode_image(torch.cat(list(future))).tolist()
        return results

    def encode(self, data: str, data_type="image"):
        if data_type == "image":
            return self.encode_image(data)
        elif data_type == "text":
            return self.encode_text([data])
        raise ValueError("data_type must be either `image` or `text`")

    def bulk_encode(self, data: List[str], data_type="image"):
        if data_type == "image":
            return self.bulk_encode_image(data)
        elif data_type == "text":
            return self.bulk_encode_text(data)
        raise ValueError("data_type must be either `image` or `text`")

    def is_greyscale(self, img_path: str):
        """Determine if an image is grayscale or not"""
        try:
            img = Image.open(requests.get(img_path, stream=True).raw)
        except MissingSchema:
            img = Image.open(img_path)
        img = img.convert("RGB")
        w, h = img.size
        for i in range(w):
            for j in range(h):
                r, g, b = img.getpixel((i, j))
                if r != g != b:
                    return False
        return True
