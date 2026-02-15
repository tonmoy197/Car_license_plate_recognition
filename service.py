# sevice.py

import cv2
import torch
import base64
import io
from PIL import Image
import numpy as np
from ultralytics.models import YOLO
from easyocr import Reader
import difflib
import imutils
from matplotlib import pyplot as plt


# from domains.domain import NumberPlateRequest, NumberPlateResponse

from pydantic import BaseModel

class NumberPlateRequest(BaseModel):
    image_base64: str

class NumberPlateResponse(BaseModel):
    area: str
    number: str

class NumberPlateService:
    def __init__(self):
        # Patch torch.load BEFORE importing YOLO
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})

        # Load YOLO model
        self.model = YOLO("models/yolo.pt")

        # Restore original torch.load
        torch.load = original_load

        # Load areas
        with open("areas.txt", "r", encoding="utf-8") as f:
            words = f.read().splitlines()


        # print(f"Loaded areas: {words}")

        vclass = [
            'গ','হ','ল','ঘ','চ','ট','থ','এ',
            'ক','খ','ভ','প','ছ','জ','ঝ','ব',
            'স','ত','দ','ফ','ঠ','ম','ন','অ',
            'ড','উ','ঢ','শ','ই','য','র'
        ]

        self.dict = []
        for w in words:
            for c in vclass:
                self.dict.append(f'{w}-{c}')

        # Initialize EasyOCR Reader
        # self.reader = Reader(['bn'], verbose = False, recog_network = 'bn_license_tps', model_storage_directory = "./models/EasyOCR/models",user_network_directory="./models/EasyOCR/user_network", download_enabled = False)
        self.reader = Reader(['bn'], verbose = False, recog_network = 'bn_license_tps', download_enabled = True)

        self.nums = set('০১২৩৪৫৬৭৮৯')

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print(f"load_image(): {path} not found")
        return img
    
    def resize_image(self, img, max_width = 500):
        if img is None:
            print(f'resize_image(): img is null')
            return
        if img.shape[0] > max_width:
            img = imutils.resize(img, max_width)
        return img
    
    def show_image(self, img):
        plt.axis("off")
        if isinstance(img, str):
            img = cv2.imread(img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    

    def decode_image(self, image_base64: str):
        """Decode base64 image string to numpy array"""
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_np

    def detect_license_plate(self, img):
        detection = self.model.predict(img, conf=0.5, verbose=False)
        if detection is None:
            print(f"detect_license_plate(): img is null")
            return
        return detection[0]

    def extract_license_text(self, img):
        result = self.reader.readtext(img, detail = False, paragraph = True)
        area = ""
        number = ""

        for c in "".join(result)[::-1]:
            if c == "-":
                if len(number) <= 4:
                    number += "-"
                else:
                    area += "-"
            elif c in self.nums:
                number += c
            else:
                area += c

        area = area[::-1]
        match = difflib.get_close_matches(area, self.dict, n = 1, cutoff = 0.5)

        if match:
            area = match[0]

        number = number[::-1]

        if number.find("-") == -1 and len(number) == 6:
            number = number[:2] + "-" + number[2:]

        return area.strip(), number.strip()

    def recognize_plate(self, request: NumberPlateRequest) -> NumberPlateResponse:
        img = self.decode_image(request.image_base64)

        detection_result = self.detect_license_plate(img)
        bbox = detection_result.boxes.data.cpu().numpy()
        xmin, ymin = bbox[0][:2].astype(int)
        xmax, ymax = bbox[0][2:4].astype(int)
        cropped_img = img[ymin:ymax, xmin:xmax]

        lp_text = self.extract_license_text(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY))
        
        response = NumberPlateResponse(
            area=lp_text[0],
            number=lp_text[1]
        )
        
        return response


if __name__ == "__main__":
    import base64
    
    test_image_path = "images/test_image.jpg"
    service = NumberPlateService()

    
    
    with open(test_image_path, 'rb') as f:
        image_bytes = f.read()
    
    test_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    test_request = NumberPlateRequest(image_base64=test_base64)
    
    result = service.recognize_plate(request=test_request)
    
    print(f"Area: {result.area}")
    print(f"Number: {result.number}")