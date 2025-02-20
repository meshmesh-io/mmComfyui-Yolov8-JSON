import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os
from urllib.parse import urlparse
import logging
from torch.hub import download_url_to_file
import cv2
from torchvision.transforms import functional as F

logger = logging.getLogger("Comfyui-Yolov8-JSON")
yolov8_model_dir_name = "yolov8"
yolov8_model_list = {
    "yolov8n(6.23MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    },
    "yolov8s(21.53MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt",
    },
    "yolov8m (49.70MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt",
    },
    "yolov8l (83.70MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt",
    },
    "yolov8x (130.53)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt",
    },
    "yolov8n-seg (6.73MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt",
    },
    "yolov8s-seg(22.79MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-seg.pt",
    },
    "yolov8m-seg  (52.36MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-seg.pt",
    },
    "yolov8l-seg  (88.11MB)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-seg.pt",
    },
    "yolov8x-seg  (137.40)": {
        "model_url": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-seg.pt",
    },
}

labelName = {
    0: "person",  
    1: "bicycle",  
    2: "car", 
    3: "motorcycle",  
    4: "airplane", 
    5: "bus",  
    6: "train", 
    7: "truck",  
    8: "boat", 
    9: "traffic light",  
    10: "fire hydrant",  
    11: "stop sign", 
    12: "parking meter", 
    13: "bench", 
    14: "bird",  
    15: "cat",  
    16: "dog",  
    17: "horse",  
    18: "sheep",  
    19: "cow",  
    20: "elephant", 
    21: "bear",  
    22: "zebra",  
    23: "giraffe",  
    24: "backpack",  
    25: "umbrella",  
    26: "handbag",  
    27: "tie",  
    28: "suitcase",  
    29: "frisbee",  
    30: "skis",  
    31: "snowboard",  
    32: "sports ball",  
    33: "kite",  
    34: "baseball bat",  
    35: "baseball glove",  
    36: "skateboard",  
    37: "surfboard", 
    38: "tennis racket",  
    39: "bottle",  
    40: "wine glass",  
    41: "cup",  
    42: "fork",  
    43: "knife",  
    44: "spoon",  
    45: "bowl",  
    46: "banana",  
    47: "apple",  
    48: "sandwich",  
    49: "orange",  
    50: "broccoli",  
    51: "carrot", 
    52: "hot dog", 
    53: "pizza",  
    54: "donut",  
    55: "cake",  
    56: "chair",  
    57: "couch",  
    58: "potted plant",  
    59: "bed",  
    60: "dining table",  
    61: "toilet",  
    62: "tv", 
    63: "laptop",  
    64: "mouse",  
    65: "remote",  
    66: "keyboard",  
    67: "cell phone",  
    68: "microwave",  
    69: "oven", 
    70: "toaster", 
    71: "sink", 
    72: "refrigerator",  
    73: "book",
    74: "clock", 
    75: "vase",  
    76: "scissors", 
    77: "teddy bear",  
    78: "hair drier",  
    79: "toothbrush",
}

def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f"using extra model: {destination}")
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f"downloading {url} to {destination}")
        download_url_to_file(url, destination)
    return destination

def get_classes(label):
    label = label.lower()
    labels = label.split(",")
    result = []
    for l in labels:
        for key, value in labelName.items():
            if l == value:
                result.append(key)
                break
    return result

def get_yolov8_label_list():
    result = []
    for key, value in labelName.items():
        result.append(value)
    return result


def get_model_list():
    input_dir = folder_paths.get_input_directory()
    files = []
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            file_parts = f.split('.')
            if len(file_parts) > 1 and (file_parts[-1] == "pt"):
                files.append(f)
    return sorted(files)

def list_yolov8_model():
    return list(yolov8_model_list.keys())

def load_yolov8_model(model_name):
    yolov8_checkpoint_path = get_local_filepath(
        yolov8_model_list[model_name]["model_url"], yolov8_model_dir_name)
    model_file_name = os.path.basename(yolov8_checkpoint_path)
    model = YOLO(yolov8_checkpoint_path)
    return model

def load_yolov8_model_path(yolov8_checkpoint_path):
    model_file_name = os.path.basename(yolov8_checkpoint_path)
    model = YOLO(yolov8_checkpoint_path)
    return model

def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        return True if allow_url else "URLs are unsupported for this path"
    if not os.path.isfile(path.strip("\"")):
        return "Invalid file path: {}".format(path)
    if not path.endswith('.pt'):
        return "Invalid file extension. Only .pt files are supported."
    return True

# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    # Larger video files were taking >.5 seconds to hash even when cached,
    # so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

def change_mask_color(mask, color):
    # Assume mask is a binary mask with values 0 and 1
    thresholded_mask = torch.where(mask > 0.5, torch.tensor(1.0, device=mask.device), torch.tensor(0.0, device=mask.device))

    # Create a colored mask
    colored_mask = torch.zeros((3, mask.shape[1], mask.shape[2]), device=mask.device)  # Initialize with zeros
    for i in range(3):  # Apply color to the mask
        colored_mask[i, :, :] = thresholded_mask * color[i]

    return colored_mask


def yolov8_segment(model, image, label_name, threshold):
    # Assuming image is a PIL Image or a NumPy array that needs to be converted to a tensor
    if isinstance(image, np.ndarray):
        # Convert from HWC to CHW format if it's a numpy array
        image = torch.from_numpy(image.transpose((2, 0, 1)))
    elif isinstance(image, Image.Image):
        # Convert from PIL Image to tensor
        image = F.to_tensor(image)

    # Check if the image tensor is already normalized (values between 0 and 1)
    if image.max() > 1:
        image = image.float() / 255.

    # Resize the image to make its dimensions divisible by 32
    height, width = image.shape[1:3]
    new_height = ((height // 32) + 1) * 32 if height % 32 else height
    new_width = ((width // 32) + 1) * 32 if width % 32 else width

    # If the dimensions are not divisible by 32, resize the image
    if height != new_height or width != new_width:
        image = F.resize(image, size=(new_height, new_width), interpolation=Image.BILINEAR)

    # Add a batch dimension if it's missing
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    # Convert image to batch, channels, height, width format
    if image.shape[0] == 3 and len(image.shape) == 4:
        image = image.permute(0, 3, 1, 2)

    # Now the image is in the correct format for the model
    if label_name is not None:
        classes = get_classes(label_name)
    else:
        classes = []
    
    results = model(image, classes=classes, conf=threshold)

    # Iterate through each detection and apply unique color masks
    for result in results:
        masks = result.masks.data
        for idx, mask in enumerate(masks):
            print('mask', idx)
            # Retrieve the class index for the current mask
            #class_id = result.masks.indices[idx]  # Assuming this gives you the class ID of each mask
            color = (0, 0, 255)  # Get the unique color for this class, default to white if not found
            
            # Convert single-channel mask to a 3-channel color mask
            colored_mask = change_mask_color(mask, color).cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            
            print('colored_mask', colored_mask)
            # Apply colored mask onto the original image
            mask_area = mask.cpu().numpy().astype(bool)
            image[mask_area] = image[mask_area] * (1 - 0.5) + colored_mask[mask_area] * 0.5

    # Convert the numpy image with colored masks back to a tensor
    image_tensor_out = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    print('image_tensor_out', 'sending')
    return image_tensor_out

def yolov8_detect(model, image, label_name, json_type, threshold):
    image_tensor = image
    image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
    image = Image.fromarray(
        (image_np.squeeze(0) * 255).astype(np.uint8)
    )  # Convert float [0,1] tensor to uint8 image

    if label_name is not None:
        classes = get_classes(label_name)
    else:
        classes = []
    results = model(image, classes=classes, conf=threshold)

    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

    image_tensor_out = torch.tensor(
        np.array(im).astype(np.float32) / 255.0
    )  # Convert back to CxHxW
    image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

    yolov8_json = []
    res_mask = []
    for result in results:
        labelme_data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": result.path,
            "imageData": None,
            "imageHeight": result.orig_shape[0],
            "imageWidth": result.orig_shape[1],
        }
        mask = np.zeros((result.orig_shape[0], result.orig_shape[1], 1), dtype=np.uint8)
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = labelName[int(box.cls)]
            points = [[x1, y1], [x2, y2]]
            shape = {
                "label": label,
                "points": points,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
            json = [label, x1, y1, x2, y2]
            yolov8_json.append(json)
            labelme_data["shapes"].append(shape)
            cv2.rectangle(
                mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1
            )
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0
        res_mask.append(mask_tensor)

    if json_type == "Labelme":
        json_data = labelme_data
    else:
        json_data = yolov8_json

    return (image_tensor_out, json_data, res_mask)


class LoadYolov8Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_yolov8_model(),),
            },
        }
    CATEGORY = "Comfyui-Yolov8-JSON"
    FUNCTION = "main"
    RETURN_TYPES = ("YOLOV8_MODEL", )

    def main(self, model_name):
        model = load_yolov8_model(model_name)
        return (model,)

class LoadYolov8ModelFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {"default": "/ComfyUI/models/yolov8/yolov8l.pt",}
                ),
            },
        }

    CATEGORY = "Comfyui-Yolov8-JSON"
    FUNCTION = "main"
    RETURN_TYPES = ("YOLOV8_MODEL",)

    def main(self, model_path):
        model_path = folder_paths.get_annotated_filepath(model_path.strip('"'))
        if model_path is None or validate_path(model_path) != True:
            raise Exception("model is not a valid path: " + model_path)
        model = load_yolov8_model_path(model_path)
        return (model,)

    @classmethod
    def IS_CHANGED(s, model_path):
        model_path = folder_paths.get_annotated_filepath(model_path)
        return calculate_file_hash(model_path)

class ApplyYolov8Model:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolov8_model": ("YOLOV8_MODEL", {}),
                "image": ("IMAGE",),
                "detect": (
                    ["all", "choose", "input"],
                    {"default": "all"},
                ),
                "label_name": (
                    "STRING",
                    {"default": "person,cat,dog", "multiline": False},
                ),
                "label_list": (
                    get_yolov8_label_list(),
                    {"default": "person"},
                ),
                "json_type": (
                    ["Labelme", "yolov8"],
                    {"default": "Labelme"},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
        }

    CATEGORY = "Comfyui-Yolov8-JSON"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "JSON", "MASK")

    def main(self, yolov8_model, image, detect , label_name,label_list,json_type, threshold):
        res_images = []
        res_jsons = []
        res_masks = []
        for item in image:
            # Check and adjust image dimensions if needed
            if len(item.shape) == 3:
                item = item.unsqueeze(0)  # Add a batch dimension if missing

            label=None
            if(detect == "choose"):
                label=label_list
            else:
                label=label_name

            image_out, json, masks = yolov8_detect(
                yolov8_model, item, label, json_type, threshold
            )
            res_images.append(image_out)
            res_jsons.extend(json)
            res_masks.extend(masks)
        return (torch.cat(res_images, dim=0), res_jsons, torch.cat(res_masks, dim=0))


class ApplyYolov8ModelSeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "yolov8_model": ("YOLOV8_MODEL", {}),
                "image": ("IMAGE",),
                "detect": (
                    ["all", "choose", "input"],
                    {"default": "all"},
                ),
                "label_name": (
                    "STRING",
                    {"default": "person,cat,dog", "multiline": False},
                ),
                "label_list": (
                    get_yolov8_label_list(),
                    {"default": "person"},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
            },
        }

    CATEGORY = "Comfyui-Yolov8-JSON"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(
        self, yolov8_model, image, detect, label_name, label_list, threshold
    ):
        print('custom yolo8 segmentation function')
        res_images = []
        res_masks_colored = []
        res_masks = []  # No need to concatenate masks anymore
        for item in image:
            # Check and adjust image dimensions if needed
            if len(item.shape) == 3:
                item = item.unsqueeze(0)  # Add a batch dimension if missing

            label = None
            if detect == "choose":
                label = label_list
            else:
                label = label_name

            image_out, masks = yolov8_segment(yolov8_model, item, label, threshold)
            res_images.append(image_out)

            for mask in masks:
                # Change color of the mask here, assuming it's a binary mask
                colored_mask = change_mask_color(mask)
                res_masks_colored.append(colored_mask)
                res_masks.append(mask)  # Append the original mask too

        return (torch.cat(res_images, dim=0), torch.cat(res_masks_colored, dim=0))
