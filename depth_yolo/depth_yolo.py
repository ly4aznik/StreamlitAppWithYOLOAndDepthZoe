from typing import Optional
import torch
import numpy as np
import cv2
from .nms import non_max_suppression
from torchvision import transforms

class_names = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'bus',
    5: 'train',
    6: 'truck'
}


def draw_predictions(image, predictions, input_size=(640, 640)):
    input_w, input_h = input_size
    orig_h, orig_w = image.shape[:2]
    scale_x, scale_y = orig_w / input_w, orig_h / input_h

    for *xyxy, conf, cls in predictions:
        x1, y1, x2, y2 = [int(x.item() * scale_x if i % 2 == 0 else x.item() * scale_y)
                          for i, x in enumerate(xyxy)]
        class_id = int(cls)
        label = f"{class_names.get(class_id, class_id)} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

class DepthYOLO:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(self.device)
        self.model.conf = confidence_threshold
        self.model.eval()


        self.depth_model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_K", pretrained=True).to(self.device)
        self.depth_model.eval()

    def __get_depth_channel(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((384, 512)),
            transforms.ToTensor(),
        ])
        inp = transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            depth_map = self.depth_model(inp)["metric_depth"].squeeze().cpu().numpy()
        dmin, dmax = depth_map.min(), depth_map.max()
        norm = ((depth_map - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
        return norm

    def __prepare_image_for_yolo(self, original_image, depth_channel=None):
        original_image_resized = cv2.resize(original_image, (640, 640))
        if depth_channel is not None:
            depth_channel_resized = cv2.resize(depth_channel, (640, 640))
            combined_image = cv2.merge([original_image_resized, depth_channel_resized])
            img = combined_image.astype(np.float32) / 255.0
        else:
            img = original_image_resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # (1, C, H, W)
        return torch.from_numpy(img)

    def __prepare_depth_map_for_print(self, depth_channel, height, width):
        resized = cv2.resize(depth_channel, (width, height))
        norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

    def process_image_with_depth_map(self, img: np.ndarray):
        depth_neg = self.__get_depth_channel(img)
        depth_img = cv2.bitwise_not(depth_neg)
        tensor = self.__prepare_image_for_yolo(img, depth_img)
        depth_img = self.__prepare_depth_map_for_print(depth_img, *img.shape[:2])
        depth_img_rgb = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
        results = self.model(tensor)
        pred = non_max_suppression(results.clone(), conf_thres=self.model.conf)[0].cpu().numpy()
        img_boxes = draw_predictions(img.copy(), pred)
        depth_boxes = draw_predictions(depth_img_rgb.copy(), pred)
        return img_boxes, depth_boxes

    def process_image(self, img: np.ndarray):
        depth_neg = self.__get_depth_channel(img)
        depth_img = cv2.bitwise_not(depth_neg)
        tensor = self.__prepare_image_for_yolo(img, depth_img)
        results = self.model(tensor)
        pred = non_max_suppression(results.clone(), conf_thres=self.model.conf)[0].cpu().numpy()
        img_boxes = draw_predictions(img.copy(), pred)
        return img_boxes

    def change_confidence(self, conf):
        self.model.conf = conf
