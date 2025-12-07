"""
Hand Detection Module using WiLoR Pre-trained Detector
参考 WiLoR/demo.py 和 WiLoR/gradio_demo.py
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from ultralytics import YOLO


class HandDetector:
    """
    Hand detector using WiLoR pre-trained YOLO model
    """
    def __init__(self, detector_path: str = '/data0/users/Robert/linweiquan/WiLoR/pretrained_models/detector.pt',
                 conf_threshold: float = 0.3, iou_threshold: float = 0.5):
        """
        Args:
            detector_path: path to WiLoR pre-trained detector
            conf_threshold: confidence threshold for detection
            iou_threshold: IoU threshold for NMS
        """
        self.detector = YOLO(detector_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        # Freeze detector parameters
        self.detector.eval()
        for param in self.detector.parameters():
            param.requires_grad = False

    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect hands in image
        Args:
            image: RGB image (H, W, 3) in BGR format (cv2 format)
        Returns:
            bboxes: list of bounding boxes [[x1, y1, x2, y2], ...]
            is_right: list of flags indicating right hand (1.0) or left hand (0.0)
        """
        detections = self.detector(image, conf=self.conf_threshold, verbose=False, iou=self.iou_threshold)[0]
        
        bboxes = []
        is_right = []
        
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            if len(bbox.shape) == 1:
                bboxes.append(bbox[:4].tolist())
                is_right.append(float(det.boxes.cls.cpu().detach().squeeze().item()))
            else:
                # Multiple detections
                for i in range(bbox.shape[0]):
                    bboxes.append(bbox[i, :4].tolist())
                    is_right.append(float(det.boxes.cls[i].cpu().detach().squeeze().item()))
        
        return bboxes, is_right

    def crop_hand(self, image: np.ndarray, bbox: List[float], 
                  target_size: Tuple[int, int] = (256, 256),
                  rescale_factor: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop and resize hand region from image
        Args:
            image: RGB image (H, W, 3) in BGR format
            bbox: bounding box [x1, y1, x2, y2]
            target_size: target output size (H, W)
            rescale_factor: factor for padding the bbox
        Returns:
            cropped_image: cropped and resized image (H, W, 3)
            transform_matrix: transformation matrix (3, 3)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        
        # Rescale bbox
        scale = rescale_factor * max(width, height) / 200.0
        bbox_size = scale * 200.0
        
        # Generate image patch
        patch_width, patch_height = target_size
        
        # Crop and resize
        img_patch, trans = self._generate_image_patch(
            image, center_x, center_y, bbox_size, bbox_size,
            patch_width, patch_height, flip=False, scale=1.0, rot=0.0
        )
        
        return img_patch, trans

    def _generate_image_patch(self, img, center_x, center_y, width, height,
                               patch_width, patch_height, flip, scale, rot):
        """Generate image patch with transformation"""
        # This is a simplified version, full implementation would use
        # the same logic as WiLoR's generate_image_patch_cv2
        from skimage.filters import gaussian
        
        # Apply blur to avoid aliasing
        downsampling_factor = (width * 1.0) / patch_width / 2.0
        if downsampling_factor > 1.1:
            img = gaussian(img, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)
        
        # Calculate crop bounds
        x1 = int(center_x - width / 2)
        y1 = int(center_y - height / 2)
        x2 = int(center_x + width / 2)
        y2 = int(center_y + height / 2)
        
        # Crop
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        cropped = img[y1:y2, x1:x2]
        
        # Resize
        if cropped.size > 0:
            resized = cv2.resize(cropped, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = np.zeros((patch_height, patch_width, 3), dtype=img.dtype)
        
        # Create transformation matrix
        trans = np.eye(3)
        trans[0, 2] = -x1
        trans[1, 2] = -y1
        scale_mat = np.eye(3)
        scale_mat[0, 0] = patch_width / max(x2 - x1, 1)
        scale_mat[1, 1] = patch_height / max(y2 - y1, 1)
        trans = scale_mat @ trans
        
        if flip:
            resized = resized[:, ::-1]
        
        return resized, trans

    def process_batch(self, images: List[np.ndarray]) -> List[Tuple[List[np.ndarray], List[float]]]:
        """
        Process a batch of images
        Args:
            images: list of RGB images
        Returns:
            list of (bboxes, is_right) tuples for each image
        """
        results = []
        for img in images:
            bboxes, is_right = self.detect(img)
            results.append((bboxes, is_right))
        return results


