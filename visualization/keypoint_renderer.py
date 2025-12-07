"""
Keypoint visualization utilities
Based on WiLoR's render_openpose.py
"""
import cv2
import math
import numpy as np
from typing import List, Tuple, Optional


def get_keypoints_rectangle(keypoints: np.ndarray, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.ndarray): Keypoint array of shape (N, 2) or (N, 3).
        threshold (float): Confidence visualization threshold (if keypoints have confidence).
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    if keypoints.shape[1] == 3:
        valid_ind = keypoints[:, -1] > threshold
        if valid_ind.sum() > 0:
            valid_keypoints = keypoints[valid_ind][:, :-1]
        else:
            return 0, 0, 0
    else:
        valid_keypoints = keypoints
    
    if len(valid_keypoints) == 0:
        return 0, 0, 0
    
    max_x = valid_keypoints[:, 0].max()
    max_y = valid_keypoints[:, 1].max()
    min_x = valid_keypoints[:, 0].min()
    min_y = valid_keypoints[:, 1].min()
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    return width, height, area


def render_keypoints(img: np.ndarray,
                     keypoints: np.ndarray,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.ndarray:
    """
    Render keypoints on input image.
    Args:
        img (np.ndarray): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.ndarray): Keypoint array of shape (N, 2) or (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
        alpha (float): Transparency factor.
    Returns:
        (np.ndarray): Image of shape (H, W, 3) with keypoints drawn on top of the original image.
    """
    img_orig = img.copy()
    height, width = img.shape[0], img.shape[1]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    # Add confidence if not present
    if keypoints.shape[1] == 2:
        keypoints_with_conf = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
    else:
        keypoints_with_conf = keypoints

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints_with_conf, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        # Draw lines between connected keypoints
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if (keypoints_with_conf[index1, -1] > threshold and 
                keypoints_with_conf[index2, -1] > threshold):
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints_with_conf[index1, :-1].astype(np.int32)
                keypoint2 = keypoints_with_conf[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), 
                        tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        
        # Draw keypoint circles
        for part in range(len(keypoints_with_conf)):
            faceIndex = part
            if keypoints_with_conf[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints_with_conf[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, 
                          tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    
    return img


def render_hand_keypoints(img: np.ndarray,
                          hand_keypoints: np.ndarray,
                          threshold: float = 0.1,
                          use_confidence: bool = False,
                          alpha: float = 1.0) -> np.ndarray:
    """
    Render hand keypoints in OpenPose format.
    Args:
        img (np.ndarray): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        hand_keypoints (np.ndarray): Hand keypoint array of shape (21, 2) or (21, 3).
        threshold (float): Confidence threshold for visualization.
        use_confidence (bool): Whether to use confidence values for visualization.
        alpha (float): Transparency factor.
    Returns:
        (np.ndarray): Image with hand keypoints drawn.
    """
    if use_confidence and hand_keypoints.shape[1] == 3:
        thicknessCircleRatioRight = 1./50 * np.ones(hand_keypoints.shape[0])
    else:
        thicknessCircleRatioRight = 1./50 * np.ones(hand_keypoints.shape[0])
    
    thicknessLineRatioWRTCircle = 0.75
    
    # Hand skeleton connections (21 keypoints: wrist + 5 fingers * 4 joints)
    # Format: [parent, child] pairs
    pairs = [0, 1,  1, 2,  2, 3,  3, 4,   # Thumb
             0, 5,  5, 6,  6, 7,  7, 8,   # Index
             0, 9,  9, 10, 10, 11, 11, 12,  # Middle
             0, 13, 13, 14, 14, 15, 15, 16,  # Ring
             0, 17, 17, 18, 18, 19, 19, 20]  # Pinky
    pairs = np.array(pairs).reshape(-1, 2)

    # Colors for each keypoint (21 keypoints)
    colors = [100.,  100.,  100.,  # Wrist (0)
              100.,    0.,    0.,  # Thumb (1-4)
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,  # Index (5-8)
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,  # Middle (9-12)
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,  # Ring (13-16)
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,  # Pinky (17-20)
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1, 3)
    
    poseScales = [1]
    
    img = render_keypoints(img, hand_keypoints, pairs, colors, 
                          thicknessCircleRatioRight, thicknessLineRatioWRTCircle, 
                          poseScales, threshold, alpha=alpha)
    return img


def render_keypoints_2d(img: np.ndarray,
                       pred_keypoints_2d: np.ndarray,
                       gt_keypoints_2d: Optional[np.ndarray] = None,
                       threshold: float = 0.1) -> np.ndarray:
    """
    Render 2D keypoints on image, optionally comparing prediction and ground truth.
    Args:
        img (np.ndarray): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        pred_keypoints_2d (np.ndarray): Predicted 2D keypoints of shape (N, 2) or (N, 3).
        gt_keypoints_2d (np.ndarray, optional): Ground truth 2D keypoints of shape (N, 2) or (N, 3).
        threshold (float): Confidence threshold.
    Returns:
        (np.ndarray): Image with keypoints drawn.
    """
    # Ensure image is in [0, 255] range
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    # Render ground truth if provided
    if gt_keypoints_2d is not None:
        # Add confidence if not present
        if gt_keypoints_2d.shape[1] == 2:
            gt_keypoints_with_conf = np.concatenate([gt_keypoints_2d, np.ones((gt_keypoints_2d.shape[0], 1))], axis=1)
        else:
            gt_keypoints_with_conf = gt_keypoints_2d
        
        # Render GT in green
        img = render_hand_keypoints(img.copy(), gt_keypoints_with_conf, threshold=threshold)
    
    # Render prediction
    if pred_keypoints_2d.shape[1] == 2:
        pred_keypoints_with_conf = np.concatenate([pred_keypoints_2d, np.ones((pred_keypoints_2d.shape[0], 1))], axis=1)
    else:
        pred_keypoints_with_conf = pred_keypoints_2d
    
    # Render prediction in red (or overlay on GT)
    img = render_hand_keypoints(img, pred_keypoints_with_conf, threshold=threshold)
    
    return img


