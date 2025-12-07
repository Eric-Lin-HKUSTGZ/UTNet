"""
Example usage of metrics and visualization modules
"""
import torch
import numpy as np
from metrics import compute_mpjpe, compute_pa_mpjpe, Evaluator
from visualization import render_hand_keypoints, MeshRenderer


def example_metrics():
    """Example of computing MPJPE and PA-MPJPE"""
    # Create dummy predictions and ground truth
    batch_size = 4
    num_joints = 21
    
    # Predicted joints (B, N, 3) in meters
    pred_joints = torch.randn(batch_size, num_joints, 3) * 0.1
    
    # Ground truth joints (B, N, 3) in meters
    gt_joints = pred_joints + torch.randn(batch_size, num_joints, 3) * 0.01
    
    # Compute MPJPE
    mpjpe = compute_mpjpe(pred_joints, gt_joints)
    print(f"MPJPE: {mpjpe.mean():.3f} mm (per sample: {mpjpe})")
    
    # Compute PA-MPJPE
    pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)
    print(f"PA-MPJPE: {pa_mpjpe.mean():.3f} mm (per sample: {pa_mpjpe})")


def example_evaluator():
    """Example of using the Evaluator class"""
    dataset_length = 100
    batch_size = 4
    num_joints = 21
    
    # Initialize evaluator
    evaluator = Evaluator(
        dataset_length=dataset_length,
        keypoint_list=None,  # Use all keypoints
        root_joint_idx=0,  # Wrist is root
        metrics=['mpjpe', 'pa_mpjpe'],
        save_predictions=False
    )
    
    # Simulate evaluation loop
    for i in range(0, dataset_length, batch_size):
        # Create dummy predictions and ground truth
        pred_joints = torch.randn(batch_size, num_joints, 3) * 0.1
        gt_joints = pred_joints + torch.randn(batch_size, num_joints, 3) * 0.01
        
        # Evaluate batch
        batch_metrics = evaluator(pred_joints, gt_joints)
        print(f"Batch {i//batch_size}: MPJPE={batch_metrics['mpjpe'].mean():.3f} mm, "
              f"PA-MPJPE={batch_metrics['pa_mpjpe'].mean():.3f} mm")
    
    # Print final results
    evaluator.log()
    
    # Get metrics dictionary
    metrics_dict = evaluator.get_metrics_dict()
    print(f"\nFinal metrics: {metrics_dict}")


def example_visualization():
    """Example of visualizing keypoints"""
    import cv2
    
    # Create dummy image
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    # Create dummy keypoints (21 keypoints, format: [x, y, confidence])
    keypoints = np.random.rand(21, 3) * 256
    keypoints[:, 2] = 1.0  # All keypoints are visible
    
    # Render keypoints
    img_with_keypoints = render_hand_keypoints(img, keypoints)
    
    # Save or display
    cv2.imwrite('example_keypoints.png', img_with_keypoints)
    print("Saved visualization to example_keypoints.png")


if __name__ == '__main__':
    print("=== Example: Computing Metrics ===")
    example_metrics()
    
    print("\n=== Example: Using Evaluator ===")
    example_evaluator()
    
    print("\n=== Example: Visualization ===")
    example_visualization()


