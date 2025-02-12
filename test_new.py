import torch
import cv2
import numpy as np
import open3d as o3d  # For point cloud visualization
import torch.nn as nn
import logging
from train_new import MultiModal3DBoundingBoxModel  # Assuming your model class is defined here
from one import test_dataloader  # Import your test_dataloader here

# Set up logging
logging.basicConfig(filename='test_results.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Visualization functions
def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, image_width=224, image_height=224):
    """
    Draws the projected 3D bounding box onto the 2D image.
    bbox: (8,3) numpy array representing the 3D bounding box coordinates in normalized coordinates.
    """
    # Convert normalized bbox (centered at image center) to pixel coordinates
    # Rescale coordinates from [-1, 1] to image pixel values
    bbox[:, 0] = ((bbox[:, 0] + 1) / 2) * image_width  # x-axis (width)
    bbox[:, 1] = ((bbox[:, 1] + 1) / 2) * image_height  # y-axis (height)
    bbox[:, 2] = bbox[:, 2]  # z-axis remains unchanged for 2D projection, unless you use it for depth visualization

    bbox = np.int32(bbox[:, :2])  # Convert to 2D by taking only x, y
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting vertical edges
    ]
    for edge in edges:
        cv2.line(image, tuple(bbox[edge[0]]), tuple(bbox[edge[1]]), color, thickness)
    return image

def visualize_point_cloud(point_cloud, gt_bbox, pred_bbox):
    """
    Visualizes the point cloud with ground truth and predicted bounding boxes.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.reshape(-1, 3))

    gt_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    pred_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.utility.Vector3dVector(gt_bbox)
    gt_lineset.lines = o3d.utility.Vector2iVector(gt_lines)
    gt_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(gt_lines))])  # Green for GT

    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_bbox)
    pred_lineset.lines = o3d.utility.Vector2iVector(pred_lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(pred_lines))])  # Red for Prediction

    o3d.visualization.draw_geometries([pcd, gt_lineset, pred_lineset])

# Load your pre-trained model
def load_model(model_path='model.pth'):
    model = MultiModal3DBoundingBoxModel()  # Replace this with your actual model class
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# The test function with logging
def test(model, test_loader, image_width=224, image_height=224):
    total_test_loss = 0.0
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss
    with torch.no_grad():  # No gradients needed for testing
        for i, batch in enumerate(test_loader):
            image = batch["image"]
            point_cloud = batch["point_cloud"]
            mask = batch["mask"].unsqueeze(1)
            bbox = batch["bbox"]

            # Perform model prediction
            predicted_bbox = model(image, point_cloud, mask)
            print('Predicted bbox = ', predicted_bbox)
            print('Ground truth Bbox = ', bbox)
            # Compute the loss
            loss = criterion(predicted_bbox, bbox)
            total_test_loss += loss.item()

            # Rescale bounding boxes to original image size if needed
            img_np = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            gt_bbox = bbox[0].cpu().numpy()  # (8,3) for ground truth bbox
            print('Ground truth bbox shape = ', gt_bbox.shape)
            pred_bbox = predicted_bbox[0].cpu().numpy()  # (8,3) for predicted bbox
            print('Predicted bbox shape = ', pred_bbox.shape)
            print('___________________________________________________________')
            # Draw bounding boxes on image
            img_with_boxes = draw_bbox(img_np.copy(), gt_bbox, color=(0, 255, 0), image_width=image_width, image_height=image_height)  # Green for GT
            img_with_boxes = draw_bbox(img_with_boxes, pred_bbox, color=(0, 0, 255), image_width=image_width, image_height=image_height)  # Red for Prediction

            # Save visualization
            cv2.imwrite(f"test_results/test_img_{i}.jpg", img_with_boxes)

            # Point Cloud Visualization
            #visualize_point_cloud(point_cloud[0].cpu().numpy().reshape(3, -1).T, gt_bbox, pred_bbox)

    avg_test_loss = total_test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")  # Log to the file
    print(f"Test Loss: {avg_test_loss:.4f}")  # Print to the console

# Running the test
if __name__ == "__main__":
    model_path = 'checkpoints\model.pth_epoch_10.pth'  # Path to your saved model weights
    model = load_model(model_path)

    # Test the model
    test(model, test_dataloader)  # Pass your test dataloader
