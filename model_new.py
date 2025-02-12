import torch
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel

class MaskEncoder(nn.Module):
    """Processes the mask using a CNN to extract spatial features."""
    def __init__(self):
        super(MaskEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 128)

    def forward(self, mask):
        x = self.conv(mask)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class PointCloudEncoder(nn.Module):
    """Processes the point cloud using a CNN-based approach (ResNet-18)."""
    def __init__(self):
        super(PointCloudEncoder, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, 256)

    def forward(self, point_cloud):
        return self.resnet(point_cloud)

class ImageEncoder(nn.Module):
    """Extracts features from the image using Vision Transformer (ViT)."""
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.vit.config.hidden_size, 256)

    def forward(self, image):
        features = self.vit(image).last_hidden_state[:, 0, :]  # CLS token
        return self.fc(features)

class BoundingBoxPredictor(nn.Module):
    """Final prediction head that outputs the 3D bounding box coordinates."""
    def __init__(self, input_dim):
        super(BoundingBoxPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 8 * 3)  # Output shape (8,3)
        )

    def forward(self, x):
        return self.fc(x).view(-1, 8, 3)  # Reshape to (batch_size, 8, 3)

class MultiModal3DBoundingBoxModel(nn.Module):
    """Full model integrating image, point cloud, and mask features."""
    def __init__(self):
        super(MultiModal3DBoundingBoxModel, self).__init__()
        self.image_encoder = ImageEncoder()
        self.point_cloud_encoder = PointCloudEncoder()
        self.mask_encoder = MaskEncoder()
        self.fusion_fc = nn.Linear(256 + 256 + 128, 512)  # Combining features
        self.predictor = BoundingBoxPredictor(512)

    def forward(self, image, point_cloud, mask):
        image_features = self.image_encoder(image)  # (batch, 256)
        point_cloud_features = self.point_cloud_encoder(point_cloud)  # (batch, 256)
        mask_features = self.mask_encoder(mask)  # (batch, 128)

        fused_features = torch.cat([image_features, point_cloud_features, mask_features], dim=1)
        fused_features = self.fusion_fc(fused_features)
        bbox = self.predictor(fused_features)

        return bbox
