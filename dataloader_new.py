import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import cv2
import numpy as np
import os

class BoundingBoxDataset(Dataset):
    def __init__(self, root_dir, resize_size=(224, 224), max_depth=1.0):
        self.root_dir = root_dir
        self.resize_size = resize_size
        self.max_depth = max_depth
        self.data = self._prepare_data()  # Prepare data in the constructor

        self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def _prepare_data(self):
        """Prepares a list of tuples (image_path, pointcloud_path, mask, bbox) for each object."""
        data = []
        folders = sorted(os.listdir(self.root_dir))

        for folder in folders:
            folder_path = os.path.join(self.root_dir, folder)

            image_path = os.path.join(folder_path, "rgb.jpg")
            pointcloud_path = os.path.join(folder_path, "pc.npy")
            mask_path = os.path.join(folder_path, "mask.npy")
            bbox_path = os.path.join(folder_path, "bbox3d.npy")

            masks = np.load(mask_path)
            bboxes = np.load(bbox_path)

            for i in range(masks.shape[0]):
                data.append((image_path, pointcloud_path, masks[i], bboxes[i]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, pointcloud_path, mask, bbox = self.data[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        point_cloud = np.load(pointcloud_path)

        image = self.image_transform(image)
        mask = torch.from_numpy(mask).long()
        mask = self.mask_transform(mask).squeeze(0)

        H_orig, W_orig = point_cloud.shape[1], point_cloud.shape[2]
        H_new, W_new = self.resize_size
        row_scale = float(H_new) / H_orig
        col_scale = float(W_new) / W_orig

        row_indices, col_indices = np.meshgrid(np.arange(H_new), np.arange(W_new), indexing='ij')
        row_indices = (row_indices / row_scale).astype(int)
        col_indices = (col_indices / col_scale).astype(int)

        row_indices = np.clip(row_indices, 0, H_orig - 1)
        col_indices = np.clip(col_indices, 0, W_orig - 1)

        point_cloud_resized = point_cloud[:, row_indices, col_indices]
        point_cloud_resized = torch.from_numpy(point_cloud_resized).float()

        if self.max_depth is not None:
            point_cloud_resized /= self.max_depth

        return {
            "image": image,
            "point_cloud": point_cloud_resized,
            "mask": mask,
            "bbox": torch.tensor(bbox, dtype=torch.float32)
        }


# Example usage:
root_dir = "F:\RWTH\Job Applications\Full Time Job Applications\Sereact\sereact\dataset\dl_challenge"  # Replace with your root directory
dataset = BoundingBoxDataset(root_dir, max_depth=1.0)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)  # 80% for training
val_size = int(0.1 * dataset_size)    # 10% for validation
test_size = dataset_size - train_size - val_size  # Remaining for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

