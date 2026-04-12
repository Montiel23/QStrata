import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_medical_data(data_flag='pathmnist', n_components=4, n_samples=None):
    "load mnist and prepare quantum processing"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # #standardize medical images
    # transform = transforms.Compose([
    #     transforms.Resize((size, size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])

    # train_dataset = DataClass(split="train", transform=transform, download=True)
    train_dataset = DataClass(split="train", download=True)
    # val_dataset = DataClass(split="val", transform=transform, download=True)
    val_dataset = DataClass(split="val", download=True)
    # test_dataset = DataClass(split="test", transform=transform, download=True)
    test_dataset = DataClass(split="test", download=True)

    def process_split(dataset, samples):
        #access raw numpy images and labels
        imgs = dataset.imgs
        labels = dataset.labels.flatten()

        #determine samples
        actual_samples = min(samples, len(imgs)) if samples else len(imgs)

        #shuffle indices to get diverse slice
        indices = np.random.permutation(len(imgs))[:actual_samples]

        selected_imgs = imgs[indices].astype(np.float32)
        selected_labels = labels[indices]

        if len(selected_imgs.shape) == 4 and selected_imgs.shape[3] == 3:
            selected_imgs = np.mean(selected_imgs, axis=3)

        #normalize to [0, 1] for better pca scaling
        selected_imgs /= 255.0

        return selected_imgs, selected_labels


    #slice raw images
    X_train_raw, y_train = process_split(train_dataset, n_samples)
    val_size = n_samples // 5 if n_samples else None
    X_val_raw, y_val = process_split(val_dataset, val_size)
    X_test_raw, y_test = process_split(test_dataset, val_size)

    # pca pipeline (flatten -> scale -> pca)

    X_train_flat = X_train_raw.reshape(len(X_train_raw), - 1).astype(float)
    X_val_flat = X_val_raw.reshape(len(X_val_raw), -1).astype(float)
    X_test_flat = X_test_raw.reshape(len(X_test_raw), -1).astype(float)

    #fit the PCA on Train only
    # scaler = StandardScaler()
    quantum_scaler = MinMaxScaler(feature_range=(0, np.pi))
    # X_train_scaled = scaler(X_train_flat)
    X_train_scaled = quantum_scaler.fit_transform(X_train_flat)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    #transform val and test using train parameters
    X_val_pca = pca.transform(quantum_scaler.transform(X_val_flat))
    X_test_pca = pca.transform(quantum_scaler.transform(X_test_flat))

    data = {
        'train': (X_train_pca, y_train),
        'val': (X_val_pca, y_val),
        'test': (X_test_pca, y_test),
        'n_classes': len(info['label']),
        'original_images': X_train_raw
    }

    return data, pca, quantum_scaler