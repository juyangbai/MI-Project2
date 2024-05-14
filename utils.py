import os
import zipfile
from tqdm import tqdm
import requests

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
from datasets import load_dataset, DatasetDict
from transformers import ViTImageProcessor


"Range: -1.9886685609817505, 2.12648868560791"
class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        # self.hparams = args
        self.save_hyperparameters(args)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
    
    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [T.ToTensor(), 
            T.Normalize(self.mean, self.std)]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

class CIFAR10ViT():
    
    def __init__(self):        
        # load cifar10 (only small portion for demonstration purposes)
        train_ds, self.test_ds = load_dataset('cifar10', split=['train', 'test'])

        # split up training into training + validation
        splits = train_ds.train_test_split(test_size=0.1)
        self.train_ds = splits['train']
        self.val_ds = splits['test']

        # Initialize the processor for image transformations
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        image_mean, image_std = self.processor.image_mean, self.processor.image_std
        size = self.processor.size["height"]

        # Set up the image transformations
        self._train_transform = T.Compose(
            [
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ]
        )
        self._val_transform = T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=image_mean, std=image_std),
            ]
        )

        self.id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
        self.label2id = {label:id for id,label in self.id2label.items()}
    
    def train_transforms(self, examples):
        examples['pixel_values'] = [self._train_transform(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(self, examples):
        examples['pixel_values'] = [self._val_transform(image.convert("RGB")) for image in examples['img']]
        return examples

    def get_datasets(self):

        self.train_ds.set_transform(self.train_transforms)
        self.val_ds.set_transform(self.val_transforms)
        self.test_ds.set_transform(self.val_transforms)

        # Return the datasets
        return self.train_ds, self.val_ds, self.test_ds

# CIFAR10VIT
    # class CIFAR10ViT():
    #     def __init__(self):
    #         # Load cifar10 (only a small portion for demonstration purposes)
    #         train_ds, self.test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])

    #         # Split up training into training + validation
    #         splits = train_ds.train_test_split(test_size=0.1)
    #         self.train_ds = splits['train']
    #         self.val_ds = splits['test']

    #         # Initialize the processor for image transformations
    #         self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    #         image_mean, image_std = self.processor.image_mean, self.processor.image_std
    #         size = self.processor.size["height"]

    #         # Set up the image transformations
    #         self._train_transform = T.Compose([
    #             T.RandomResizedCrop(size),
    #             T.RandomHorizontalFlip(),
    #             T.ToTensor(),
    #             T.Normalize(mean=image_mean, std=image_std),
    #         ])
    #         self._val_transform = T.Compose([
    #             T.Resize(size),
    #             T.CenterCrop(size),
    #             T.ToTensor(),
    #             T.Normalize(mean=image_mean, std=image_std),
    #         ])

    #     def _apply_transforms(self, example):
    #         """Applies the appropriate transformation to a single example."""
    #         return {'pixel_values': self._val_transform(example['img'].convert('RGB')), 'labels': example['label']}

    #     def get_datasets(self):
    #         # Use map to apply transformations to all elements in the dataset
    #         self.train_ds = self.train_ds.map(lambda example: self._apply_transforms(example), remove_columns=['img'])
    #         self.val_ds = self.val_ds.map(lambda example: self._apply_transforms(example), remove_columns=['img'])
    #         self.test_ds = self.test_ds.map(lambda example: self._apply_transforms(example), remove_columns=['img'])

    #         # Return the datasets
    #         return DatasetDict({'train': self.train_ds, 'validation': self.val_ds, 'test': self.test_ds})


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return dict(accuracy=accuracy_score(predictions, labels))


class AdversarialDataset(Dataset):
    def __init__(self, dataset, attack_fn):
        """
        Args:
            dataset: The original dataset.
            attack_fn: Function to generate adversarial examples.
        """
        self.dataset = dataset
        self.attack_fn = attack_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve the original data
        item = self.dataset[idx]
        image, label = item['pixel_values'], item['label']

        # Apply the adversarial attack
        adv_image = self.attack_fn(image)

        # Return the adversarial image and the label
        return {'pixel_values': adv_image, 'labels': label}

if __name__ == '__main__':

    # data = CIFAR10Data()
    # train_dataloader = data.train_dataloader()
    # val_dataloader = data.val_dataloader()
    # test_dataloader = data.test_dataloader()

    data = CIFAR10ViT()
    train_dataset, val_dataset, test_dataset = data.get_datasets()