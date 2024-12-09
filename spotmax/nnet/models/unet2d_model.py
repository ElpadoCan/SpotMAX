import random
import torch
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import logging
from pathlib import Path
import shutil as sh

from cellacdc import printl

from .base_model import BaseModel
from .unet2D.unet_2D_model import UNet2D
from .unet2D.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff

try:
    import wandb
    WANDB_INSTALLED = True
except Exception as err:
    WANDB_INSTALLED = False

class SegmentationDataset(Dataset):
    """Pytorch dataset for uploading images and masks to the model.

    Args:
        Dataset (Pytorch dataset): Pytorch dataset.
    """
    def __init__(self, imgs: np.ndarray, masks: np.ndarray, transform=None):

        self.imgs = imgs.reshape(imgs.shape[0], 1, imgs.shape[1], imgs.shape[2])
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.imgs[idx]
        mask = self.masks[idx]

        if self.transform is not None:

            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            img = self.transform(np.squeeze(img))

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            mask = self.transform(mask)

        return {"image": img, "mask": mask}


class Unet2DModel(BaseModel):
    """2D U-Net model class."""

    def __init__(
        self,
        config:dict) -> None:
        """Initialize the 2D U-Net model.

        Args:
            config (dict): Configuration dictionary.
        """

        self.model_config = config['model']
        self.trainer_config = config['trainer']
        device_str = config['device']
        self.device = torch.device(device_str)
        self.model_dir = os.path.expanduser(
            self.model_config['model_dir']
        )
        self.best_model_location = os.path.expanduser(
            self.model_config['best_model_path']
        )
        self.training_directory = os.path.expanduser(
            self.model_config['training_path']
        )
        self.net = None        

    def initialize_network(self):
        """Initialize the 2D U-Net model. This function is called by the train function."""
        self.net = UNet2D(
            n_channels=self.model_config['n_channels'],
            n_classes=self.model_config['n_classes'],
            bilinear=self.model_config['bilinear'],
        ).to(self.device)

    def train(
            self,
            X_train,
            y_train,
            X_val,
            y_val
        ):
        """Train the 2D U-Net model.

        Args:
            X_train (np.ndarray): Training images.
            y_train (np.ndarray): Training masks.
            X_val (np.ndarray): Validation images.
            y_val (np.ndarray): Validation masks.
        """

        # Get parameters from config
        batch_size = self.trainer_config['batch_size']
        epochs = self.trainer_config['epochs']
        learning_rate = self.trainer_config['learning_rate']
        save_checkpoint = self.trainer_config['save_checkpoint']
        amp = self.trainer_config['amp']
        if WANDB_INSTALLED:
            wandb_mode = self.trainer_config.get('wandb_mode', 'disabled')
        else:
            wandb_mode = 'disabled'

        # Initialize the network
        if not self.net:
            self.initialize_network()

        # Dataset
        n_train = len(X_train)
        train_set = SegmentationDataset(imgs=X_train, masks=y_train, transform=None)
        val_set = SegmentationDataset(imgs=X_val, masks=y_val, transform=None)

        # Create data loaders
        loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)

        # Setting Experiment
        experiment = wandb.init(project="spotMAX AI", entity="lfk", mode=wandb_mode)
        experiment.config.update(
            dict(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                val_percent=0,
                save_checkpoint=save_checkpoint,
                img_scale=False,
                amp=amp,
            )
        )

        # Optimizer
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-8, amsgrad=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=8)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss()
        global_step = 0

        # Clean previous checkpoints
        sh.rmtree(self.training_directory, ignore_errors=True, onerror=None)

        val_scores = []

        # Training
        for epoch in range(epochs):

            self.net.train()
            epoch_loss = 0

            # For each batch in the training set
            with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
                for batch in train_loader:

                    images = batch["image"].to(self.device, dtype=torch.float32)
                    true_masks = torch.squeeze(batch["mask"]).to(self.device, dtype=torch.long)

                    assert images.shape[1] == self.net.n_channels, (
                        f"Network has been defined with {self.net.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = self.net(images)
                        loss = (
                            criterion(masks_pred, true_masks) +
                            dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, self.net.n_classes)
                                .permute(0, 3, 1, 2).float(),
                                multiclass=True,
                            )
                        )

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({"train loss": loss.item(), "step": global_step, "epoch": epoch})
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Validation
                val_score = self._evaluate(val_loader)
                if not isinstance(val_score, int):
                    val_score = val_score.item()
                
                val_scores.append(val_score)
                scheduler.step(val_score)

                # Calculate loss over the validation dataset
                try:
                    val_loss = self._val_loss(val_loader, criterion)
                except Exception as err:
                    val_loss = np.inf

                # Calculate the average loss per epoch
                epoch_loss /= n_train // batch_size

                logging.info("Validation Dice score: {}".format(val_score))
                experiment.log(
                    {
                        "learning rate": optimizer.param_groups[0]["lr"],
                        "validation Dice": val_score,
                        "images": wandb.Image(images[0].cpu().squeeze()),
                        "masks": {
                            "true": wandb.Image(true_masks[0].float().cpu()),
                            "pred": wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                        },
                        "validation loss": val_loss,
                        "step": global_step,
                        "epoch": epoch,
                        "epoch avg loss": epoch_loss,
                    }
                )

                pbar.set_postfix(
                    **{
                        "Loss (batch)": loss.item(),
                        "Validation Dice score": val_score,
                        "Learning rate": optimizer.param_groups[0]["lr"],
                    }
                )

                # Save a checkpoint for every epoch
                Path(self.training_directory).mkdir(parents=True, exist_ok=True)
                torch.save(self.net.state_dict(), self.training_directory + "/checkpoint_epoch{}.pth".format(epoch + 1))
                logging.info(f"Checkpoint {epoch + 1} saved!")

                # Save the best
                if val_score == max(val_scores):
                    Path(self.model_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(self.net.state_dict(), self.training_directory + "/unet_best.pth")
                    logging.info(f"Best saved at epch {epoch + 1}!")

        # Close the experiment
        experiment.finish()

    def _val_loss(self, dataloader:DataLoader, criterion) -> float:
        """Calculate the loss over the validation dataset

        Args:
            dataloader (Dataloader): Dataloader for the validation dataset
            criterion (Criterion): Loss function

        Returns:
            float: Loss over the validation dataset
        """
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:

                images = batch["image"].to(self.device, dtype=torch.float32)
                true_masks = torch.squeeze(batch["mask"]).to(self.device, dtype=torch.long)

                masks_pred_val = self.net(images)

                dice = dice_loss(
                    F.softmax(masks_pred_val, dim=1).float(),
                    F.one_hot(true_masks, self.net.n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True,
                )

                loss = criterion(masks_pred_val, true_masks) + dice
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on the validation dataset

        Args:
            dataloader (DataLoader): Dataloader for the validation dataset

        Returns:
            float: Dice score
        """

        self.net.eval()
        num_val_batches = len(dataloader)
        dice_score = 0

        # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image, mask_true = batch["image"], batch["mask"]
            # move images and labels to correct device and type
            image = image.to(device=self.device, dtype=torch.float32)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, self.net.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                mask_pred = self.net(image)

                # convert to one-hot format
                if self.net.n_classes == 1:
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), self.net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(
                        mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False
                    )

        self.net.train()

        if num_val_batches == 0:
            return dice_score
        return dice_score / num_val_batches

    def _predict_img(self, full_img:np.ndarray) -> np.ndarray:
        """Predict the mask for a single image

        Args:
            full_img (np.ndarray): Image to predict the mask for

        Returns:
            np.ndarray: Predicted mask
        """

        self.net.eval()
        full_img = full_img.reshape(1, 1, full_img.shape[0], full_img.shape[1])
        img = torch.from_numpy(full_img)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.net(img)
            if self.net.n_classes > 1:
                probs = F.softmax(output, dim=1)[0]
            else:
                probs = torch.sigmoid(output)[0]

        probs = probs.cpu()
        to_return = probs if self.net.n_classes == 1 else probs[1]
        return to_return.numpy()


    def predict(self, images: np.ndarray) -> np.ndarray:
        """Predict the mask for a batch of images

        Args:
            images (np.ndarray): Batch of images to predict the mask for

        Returns:
            np.ndarray: Predicted masks
        """

        if not self.net:
            self.load()

        desc = 'Running inference'
        masks = [
            self._predict_img(full_img=image)
            for image in tqdm(
                images, total=len(images), desc=desc, ncols=100,
                leave=False
            )
        ]

        return np.asarray(masks)

    def load(self, epoch: int=None) -> None:
        """Load the model

        Args:
            epoch (int, optional): Epoch to load. Defaults to None.
        """
        if not self.net:
            self.initialize_network()
        if not epoch:
            model_to_load = self.best_model_location
        else:
            model_to_load = os.path.join(
                self.model_dir, f'/checkpoint_epoch{epoch}.pth'
            )

        self.net.load_state_dict(
            torch.load(model_to_load, map_location=self.device)
        )
