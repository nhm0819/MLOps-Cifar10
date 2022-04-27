# Weights & Biases
# import wandb

# Pytorch modules
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Pytorch-Lightning
from pytorch_lightning import LightningModule

import torchmetrics
import timm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class Classifier(LightningModule):
    def __init__(self, **kwargs):
        """method used to define our model parameters"""
        super().__init__()

        self.args = kwargs
        Path("../data/models/plots").mkdir(parents=True, exist_ok=True)

        self.model = timm.create_model(
            self.args.get("model_name", "resnet50"),
            pretrained=True,
            num_classes=self.args.get("num_classes", 10),
        )
        # self.criterion = F.CrossEntropyLoss()

        self.width = self.args.get("width", 32)
        self.height = self.args.get("height", 32)

        # metrics
        self.accuracy = torchmetrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        # x = self.softmax(x)
        return x

    # convenient method to get the loss on a batch
    def loss(self, x, y):
        logits = self(x)  # this calls self.forward
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        x, y = batch
        if batch_idx == 0:
            self.reference_image = (x[0]).unsqueeze(
                0
            )  # pylint: disable=attribute-defined-outside-init
            # self.reference_image.resize((1,1,28,28))
            # print("\n\nREFERENCE IMAGE!!!")
            # print(self.reference_image.shape)
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        # Log training loss
        self.log("train_loss", loss)

        # Log metrics
        self.log("train_acc", self.accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.log("val_loss", loss)  # default on val/test is on_epoch only
        self.log("val_acc", self.accuracy(preds, y))

        return logits

    # def validation_epoch_end(self, validation_step_outputs):
    #     dummy_input = torch.zeros((3, self.height, self.width), device=self.device)
    #     model_filename = f"model_{str(self.global_step).zfill(5)}.pt"
    #     self.to_torchscript(model_filename, method="script", example_inputs=dummy_input)
    #     # wandb.save(model_filename)
    #
    #     flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
    #     # self.logger.experiment.log(
    #     #     {"valid_logits": wandb.Histogram(flattened_logits.to("cpu")),
    #     #      "global_step": self.global_step})

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self.loss(x, y)
        preds = torch.argmax(logits, 1)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.accuracy(preds, y), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = Adam(self.parameters(), lr=self.args.get("lr", 0.0001))
        scheduler = ReduceLROnPlateau(optimizer, min_lr=1e-7)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def makegrid(self, output, numrows):  # pylint: disable=no-self-use
        """Makes grids.

        Args:
             output : Tensor output
             numrows : num of rows.
        Returns:
             c_array : gird array
        """
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b_array = np.array([]).reshape(0, outer.shape[2])
        c_array = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b_array = np.concatenate((img, b_array), axis=0)
            j += 1
            if j == numrows:
                c_array = np.concatenate((c_array, b_array), axis=1)
                b_array = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c_array

    def show_activations(self, x_var):
        """Showns activation
        Args:
             x_var: x variable
        """
        plt.imsave(
            f"../data/models/plots/input_{self.current_epoch}_epoch.png",
            torch.Tensor.cpu(x_var[0][0]),
        )

        # logging layer 1 activations
        out = self.model.conv1(x_var)
        c_grid = self.makegrid(out, 4)
        self.logger.experiment.add_image(
            "layer 1", c_grid, self.current_epoch, dataformats="HW"
        )

        plt.imsave(
            f"../data/models/plots/activation_{self.current_epoch}_epoch.png", c_grid
        )

    def training_epoch_end(self, outputs):
        """Training epoch end.

        Args:
             outputs: outputs of train end
        """
        self.show_activations(self.reference_image)

    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros((3, self.height, self.width), device=self.device)
    #     model_filename = "model_final.pt"
    #     # self.to_onnx(model_filename, dummy_input, export_params=True)
    #     self.to_torchscript(model_filename, method="script", example_inputs=dummy_input)
    #     # wandb.save(model_filename)
    #
    #     # flattened_logits = torch.flatten(torch.cat(test_step_outputs))
    #     # self.logger.experiment.log(
    #     #     {"test_logits": wandb.Histogram(flattened_logits.to("cpu")),
    #     #      "global_step": self.global_step})
