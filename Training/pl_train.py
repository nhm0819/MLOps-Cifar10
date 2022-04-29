"""kubeflow pytorch-lightning training script"""
from pathlib import Path
from argparse import ArgumentParser
from Training.pl_model import Classifier
from Training.pl_datamodule import CIFAR10DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Argument parser for user defined paths
parser = ArgumentParser()

# Train hyperparams
parser.add_argument(
    "--model", default="resnet50", type=str, help="model structure name in timm package"
)
parser.add_argument("--gpus", default=-1, type=int, help="num of gpus")
parser.add_argument("--max_epochs", default=30, type=int, help="training max epochs")
parser.add_argument("--num_classes", default=10, type=int, help="num_classes")
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=256,
    metavar="N",
    help="batch size / num_gpus",
)
parser.add_argument(
    "--train_num_workers", type=int, default=10, metavar="N", help="number of workers"
)
parser.add_argument(
    "--val_batch_size", type=int, default=256, metavar="N", help="batch size / num_gpus"
)
parser.add_argument(
    "--val_num_workers", type=int, default=10, metavar="N", help="number of workers"
)
parser.add_argument("--lr", type=float, default=1e-3, metavar="N", help="learning rate")

# log args
parser.add_argument(
    "--check_val_every_n_epoch",
    type=int,
    default=10,
    metavar="N",
    help="checkpoint period",
)

parser.add_argument(
    "--log_every_n_steps",
    default=50,
    type=int,
    help="log every n steps",
)

# container IO
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="/train/models",
    help="Path to save model checkpoints (default: output/train/models)",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="../data",
    help="Cifar10 Dataset path (default: ../data)",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet.pth",
    help="Name of the model to be saved as (default: resnet.pth)",
)

parser.add_argument(
    "--mlpipeline_ui_metadata",
    default="mlpipeline-ui-metadata.json",
    type=str,
    help="Path to write mlpipeline-ui-metadata.json",
)

parser.add_argument(
    "--mlpipeline_metrics",
    default="mlpipeline-metrics.json",
    type=str,
    help="Path to write mlpipeline-metrics.json",
)

parser.add_argument("--trial_id", default=0, type=int, help="Trial id")
parser.add_argument(
    "--model_params", default=None, type=str, help="Model parameters for trainer"
)
parser.add_argument(
    "--results", default="results.json", type=str, help="Training results"
)


if __name__ == "__main__":
    # parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()

    # Enabling Tensorboard Logger, ModelCheckpoint, Earlystopping

    lr_logger = LearningRateMonitor()
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="cifar10_{epoch:02d}_{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Creating parent directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Setting the datamodule specific arguments
    datamodule_args = {
        "dataset_path": args.dataset_path,
        "train_batch_size": args.train_batch_size,
        "train_num_workers": args.train_num_workers,
        "val_batch_size": args.val_batch_size,
        "val_num_workers": args.val_num_workers,
    }

    datamodule = CIFAR10DataModule(**datamodule_args)

    # Initiating the training process
    trainer = Trainer(
        # logger=wandb_logger,    # W&B integration
        log_every_n_steps=args.log_every_n_steps,  # set the logging frequency
        gpus=args.gpus,  # use all GPUs
        max_epochs=args.max_epochs,  # number of epochs
        deterministic=True,  # keep it deterministic
        enable_checkpointing=True,
        callbacks=[
            # Logger(samples),
            early_stopping,
            checkpoint_callback,
        ],  # see Callbacks section
        precision=16,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        strategy="ddp",
        auto_scale_batch_size=True,
    )

    model_config = {
        "model_name": args.model,
        "num_classes": args.num_classes,
        "lr": args.lr,
        "width": 32,
        "height": 32,
    }
    # model = trainer.lightning_module
    model = Classifier(**model_config)

    trainer.fit(model, datamodule)
    trainer.test(datamodule=datamodule)

    # if trainer.global_rank == 0:
    #     # Mar file generation
    #
    #     cifar_dir, _ = os.path.split(os.path.abspath(__file__))
    #
    #     mar_config = {
    #         "MODEL_NAME": "cifar10_test",
    #         "MODEL_FILE": os.path.join(cifar_dir, "cifar10_train.py"),
    #         "HANDLER": os.path.join(cifar_dir, "cifar10_handler.py"),
    #         "SERIALIZED_FILE": os.path.join(CHECKPOINT_DIR, script_dict["model_name"]),
    #         "VERSION": "1",
    #         "EXPORT_PATH": CHECKPOINT_DIR,
    #         "CONFIG_PROPERTIES": os.path.join(cifar_dir, "config.properties"),
    #         "EXTRA_FILES": "{},{}".format(
    #             os.path.join(cifar_dir, "class_mapping.json"),
    #             os.path.join(cifar_dir, "classifier.py"),
    #         ),
    #         "REQUIREMENTS_FILE": os.path.join(cifar_dir, "requirements.txt"),
    #     }
    #
    #     MarGeneration(mar_config=mar_config, mar_save_path=CHECKPOINT_DIR)
    #
    #     classes = [
    #         "airplane",
    #         "automobile",
    #         "bird",
    #         "cat",
    #         "deer",
    #         "dog",
    #         "frog",
    #         "horse",
    #         "ship",
    #         "truck",
    #     ]
    #
    #     # print(dir(trainer.ptl_trainer.model.module))
    #     # model = trainer.ptl_trainer.model
    #
    #     target_index_list = list(set(model.target))
    #
    #     class_list = []
    #     for index in target_index_list:
    #         class_list.append(classes[index])
    #
    #     confusion_matrix_dict = {
    #         "actuals": model.target,
    #         "preds": model.preds,
    #         "classes": class_list,
    #         "url": script_dict["confusion_matrix_url"],
    #     }
    #
    #     test_accuracy = round(float(model.test_acc.compute()), 2)
    #
    #     print("Model test accuracy: ", test_accuracy)
    #
    #     if "model_params" in args and args["model_params"] is not None:
    #         data = {}
    #         data[trial_id] = test_accuracy
    #
    #         Path(os.path.dirname(args["results"])).mkdir(parents=True, exist_ok=True)
    #
    #         results_file = Path(args["results"])
    #         if results_file.is_file():
    #             with open(results_file, "r") as fp:
    #                 old_data = json.loads(fp.read())
    #             data.update(old_data)
    #
    #         with open(results_file, "w") as fp:
    #             fp.write(json.dumps(data))
    #
    #     visualization_arguments = {
    #         "input": {
    #             "tensorboard_root": TENSORBOARD_ROOT,
    #             "checkpoint_dir": CHECKPOINT_DIR,
    #             "dataset_path": DATASET_PATH,
    #             "model_name": script_dict["model_name"],
    #             "confusion_matrix_url": script_dict["confusion_matrix_url"],
    #         },
    #         "output": {
    #             "mlpipeline_ui_metadata": args["mlpipeline_ui_metadata"],
    #             "mlpipeline_metrics": args["mlpipeline_metrics"],
    #         },
    #     }
    #
    #     markdown_dict = {"storage": "inline", "source": visualization_arguments}
    #
    #     print("Visualization Arguments: ", markdown_dict)
    #
    #     visualization = Visualization(
    #         test_accuracy=test_accuracy,
    #         confusion_matrix_dict=confusion_matrix_dict,
    #         mlpipeline_ui_metadata=args["mlpipeline_ui_metadata"],
    #         mlpipeline_metrics=args["mlpipeline_metrics"],
    #         markdown=markdown_dict,
    #     )
    #
    #     checpoint_dir_contents = os.listdir(CHECKPOINT_DIR)
    #     print(f"Checkpoint Directory Contents: {checpoint_dir_contents}")
