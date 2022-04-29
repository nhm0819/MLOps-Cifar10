from kfp import components
from kfp import dsl
from kfp import compiler
import kubernetes as k8s

data_op = components.load_component_from_file("Components/data_component.yaml")
train_op = components.load_component_from_file("Components/train_component.yaml")


# training args
MODEL = "resnet50"
GPUS = "-1"
MAX_EPOCHS = "30"
NUM_CLASSES = "10"
TRAIN_BATCH_SIZE = "256"
TRAIN_NUM_WORKERS = "8"
VAL_BATCH_SIZE = "256"
VAL_NUM_WORKERS = "8"
LR = "0.001"


@dsl.pipeline(
    name="Pytorch Lightning Training pipeline", description="Cifar 10 dataset pipeline"
)
def pl_pipeline():

    data_task = data_op().set_display_name("Data Preprocess")

    shm_volume = dsl.PipelineVolume(
        volume=k8s.client.V1Volume(
            name="shm", empty_dir=k8s.client.V1EmptyDirVolumeSource(medium="Memory")
        )
    )
    train_task = (
        train_op(
            dataset_path=data_task.outputs["output_data"],
            model=MODEL,
            gpus=GPUS,
            max_epochs=MAX_EPOCHS,
            num_classes=NUM_CLASSES,
            train_batch_size=TRAIN_BATCH_SIZE,
            train_num_workers=TRAIN_NUM_WORKERS,
            val_batch_size=VAL_BATCH_SIZE,
            val_num_workers=VAL_NUM_WORKERS,
            lr=LR,
        )
        .after(data_task)
        .set_display_name("Training")
        .add_pvolumes({"/dev/shm": shm_volume})
        # .apply(onprem.mount_pvc(pvc_name="shm-volume-claim", volume_name='shm-volume', volume_mount_path='/dev/shm'))
    ).set_gpu_limit(1)


if __name__ == "__main__":
    compiler.Compiler().compile(pl_pipeline, package_path="pytorch_lightning.yaml")
