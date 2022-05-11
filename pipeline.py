from kfp import components
from kfp import dsl
from kfp import compiler
import kubernetes as k8s
from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
import os


# load components
data_op = components.load_component_from_file("Components/data_component.yaml")
train_op = components.load_component_from_file("Components/train_component.yaml")
minio_op = components.load_component_from_file("Components/minio_component.yaml")


# pipeline args
NAMESPACE = "kubeflow-user-example-com"
SECRET_NAME = "mlpipeline-minio-artifact"  # SECRET_NAME = "minio-s3-secret"
MINIO_ENDPOINT = "minio-service.kubeflow:9000"  # S3_ENDPOINT = 'minio-service.kubeflow.svc.cluster.local:9000'
# MINIO_ENDPOINT = "http://" + S3_ENDPOINT
# MINIO_REGION = "us-east-1"
BUCKET_NAME = "mlpipeline"
LOG_DIR = f"logs/{dsl.RUN_ID_PLACEHOLDER}"
CHECKPOINT_DIR = f"ckpt/{dsl.RUN_ID_PLACEHOLDER}"

# training args
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
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

    # Data Load
    data_task = data_op().set_display_name("Data Preprocess")
    data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # Training
    shm_volume = dsl.PipelineVolume(
        volume=k8s.client.V1Volume(
            name="shm", empty_dir=k8s.client.V1EmptyDirVolumeSource(medium="Memory")
        )
    )
    train_task = (
        train_op(
            dataset_path=data_task.outputs["preprocessed_path"],
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
        .add_env_variable(V1EnvVar(name="WANDB_API_KEY", value=WANDB_API_KEY))
    ).set_gpu_limit(1)

    train_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # Minio Upload
    (
        minio_op(
            bucket_name=BUCKET_NAME,
            folder_name=LOG_DIR,
            input_path=train_task.outputs["log_dir"],
            filename="",
        )
        .after(train_task)
        .set_display_name("training logs Pusher")
        .add_env_variable(V1EnvVar(name="MINIO_ENDPOINT", value=MINIO_ENDPOINT))
        .add_env_variable(
            V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="accesskey"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="secretkey"
                    )
                ),
            )
        )
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
    )

    (
        minio_op(
            bucket_name=BUCKET_NAME,
            folder_name=CHECKPOINT_DIR,
            input_path=train_task.outputs["checkpoint_dir"],
            filename="",
        )
        .after(train_task)
        .set_display_name("checkpoint results Pusher")
        .add_env_variable(V1EnvVar(name="MINIO_ENDPOINT", value=MINIO_ENDPOINT))
        .add_env_variable(
            V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="accesskey"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="secretkey"
                    )
                ),
            )
        )
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
    )


if __name__ == "__main__":
    compiler.Compiler().compile(pl_pipeline, package_path="pytorch_lightning.yaml")
