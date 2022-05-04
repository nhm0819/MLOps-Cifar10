import kfp
from kfp import components
from kfp import dsl
from kfp import compiler
import kubernetes as k8s
from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
import os


# load components
data_op = components.load_component_from_file("Components/data_component.yaml")
train_op = components.load_component_from_file("Components/train_component.yaml")


# pipeline args
INGRESS_GATEWAY = "http://istio-ingressgateway.istio-system.svc.cluster.local"
AUTH = "MTY1MDQzODUwM3xOd3dBTkVkVVFVeENXVlEzUjFNMFZsRlNRa1JMV1VrelExZFpXbGhPTjBaSlRqUlpRamRMVWtGRFVVd3lVMDVJVVZKTFVsQTFOVkU9fISevwge92e4Q9DScFYLSmPU39-BbeWSrp1dfAh3TTpi"
NAMESPACE = "kubeflow-user-example-com"
COOKIE = "authservice_session=" + AUTH
EXPERIMENT = "Default"

SECRET_NAME = "mysecret"
# S3_ENDPOINT = 'minio-service.kubeflow.svc.cluster.local:9000'
S3_ENDPOINT = "minio-service.kubeflow:9000"
MINIO_ENDPOINT = "http://" + S3_ENDPOINT
MINIO_REGION = "us-east-1"
LOG_BUCKET = "mlpipeline"

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
            result_path=f"s3://{LOG_BUCKET}",
        )
        .after(data_task)
        .set_display_name("Training")
        .add_pvolumes({"/dev/shm": shm_volume})
        .add_env_variable(
            V1EnvVar(name="WANDB_API_KEY", value=os.environ["WANDB_API_KEY"])
        )
        .add_env_variable(V1EnvVar(name="S3_ENDPOINT", value=S3_ENDPOINT))
        .add_env_variable(V1EnvVar(name="AWS_ENDPOINT_URL", value=MINIO_ENDPOINT))
        .add_env_variable(
            V1EnvVar(
                name="AWS_ACCESS_KEY_ID",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="AWS_ACCESS_KEY_ID"
                    )
                ),
            )
        )
        .add_env_variable(
            V1EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                value_from=V1EnvVarSource(
                    secret_key_ref=V1SecretKeySelector(
                        name=SECRET_NAME, key="AWS_SECRET_ACCESS_KEY"
                    )
                ),
            )
        )
        .add_env_variable(V1EnvVar(name="AWS_REGION", value=MINIO_REGION))
        .add_env_variable(V1EnvVar(name="S3_USE_HTTPS", value="0"))
        .add_env_variable(V1EnvVar(name="S3_VERIFY_SSL", value="0"))
    ).set_gpu_limit(1)
    train_task.execution_options.caching_strategy.max_cache_staleness = "P0D"


if __name__ == "__main__":
    compiler.Compiler().compile(pl_pipeline, package_path="pytorch_lightning.yaml")
