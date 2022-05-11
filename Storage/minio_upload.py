from minio import Minio
import os
from argparse import ArgumentParser


def upload_artifacts_to_minio(
    client: Minio,
    source: str,
    destination: str,
    bucket_name: str,
    output_dict: dict,
):
    import urllib3

    """Uploads artifacts to minio server.

    Args:
        client : Result client
        source : source path of artifacts.
        destination : destination path of artifacts
        bucket_name : minio bucket name.
        output_dict : dict of output containing destination paths,
                      source and bucket names
    Raises:
        Exception : on MaxRetryError, NewConnectionError,
                    ConnectionError.
    Returns:
        output_dict : dict of output containing destination paths,
                      source and bucket names
    """
    print(f"source {source} destination {destination}")
    try:
        client.fput_object(
            bucket_name=bucket_name,
            file_path=source,
            object_name=destination,
        )
        output_dict[destination] = {
            "bucket_name": bucket_name,
            "source": source,
        }
    except (
        urllib3.exceptions.MaxRetryError,
        urllib3.exceptions.NewConnectionError,
        urllib3.exceptions.ConnectionError,
        RuntimeError,
    ) as expection_raised:
        print(str(expection_raised))
        raise Exception(expection_raised)  # pylint: disable=raise-missing-from

    return output_dict


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--bucket_name",
        type=str,
        default="mlpipeline",
        help="Minio bucket name",
    )

    parser.add_argument(
        "--folder_name",
        type=str,
        default="test",
        help="Path to destination folder",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="../results",
        help="Input path of the file or folder to upload",
    )

    parser.add_argument(
        "--filename",
        type=str,
        help="Input path of the file or folder to upload",
    )

    args = parser.parse_args()

    bucket_name = args.bucket_name
    input_path = args.input_path
    folder_name = args.folder_name
    filename = args.filename

    if filename:
        input_path = os.path.join(input_path, filename)

    endpoint = os.environ["MINIO_ENDPOINT"]
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    output_dict = {}
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )

    if os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):  # pylint: disable=unused-variable
            for file in files:
                source = os.path.join(root, file)
                artifact_name = "/".join(source.split("/")[5:])
                destination = os.path.join(folder_name, artifact_name)
                upload_artifacts_to_minio(
                    client=client,
                    source=source,
                    destination=destination,
                    bucket_name=bucket_name,
                    output_dict=output_dict,
                )
    else:
        artifact_name = input_path.split("/")[-1]
        destination = os.path.join(folder_name, artifact_name)
        upload_artifacts_to_minio(
            client=client,
            source=input_path,
            destination=destination,
            bucket_name=bucket_name,
            output_dict=output_dict,
        )
