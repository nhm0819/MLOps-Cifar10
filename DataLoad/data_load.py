"""Cifar10 pre-process module."""
import subprocess
from pathlib import Path
from argparse import ArgumentParser
import torchvision
import webdataset as wds
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_path", default="../data", type=str)

    parser.add_argument(
        "--mlpipeline_ui_metadata",
        type=str,
        default="./",
        help="Path to write mlpipeline-ui-metadata.json",
    )

    args = vars(parser.parse_args())
    output_path = args["output_path"]

    Path(output_path).mkdir(parents=True, exist_ok=True)

    trainset = torchvision.datasets.CIFAR10(root=output_path, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=output_path, train=False, download=True)

    Path(output_path + "/train").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/val").mkdir(parents=True, exist_ok=True)
    Path(output_path + "/test").mkdir(parents=True, exist_ok=True)

    RANDOM_SEED = 42
    y = trainset.targets
    trainset, valset, y_train, y_val = train_test_split(
        trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=RANDOM_SEED
    )

    for name in [(trainset, "train"), (valset, "val"), (testset, "test")]:
        with wds.ShardWriter(
            output_path + "/" + str(name[1]) + "/" + str(name[1]) + "-%d.tar",
            maxcount=1000,
        ) as sink:
            for index, (image, cls) in enumerate(name[0]):
                sink.write({"__key__": "%06d" % index, "ppm": image, "cls": cls})

    entry_point = ["ls", "-R", output_path]
    run_code = subprocess.run(
        entry_point, stdout=subprocess.PIPE
    )  # pylint: disable=subprocess-run-check
    print(run_code.stdout)

    # y_array = np.array(y)
    #
    # label_names = [
    #     "airplane",
    #     "automobile",
    #     "bird",
    #     "cat",
    #     "deer",
    #     "dog",
    #     "frog",
    #     "horse",
    #     "ship",
    #     "truck",
    # ]
    # label_counts = dict(zip(*np.unique(y_array, return_counts=True)))
    # label_dict = {}
    # TOTAL_COUNT = len(y)
    # for key, value in label_counts.items():
    #     print(
    #         "Label Counts of [{}]({}) : {}".format(key, label_names[key].upper(), value)
    #     )
    #     label_dict[label_names[key].upper()] = int(value)
    #
    # label_dict["TOTAL_COUNT"] = int(TOTAL_COUNT)
    #
    # markdown_dict = {"storage": "inline", "source": label_dict}
