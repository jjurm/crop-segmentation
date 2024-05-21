from collections import OrderedDict

import wandb
import yaml


class LabelEncoder:
    def __init__(self, label_encoder_artifact: str):
        artifact = wandb.run.use_artifact(label_encoder_artifact, type="label_encoder")
        label_encoder_filename = artifact.file()
        with open(label_encoder_filename, 'r') as file:
            encoder_list = yaml.safe_load(file)["labels"]

        for i in range(len(encoder_list)):
            if not isinstance(encoder_list[i]["id"], list):
                encoder_list[i]["id"] = [encoder_list[i]["id"]]

        # Mapping from model label to dataset label
        self.entries_with_name: list[tuple[int, list[int], str]] = [
            (i, mapping["id"], mapping["name"])
            for i, mapping in enumerate(encoder_list)
        ]
        self.entries: list[tuple[int, list[int]]] = [
            (i, mapping["id"])
            for i, mapping in enumerate(encoder_list)
        ]

        # Mapping from dataset label to model label
        self.dataset_to_model: dict[int, int] = OrderedDict([
            (dataset_label, i)
            for i, dataset_labels in self.entries
            for dataset_label in dataset_labels
        ])
        self.dataset_to_model_with_name: dict[int, tuple[int, str]] = OrderedDict([
            (dataset_label, (i, name))
            for i, dataset_labels, name in self.entries_with_name
            for dataset_label in dataset_labels
        ])

        self.class_names: list[str] = [mapping["name"] for mapping in encoder_list]

    @property
    def num_classes(self):
        return len(self.entries)
