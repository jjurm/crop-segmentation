from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

FILENAME = 'meta.json'


@dataclass
class MediansMetadataPerSplit(DataClassJsonMixin):
    size: int


@dataclass
class MediansMetadata(DataClassJsonMixin):
    bands: list[str]
    img_size: list[int]
    num_subpatches: int = None
    train: MediansMetadataPerSplit = None
    val: MediansMetadataPerSplit = None
    test: MediansMetadataPerSplit = None

    def get_split(self, split: str) -> MediansMetadataPerSplit:
        if split == 'train':
            return self.train
        elif split == 'val':
            return self.val
        elif split == 'test':
            return self.test
        else:
            raise ValueError(f"Unknown split: {split}")
