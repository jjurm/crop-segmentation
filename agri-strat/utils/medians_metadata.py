from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin


@dataclass
class MediansMetadata(DataClassJsonMixin):
    bands: list[str]
    img_size: list[int]
    num_patches: int
    num_subpatches_per_patch: int = None
    class_pixel_counts: dict[int, int] = field(default_factory=dict)
