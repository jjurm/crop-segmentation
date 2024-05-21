from lightning.pytorch.callbacks import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items
