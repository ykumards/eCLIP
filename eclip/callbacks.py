from pytorch_lightning import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.total = self.total_steps
        return bar
