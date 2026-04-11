from pathlib import Path

import lightning as L
import litdata as ld  # type: ignore
import torch
from diffusers.pipelines.deprecated.audio_diffusion.mel import Mel
from litdata import optimize
from torchvision.transforms import v2  # type: ignore

from dense_audio_classifier.data.irmas import INSTRUMENTS, IRMAS

instrument_to_idx = {inst: i for i, inst in enumerate(INSTRUMENTS)}


class IRMASDataModule(L.LightningDataModule):
    def __init__(
        self,
        index_path: str | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        data_optimize_output_dir: Path = Path("data"),
    ):
        super().__init__()
        self.index_path = index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_optimize_output_dir = data_optimize_output_dir
        self.dataset_uri = "hf://datasets/confit/irmas/irmas.py"
        self.mel = Mel(
            sample_rate=44100,  # see https://www.upf.edu/web/mtg/irmas
            n_fft=2048,
            x_res=256,
            y_res=256,
            hop_length=512,
        )
        self.transform_slice = v2.Compose(
            [
                v2.PILToTensor(),  # Convert tensor to PIL Image
                v2.ConvertImageDtype(torch.float32),  # cleaner than lambda
                v2.Normalize(mean=[0.5], std=[0.5]),  # maps [0,1] → [-1,1]
            ]
        )
        self.ds_train: ld.StreamingDataset | None = None
        self.ds_test: ld.StreamingDataset | None = None

    def transform(self, audio):
        self.mel.load_audio(audio["file"])
        mel_image = self.mel.audio_slice_to_image(
            0
        )  # Get the first slice as an example
        # Convert the PIL Image to a tensor and normalize
        mel_tensor = self.transform_slice(mel_image)
        # tensor multi label, e.g. [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0] for "cel" and "gel"
        label = torch.zeros(len(INSTRUMENTS), dtype=torch.int)
        label[[instrument_to_idx[i] for i in audio["instrument"]]] = 1
        return {"mel": mel_tensor, "label": label}

    def optimize(self):
        if (
            not (self.data_optimize_output_dir / "_out_train").exists()
            and not (self.data_optimize_output_dir / "_out_test").exists()
        ):
            builder = IRMAS()
            builder.download_and_prepare()
            train_ds = builder.as_dataset("train").to_list()  # type: ignore
            test_ds = builder.as_dataset("test").to_list()  # type: ignore
            optimize(
                fn=self.transform,
                inputs=train_ds,
                output_dir=str(self.data_optimize_output_dir / "_out_train"),
                chunk_bytes="64MB",
                num_workers=self.num_workers,
                start_method="spawn",
            )
            optimize(
                fn=self.transform,
                inputs=test_ds,
                output_dir=str(self.data_optimize_output_dir / "_out_test"),
                chunk_bytes="64MB",
                num_workers=self.num_workers,
                start_method="spawn",
            )

    def prepare_data(self) -> None:
        # download
        self.optimize()
        self.ds_train = ld.StreamingDataset(
            str(self.data_optimize_output_dir / "_out_train"),
            index_path=self.index_path,
        )
        self.ds_test = ld.StreamingDataset(
            str(self.data_optimize_output_dir / "_out_test"), index_path=self.index_path
        )

    def train_dataloader(self):
        assert self.ds_train is not None, (
            "Dataset not prepared. Call prepare_data() first."
        )
        return ld.StreamingDataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        assert self.ds_test is not None, (
            "Dataset not prepared. Call prepare_data() first."
        )

        return ld.StreamingDataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
