from functools import partial
from pathlib import Path
from typing import Generator


import io

import librosa
import lightning as L
import litdata as ld  # type: ignore
import torch
from diffusers.pipelines.deprecated.audio_diffusion.mel import Mel
from litdata import optimize
from torchvision.transforms import v2  # type: ignore
from datasets import load_dataset  # type: ignore
from pyarrow.parquet import ParquetFile

# From https://zenodo.org/records/1432913
INSTRUMENTS = {
    "accordion": 0,
    "banjo": 1,
    "bass": 2,
    "cello": 3,
    "clarinet": 4,
    "cymbals": 5,
    "drums": 6,
    "flute": 7,
    "guitar": 8,
    "mallet_percussion": 9,
    "mandolin": 10,
    "organ": 11,
    "piano": 12,
    "saxophone": 13,
    "synthesizer": 14,
    "trombone": 15,
    "trumpet": 16,
    "ukulele": 17,
    "violin": 18,
    "voice": 19,
}


def _transform_parquet(
    parquet_path: str,
    mel: Mel,
    transform_slice: v2.Compose,
) -> Generator[dict[str, torch.Tensor], None, None]:
    """Standalone generator so litdata workers only pickle mel/transform_slice,
    not the entire LightningDataModule (which carries a trainer/wandb reference)."""
    parquet_file = ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=32):
        df = batch.to_pandas()
        df["mel_img"] = df["mp3_bytes"].apply(
            lambda x: (
                mel.load_audio(raw_audio=librosa.load(io.BytesIO(x), mono=True, sr=mel.sr)[0])
                or mel.audio_slice_to_image(0)
            )
        )
        for row in df.itertuples():
            yield {"mel": transform_slice(row.mel_img), "label": row.true, "mask": row.mask}


class OpenMICDataModule(L.LightningDataModule):
    def __init__(
        self,
        index_path: str | None = None,
        batch_size: int = 32,
        num_workers: int = 4,
        num_workers_optimize: int = 1,
        base_data: Path = Path("data"),
    ):
        super().__init__()
        self.index_path = index_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_workers_optimize = num_workers_optimize
        self.base_data = base_data

        self.mel = Mel(
            sample_rate=22050,  # resampled to match audio-diffusion-breaks-256
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

    def optimize(self):
        if (
            not (self.base_data / "openmic_out_train").exists()
            and not (self.base_data / "openmic_out_test").exists()
        ):
            load_dataset("CPJKU/openmic", split="train").to_parquet(
                self.base_data / "openmic_train.parquet"
            )
            load_dataset("CPJKU/openmic", split="test").to_parquet(
                self.base_data / "openmic_test.parquet"
            )

            fn = partial(_transform_parquet, mel=self.mel, transform_slice=self.transform_slice)

            optimize(
                fn=fn,
                inputs=[str(self.base_data / "openmic_train.parquet")],
                output_dir=str(self.base_data / "openmic_out_train"),
                chunk_bytes="64MB",
                num_workers=self.num_workers_optimize,
                start_method="spawn",
            )
            optimize(
                fn=fn,
                inputs=[str(self.base_data / "openmic_test.parquet")],
                output_dir=str(self.base_data / "openmic_out_test"),
                chunk_bytes="64MB",
                num_workers=self.num_workers_optimize,
                start_method="spawn",
            )

    def prepare_data(self) -> None:
        # download
        self.optimize()
        self.ds_train = ld.StreamingDataset(
            str(self.base_data / "openmic_out_train"),
            index_path=self.index_path,
        )
        self.ds_test = ld.StreamingDataset(
            str(self.base_data / "openmic_out_test"), index_path=self.index_path
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

    def val_dataloader(self):
        assert self.ds_test is not None, (
            "Dataset not prepared. Call prepare_data() first."
        )

        return ld.StreamingDataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
