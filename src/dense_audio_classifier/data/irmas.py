# coding=utf-8

"""IRMAS dataset."""

import os
import re
import textwrap
from typing import Any, Iterator, cast
import datasets  # type: ignore
from datasets.builder import Key  # type: ignore

SAMPLE_RATE = 44_100

_IRMAS_TRAIN_SET_URL = "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip"
_IRMAS_TEST_SET_PART1_URL = (
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip"
)
_IRMAS_TEST_SET_PART2_URL = (
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip"
)
_IRMAS_TEST_SET_PART3_URL = (
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip"
)


INSTRUMENTS = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]


class IRMASConfig(datasets.BuilderConfig):
    """BuilderConfig for IRMAS."""

    def __init__(self, features, **kwargs):
        super(IRMASConfig, self).__init__(
            version=datasets.Version("0.0.1", ""), **kwargs
        )
        self.features = features


class IRMAS(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        IRMASConfig(
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=SAMPLE_RATE),
                    "instrument": datasets.Sequence(datasets.Value("string")),
                    "label": datasets.Sequence(datasets.ClassLabel(names=INSTRUMENTS)),
                }
            ),
            name="irmas",
            description=textwrap.dedent(
                """\
                IRMAS is intended to be used for training and testing methods for the automatic recognition of predominant instruments in musical audio. 
                The instruments considered are: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and human singing voice.
                """
            ),
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=self.BUILDER_CONFIGS[0].features,
            supervised_keys=None,
            homepage="https://zenodo.org/records/1290750",
            citation="""
                @inproceedings{bosch2012comparison,
                  title={A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals.},
                  author={Bosch, Juan J and Janer, Jordi and Fuhrmann, Ferdinand and Herrera, Perfecto},
                  booktitle={ISMIR},
                  pages={559--564},
                  year={2012}
                }
            """,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        (
            train_archive_path,
            test_archive_part1_path,
            test_archive_part2_path,
            test_archive_part3_path,
        ) = cast(
            tuple[str, str, str, str],
            dl_manager.download_and_extract(
                [
                    _IRMAS_TRAIN_SET_URL,
                    _IRMAS_TEST_SET_PART1_URL,
                    _IRMAS_TEST_SET_PART2_URL,
                    _IRMAS_TEST_SET_PART3_URL,
                ]
            ),
        )

        # train_archive_path = cast(
        #     str, dl_manager.download_and_extract(_IRMAS_TRAIN_SET_URL)
        # )
        # test_archive_part1_path = cast(
        #     str, dl_manager.download_and_extract(_IRMAS_TEST_SET_PART1_URL)
        # )
        # test_archive_part2_path = cast(
        #     str, dl_manager.download_and_extract(_IRMAS_TEST_SET_PART2_URL)
        # )
        # test_archive_part3_path = cast(
        #     str, dl_manager.download_and_extract(_IRMAS_TEST_SET_PART3_URL)
        # )

        extensions = [".wav"]
        _, _train_walker = fast_scandir(train_archive_path, extensions, recursive=True)
        _test_walker = []
        for part in [
            test_archive_part1_path,
            test_archive_part2_path,
            test_archive_part3_path,
        ]:
            _, _walker = fast_scandir(part, extensions, recursive=True)
            _test_walker.extend(_walker)

        return [
            datasets.SplitGenerator(
                name=str(datasets.Split.TRAIN),
                gen_kwargs={"audio_filepaths": _train_walker, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=str(datasets.Split.TEST),
                gen_kwargs={"audio_filepaths": _test_walker, "split": "test"},
            ),
        ]

    def _generate_examples(self, **kwargs) -> Iterator[tuple[Key, dict[str, Any]]]:
        audio_filepaths = kwargs["audio_filepaths"]

        def extract_bracketed_items(filename):
            # Regex pattern to find text inside square brackets
            pattern = r"\[([^\]]+)\]"
            # Find all occurrences of the pattern
            items = re.findall(pattern, filename)
            return items

        def deduplicate(lst):
            return list(dict.fromkeys(lst))

        split = kwargs.get("split")
        if split == "train":
            for guid, audio_path in enumerate(audio_filepaths):
                labels = extract_bracketed_items(audio_path)
                labels = deduplicate(labels)
                labels = [label for label in labels if label in INSTRUMENTS]
                yield (
                    Key(0, guid),
                    {
                        "id": str(guid),
                        "file": audio_path,
                        "audio": audio_path,
                        "instrument": labels,
                        "label": labels,
                    },
                )

        elif split == "test":
            for guid, audio_path in enumerate(audio_filepaths):
                labels = []
                with open(
                    audio_path.replace(".wav", ".txt"), "r", encoding="utf-8"
                ) as f:
                    for line in f:
                        labels.append(line.strip())
                labels = deduplicate(labels)
                yield (
                    Key(0, guid),
                    {
                        "id": str(guid),
                        "file": audio_path,
                        "audio": audio_path,
                        "instrument": labels,
                        "label": labels,
                    },
                )


def fast_scandir(path: str, exts: list[str], recursive: bool = False):
    # Scan files recursively faster than glob
    # From github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    subfolders: list[str] = []
    files: list[str] = []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files
