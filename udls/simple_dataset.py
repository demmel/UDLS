import torch
from udls import SimpleLMDBDataset
from pathlib import Path
import librosa as li
from os import makedirs, path
from tqdm import tqdm
import numpy as np
import random


def dummy_load(name):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = li.load(name, 16000)[0]
    if len(x) % 2048:
        x = x[:-(len(x) % 2048)]
    x = x.reshape(-1, 2048)
    if x.shape[0]:
        return x
    else:
        return None


def simple_audio_preprocess(sampling_rate, N):
    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        pad = (N - (len(x) % N)) % N
        x = np.pad(x, (0, pad))

        x = x.reshape(-1, N)
        return x.astype(np.float32)

    return preprocess


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        out_database_location,
        folder_list=None,
        file_list=None,
        preprocess_function=dummy_load,
        transforms=None,
        extension="*.wav,*.aif",
        map_size=1e11,
        split_percent=.2,
        split_set="train",
        seed=0,
        label_suffix=None,
    ):
        super().__init__()

        assert folder_list is not None or file_list is not None
        makedirs(out_database_location, exist_ok=True)

        self.env = SimpleLMDBDataset(out_database_location, map_size)

        self.folder_list = folder_list
        self.file_list = file_list
        self.label_suffix = label_suffix

        self.preprocess_function = preprocess_function
        self.extension = extension

        self.transforms = transforms

        # IF NO DATA INSIDE DATASET: PREPROCESS
        self.len = len(self.env)

        if self.len == 0:
            self._preprocess()
            self.len = len(self.env)

        if self.len == 0:
            raise Exception("No data found !")

        self.index = np.arange(self.len)
        np.random.seed(seed)
        np.random.shuffle(self.index)

        if split_set == "train":
            self.len = int(np.floor((1 - split_percent) * self.len))
            self.offset = 0

        elif split_set == "test":
            self.offset = int(np.floor((1 - split_percent) * self.len))
            self.len = self.len - self.offset

        elif split_set == "full":
            self.offset = 0

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []

        def input_name_with_label(file):
            path = Path(file)
            if self.label_suffix is not None:
                return (path, path.with_name(f'{path.name}{self.label_suffix}'))
            else:
                return (path, path)

        def input_names_with_labels(file_list):
            return [input_name_with_label(file) for file in file_list]

        # POPULATE WAV LIST
        if self.folder_list is not None:
            for folder in self.folder_list.split(","):
                print("Recursive search in {}".format(folder))
                for ext in extension:
                    files = list(Path(folder).rglob(ext))
                    files = filter(
                        lambda file: not file.name.endswith(self.label_suffix),
                        files
                    ) if self.label_suffix is not None else files
                    wavs.extend(
                        input_names_with_labels(files)
                    )

        else:
            with open(self.file_list, "r") as file_list:
                wavs = input_names_with_labels(file_list.read().split("\n"))

        loader = tqdm(wavs)
        for (input, label) in loader:
            loader.set_description("{}".format(path.basename(input)))
            input_preprocessed = self.preprocess_function(input)
            label_preprocessed = input_preprocessed if input == label else self.preprocess_function(
                label)

            if input_preprocessed is None or label_preprocessed is None:
                print(f"No data for {input}")
            elif len(input_preprocessed) != len(label_preprocessed):
                print(
                    f"Input length '{len(input_preprocessed)}' does not match label length '{len(label_preprocessed)}' for {input}"
                )

            for o in zip(input_preprocessed, label_preprocessed):
                self.env[idx] = o
                idx += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        (input, label) = self.env[self.index[index + self.offset]]

        if self.transforms is not None:
            rng_state = random.getstate()
            np_rng_state = np.random.get_state()
            input = self.transforms(input)

            random.setstate(rng_state)
            np.random.set_state(np_rng_state)
            label = self.transforms(label)

        return (input, label)
