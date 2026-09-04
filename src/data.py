import json

from torch.utils.data import Dataset


class ColBERTTripletDataset(Dataset):

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @classmethod
    def from_jsonl(cls, path):

        examples = []

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                examples.append(
                    json.loads(line)
                )

        return cls(examples)