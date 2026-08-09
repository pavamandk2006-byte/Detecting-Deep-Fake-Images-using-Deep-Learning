from PIL import Image
from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        transform=None
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")


        label = self.labels[index]


        if self.transform:

            image = self.transform(
                image
            )


        return image, label