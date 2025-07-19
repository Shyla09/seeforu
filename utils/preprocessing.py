#Preprocessing
import os
import nltk
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.tokenizer import Vocabulary

nltk.download('punkt')

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None):
        self.root_dir = root_dir
        self.df = self._load_captions(captions_file)
        self.transform = transform
        self.vocab = vocab

    def _load_captions(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        data = [line.strip().split('\t') for line in lines]
        ids = [x[0].split('#')[0] for x in data]
        captions = [x[1] for x in data]
        return list(zip(ids, captions))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_id, caption = self.df[index]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)
