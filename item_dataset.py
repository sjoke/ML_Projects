from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import requests
import io
from utils.log import lg
import logging

log = logging.getLogger('no_item_images')
handler = logging.FileHandler('no_item_images.txt', mode='w')
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(message)s'))
log.addHandler(handler)


class ItemDataset(Dataset):
    def __init__(self, items, transforms=None, download=False):
        """
        Args:
            csv_file (string): Path to the csv file with items meta.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.items = items
        self.transforms = transforms
        self.image_path = Path('image_set')
        self.download = download
        if download:
            self.sess = requests.Session()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if not self.download:
            return self.get_from_disk(i)
        else:
            return self.get_from_net(i)
        

    def get_from_disk(self, i):
        row = self.items.iloc[i, :]
        item_id, cate1, label = row['item_id'], row['cate1'], row['label']
        file_path = self.image_path / str(cate1) / str(item_id)

        try:
            img = Image.open(file_path)
            img = img.convert(mode='RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return item_id, label, img
        except Exception:
            return None

    def get_from_net(self, i):
        row = self.items.iloc[i, :]
        item_id, url, label = row['item_id'], row['url'], row['label']
        try:
            r = self.sess.get(url, stream=True)
            img = Image.open(io.BytesIO(r.content))
            img = img.convert(mode='RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return item_id, label, img
        except Exception as e:
            log.info('%s', item_id)
            lg.error("get %s:\n %s", url, e)
            return None
        