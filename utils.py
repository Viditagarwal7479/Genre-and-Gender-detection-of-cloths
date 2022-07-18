import os
from multiprocessing import Pool
from urllib.request import urlretrieve

import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm.notebook import tqdm


def link_to_image(x: tuple) -> None:
    image_download_link, image_id = x
    urlretrieve(image_download_link, f'images/{image_id}.png')


def download_data():
    if len(os.listdir('images')):
        print('images folder is not empty thus images not downloaded')
        return None
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    main = pd.concat((train, test))
    links_id = []
    for i in main.iterrows():
        if i[1]['link'] != 'undefined':
            links_id.append((i[1]['link'], i[1]['id']))
    with Pool(64) as p:
        print(p.map(link_to_image, links_id))
    print(len(os.listdir('images')))


def prepare_csv():
    if os.path.exists('train.csv') and os.path.exists('test.csv'):
        print('train.csv and test.csv already exist thus not prepared again')
        return None
    df = pd.read_csv('styles.csv', usecols=['gender', 'masterCategory', 'id', 'usage'])
    y = df[df['gender'] != 'Unisex']
    y = y[y['gender'] != 'Boys']
    y = y[y['gender'] != 'Girls']
    y = y[y['masterCategory'] != 'Free Items']
    y = y[y['masterCategory'] != 'Accessories']
    y = y[y['masterCategory'] != 'Footwear']
    y = y[y['masterCategory'] != 'Personal Care']
    y = y[y['id'] != 30869]
    path = pd.read_csv('images.csv')
    train = {
        'id': [],
        'path': [],
        'gender': [],  # 2 Men, 3 Women
        'genre': [],  # 0 Ethnic, 1 Western
        'link': []
    }
    test = {
        'id': [],
        'path': [],
        'gender': [],  # 2 Men, 3 Women
        'genre': [],  # 0 Ethnic, 1 Western
        'link': []
    }
    western = 0
    ethnic = 0
    pbar = tqdm(range(5943))
    for i in y.iterrows():
        if western == 3000 and i[1]['usage'] != 'Ethnic':
            continue
        idx = i[1]['id']
        link = str(path[path['filename'] == f'{idx}.jpg']['link'].values[0])
        if link == 'undefined':
            continue
        if ((i[1]['usage'] == 'Ethnic') and ethnic < 2400) or ((i[1]['usage'] != 'Ethnic') and western < 2400):
            train['id'].append(idx)
            train['path'].append(f'images/{idx}.png')
            train['gender'].append(2 + int(i[1]['gender'] != 'Men'))
            train['genre'].append(0 + int(i[1]['usage'] != 'Ethnic'))
            train['link'].append(link)
        else:
            test['id'].append(idx)
            test['path'].append(f'images/{idx}.png')
            test['gender'].append(2 + int(i[1]['gender'] != 'Men'))
            test['genre'].append(0 + int(i[1]['usage'] != 'Ethnic'))
            test['link'].append(link)
        if i[1]['usage'] == 'Ethnic':
            ethnic = ethnic + 1
        else:
            western = western + 1
        pbar.update()
    pbar.close()
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    print('Done train.csv and test.csv have been created')


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, image_transform: torchvision.transforms = None) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = image_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Image, torch.LongTensor]:
        image_label = torch.zeros(4)
        try:
            image = Image.open(self.df.iloc[idx]['path'])
            image_label[0] = int(self.df.iloc[idx]['genre'])
            image_label[1] = int(self.df.iloc[idx]['gender'])
        except:
            image = Image.open('images/40097.png')
            image_label[0] = 2
            image_label[1] = 1
        image = self.transform(image)
        image_label[2] = -1
        image_label[3] = -1
        return image, image_label.type(torch.LongTensor)


if __name__ == '__main__':
    prepare_csv()
    download_data()
