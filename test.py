import torch
import torchvision
from tqdm import tqdm

import config
import utils

model = torchvision.models.alexnet()
model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=4)
model = model.load_state_dict(torch.load('alexnet_cloth_genre_gender_new.pth'))
model.to(config.device, non_blocking=True)
test_data = utils.FashionDataset('test.csv', config.transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.test_batch_size, shuffle=False,
                                          pin_memory=True, num_workers=config.num_workers)

with torch.no_grad():
    model.eval()
    total = 0
    gender_accuracy = 0
    genre_accuracy = 0
    for img, label in tqdm(test_loader):
        img, label = img.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True)
        y = model(img)
        genre_accuracy = genre_accuracy + sum(label[:, 0] == y[:, :2].argmax(axis=1))
        gender_accuracy = gender_accuracy + sum(label[:, 1] == 2 + y[:, 2:].argmax(axis=1))
        total = total + len(label)

print('Accuracy of model to detect Gender is : ', gender_accuracy / total)
print('Accuracy of model to detect Genre is : ', genre_accuracy / total)
