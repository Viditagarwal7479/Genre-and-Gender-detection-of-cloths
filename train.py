import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

import config
import utils

utils.prepare_csv()
utils.download_data()

model = torchvision.models.alexnet()
model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features, out_features=4)
model.to(config.device)
loss_function = torch.nn.MultiLabelMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
train_data = utils.FashionDataset('train.csv', config.transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.train_batch_size, shuffle=True,
                                           pin_memory=True, num_workers=config.num_workers)

losses_epoch = []
for i in range(config.epochs):
    losses = []
    for img, label in tqdm(train_loader):
        img, label = img.to(config.device, non_blocking=True), label.to(config.device, non_blocking=True)
        y = model(img)
        loss = loss_function(y, label)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(sum(losses) / len(losses))
    losses_epoch.append(sum(losses) / len(losses))
plt.plot(range(len(losses_epoch)), losses_epoch)
plt.show()

model.cpu()
torch.save(model.state_dict(), 'alexnet_cloth_genre_gender_new.pth')
print('Done')
