import math

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from models import MAE
from torch.optim import lr_scheduler
from torchvision import transforms as T
from datasets import LoadDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


epochs = 50
B, C, H, W = 16, 3, 320, 320
emb_dim = 512
num_encode_layers = 6
n_head = 8
patch_size = 10
num_workers = 4

imgs_root = "D:\\test_imgs"

lr = 0.01
device = torch.device('cuda')

batch_idx = torch.arange(B).reshape(B, 1)
net = MAE(C, emb_dim, num_encode_layers, n_head, H, W, patch_size, device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
loss_fn = nn.MSELoss(reduction='mean')

trans = T.Compose([
    T.ToTensor()
])

dataset = LoadDataset(imgs_root, H, W, transform=trans)
total_iters = math.ceil(len(dataset) / B)
# pth_path = "D:\\final.pth"
# net.load_state_dict(torch.load(pth_path))




if __name__ == '__main__':
    writer = SummaryWriter(r'runs/training')
    train_loader = DataLoader(dataset, B, True, num_workers=num_workers)

    for epoch in range(1, epochs + 1):
        net.train()
        train_tqdm = tqdm(train_loader)
        total_loss = 0
        iters = 0
        for input_data in train_tqdm:
            optimizer.zero_grad()
            input_data = input_data.to(device)

            out, mask_patches, _, _ = net(input_data)

            loss = loss_fn(input=out, target=mask_patches)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iters += 1
            train_tqdm.set_description(f"epoch: {epoch}, loss: {format(total_loss / iters, '.5f')}, lr: {format(lr_scheduler.get_last_lr()[0], '.5f')}")
            writer.add_scalar('loss', total_loss / iters, iters + (epoch - 1) * total_iters)
            writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], iters + (epoch - 1) * total_iters)
        lr_scheduler.step()
    torch.save(net.state_dict(), "D:\\final.pth")
    writer.close()

