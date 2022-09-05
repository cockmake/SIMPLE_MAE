import numpy as np

from models import MAE
import torch
import cv2.cv2 as cv
from models import get_embed_patch
from torchvision import transforms as T

device = torch.device('cuda')

C, H, W = 3, 320, 320
patch_size = 8
net = MAE(3, 512, 6, 8, H, W, patch_size)

trans = T.Compose([
    T.ToTensor()
])
def init_model():
    pth_path = "D:\\final.pth"
    net.load_state_dict(torch.load(pth_path))
    net.eval()


def get_img_from_patch_embedding(x):
    x = x.reshape(H // patch_size, W // patch_size, patch_size, patch_size, C).permute(0, 2, 1, 3, 4)
    x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
    out_img = x.detach().numpy()
    return out_img

def infer_img(img_path):

    img = cv.imread(img_path)
    # src_H, src_W = img.shape[:-1]

    img = cv.resize(img, (W, H))
    cv.imshow("src", img)
    x = trans(img).unsqueeze(dim=0)
    all_patches = get_embed_patch(x, patch_size)

    out, _, mask_idx, unmask_idx = net(x)
    board = torch.zeros((1, 1600, 192))
    board[0, mask_idx, :] = out.detach()
    board[0, unmask_idx, :] = all_patches[0, unmask_idx, :]

    img2 = (get_img_from_patch_embedding(board) * 255).astype(np.uint8)
    cv.imshow("out", img2)
    cv.waitKey(0)

def main():
    infer_img(r"D:\test_imgs\person.png")

if __name__ == '__main__':
    main()