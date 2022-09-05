import cv2
import torch.nn as nn
import torch
from torch import Tensor
def get_embed_patch(x: Tensor, patch_size):
    B, C, H, W = x.shape
    x = x.reshape((B, C, H // patch_size, patch_size, W // patch_size, patch_size)).permute(
        (0, 2, 4, 3, 5, 1)).flatten(start_dim=-3, end_dim=-1).flatten(start_dim=1, end_dim=2)
    return x  # [N, num_patches, H // patch_size * W // patch_size * C]


def batch_masking(mask_ratio, all_embed_patch, device=None):

    batch_size, num_patches = all_embed_patch.shape[:2]
    num_mask = int(num_patches * mask_ratio)
    # rand服从均匀分布
    total_indices = torch.rand((batch_size, num_patches)).argsort()
    if device is not None:
        total_indices = total_indices.to(device)
    mask_idx, unmask_idx = total_indices[:, :num_mask], total_indices[:, num_mask:]
    batch_idx = torch.arange(batch_size).reshape((batch_size, 1))
    mask_patches, unmask_patches = all_embed_patch[batch_idx, mask_idx, :], all_embed_patch[batch_idx, unmask_idx, :]
    return mask_idx, mask_patches, unmask_idx, unmask_patches



class MAEEncoder(nn.Module):
    def __init__(self, emb_dim=512, n_head=8, num_encode_layers=6):
        super(MAEEncoder, self).__init__()
        self.encode_layer = nn.TransformerEncoderLayer(emb_dim, n_head)
        self.encoder = nn.TransformerEncoder(self.encode_layer, num_encode_layers)

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape, 'encode')
        return x

class ClassPositionEncode(nn.Module):
    def __init__(self, n_patch, emb_dim):
        super(ClassPositionEncode, self).__init__()
        self.cls = torch.randn((1, 1, emb_dim))
        self.cls_encode = torch.nn.Parameter(self.cls, True)

        self.position = torch.randn((1, n_patch + 1, emb_dim))
        self.pe_encode = torch.nn.Parameter(self.position, True)

    # def forward(self, x):
    #     cls_encodes = torch.expand_copy(self.cls_encode, (x.shape[0], -1, -1))
    #     x = torch.cat([cls_encodes, x], dim=1)
    #     pe = x + self.pe_encode
    #     return pe

    def forward(self, unmask_patch_embed, unmask_idx):
        # unmask_patch_embed [N, num_unmask_patch, emb_dim]

        # tmp = self.pe_encode.squeeze()[unmask_idx + 1, :]

        # tmp2 = []
        # for i in range(unmask_idx.shape[0]):
        #     tmp2.append(self.pe_encode.squeeze()[unmask_idx[i] + 1])
        # tmp2 = torch.stack(tmp2, dim=0)
        # print(tmp2.shape)
        # print(torch.equal(tmp, tmp2))

        pe = unmask_patch_embed + self.pe_encode.squeeze(dim=0)[unmask_idx + 1, :]  # 这里+1是因为cls_pos也有编码
        # print(pe.shape, 'pos_encode')
        return pe





# def get_img_from_patch_embedding(x):
#     x = x.reshape(320 // 8, 320 // 8, 8, 8, 3).permute(0, 2, 1, 3, 4)
#     x = x.flatten(start_dim=0, end_dim=1).flatten(start_dim=1, end_dim=2)
#     out_img = x.detach().numpy()
#     cv2.imshow("1", out_img)
#     cv2.waitKey(0)
#     return out_img


class MAEPatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=512, patch_size=8, device=None):
        super(MAEPatchEmbedding, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.patch = nn.Linear(patch_size * patch_size * in_channels, embed_dim)


    def forward(self, x):
        # [N, C, H, W]
        x = get_embed_patch(x, self.patch_size)
        # [N, total_patch, patch_size ** 2 * in_channels]
        mask_idx, mask_patches, unmask_idx, unmask_patches = batch_masking(0.75, x, self.device)
        # print(mask_patches.shape, 'mask_patches')
        # print(unmask_patches.shape, 'unmask_patches')

        x = self.patch(unmask_patches)
        # print(x.shape, 'unmask_emb_patch')
        return x, unmask_idx, mask_patches, mask_idx


class SelfMultiheadAttention(nn.Module):
    def __init__(self, emb_dim, n_head):
        super(SelfMultiheadAttention, self).__init__()
        self.attn = nn.MultiheadAttention(emb_dim, n_head)

    def forward(self, x):
        x, attn_weight = self.attn(x, x, x)
        return x

class MAEDecoder(nn.Module):
    def __init__(self, emb_dim: int = 512, n_head: int = 8, num_decode_layers=1, in_channels=3, H=320, W=320, patch_size=8):
        super(MAEDecoder, self).__init__()
        self.mask_embed = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.decode_pos_embed = nn.Embedding((H // patch_size) * (W // patch_size), emb_dim)  # 这里并没有cls的Token

        self.decoder = nn.Sequential()
        for _ in range(num_decode_layers):
            self.decoder.append(nn.Sequential(
                SelfMultiheadAttention(emb_dim, n_head),
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim * 4),
                nn.Linear(emb_dim * 4, emb_dim),
                nn.LayerNorm(emb_dim),
            ))

        self.head = nn.Linear(emb_dim, patch_size * patch_size * in_channels)

    def forward(self, unmask_embed_patch, unmask_idx, mask_patch, mask_idx):

        mask_token = self.mask_embed.repeat(mask_patch.shape[0], mask_patch.shape[1], 1)  # 也可以全部重置为0 但是可学习的更好
        mask_token += self.decode_pos_embed(mask_idx)  # 位置编码至关重要

        # print(mask_token.shape, 'mask_token')


        concat_all_tokens = torch.concat([unmask_embed_patch, mask_token], dim=1)
        # print(concat_all_tokens.shape, 'concat_all_tokens')

        # 恢复到原来的次序
        dec_input_tokens = torch.empty_like(concat_all_tokens)
        batch_idx = torch.arange(unmask_embed_patch.shape[0]).reshape(-1, 1)
        shuffle_indices = torch.concat([unmask_idx, mask_idx], dim=1)

        dec_input_tokens[batch_idx, shuffle_indices] = concat_all_tokens  # 绝妙的操作

        out = self.decoder(dec_input_tokens)
        out = out[batch_idx, mask_idx, :]
        out = self.head(out)
        # print(out.shape, 'decode_out')
        return out


class MAE(nn.Module):
    def __init__(self, in_channels=3, emb_dim=512, num_encode_layers=6, n_head=8,
                 H=320, W=320, patch_size=8, device=None):
        super(MAE, self).__init__()
        self.mpe = MAEPatchEmbedding(in_channels, emb_dim, patch_size, device)

        self.pe = ClassPositionEncode((H // patch_size) * (W // patch_size), emb_dim)
        self.encoder = MAEEncoder(emb_dim, n_head, num_encode_layers)

        self.decoder = MAEDecoder(emb_dim, n_head, 1, in_channels, H, W, patch_size)


    def forward(self, x):
        unmask_patches, unmask_idx, mask_patches, mask_idx = self.mpe(x)
        x = self.pe(unmask_patches, unmask_idx)
        x = self.encoder(x)

        out = self.decoder(x, unmask_idx, mask_patches, mask_idx)

        return out, mask_patches, mask_idx, unmask_idx

def main():
    B, C, H, W = 8, 3, 320, 320
    emb_dim = 512
    num_encode_layers = 6
    n_head = 8
    patch_size = 8

    x = torch.randn((B, C, H, W))

    mae = MAE(C, emb_dim, num_encode_layers, n_head, H, W, patch_size)

    out, mask_patches, mask_idx = mae(x)
    batch_idx = torch.arange(B).reshape(B, 1)
    out = out[batch_idx, mask_idx]
    print(out.shape)


if __name__ == '__main__':
    main()

