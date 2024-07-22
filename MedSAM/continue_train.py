import os
import torch
import argparse
import logging
from datetime import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry
import monai
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from monai.metrics import DiceMetric

# set seeds
torch.manual_seed(2024)
torch.cuda.empty_cache()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

# 设置参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='/jizhicfs/chengpenghu/med_datasets/data_preprocessed/npy/BIN_CT', help="Path to the dataset")
parser.add_argument("--checkpoint", type=str, default="/jizhicfs/chengpenghu/med_datasets/work_dir/SAM/sam_vit_b_01ec64.pth", help="Path to the SAM pre-trained checkpoint")
parser.add_argument("--medsam_checkpoint", type=str, default='/jizhicfs/chengpenghu/med_datasets/work_dir/MedSAM/medsam_vit_b.pth', help="Path to the MedSAM pre-trained checkpoint")
parser.add_argument("--output_dir", "-o", type=str, default="./outputs/test", help="Directory to save the model and logs")
parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size for training")
parser.add_argument("--num_epochs", "-e", type=int, default=150, help="Number of epochs for training")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
args = parser.parse_args()

# 设置日志记录
log_file = os.path.join(args.output_dir, "training.log")
logging.basicConfig(
    filename=log_file,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# 将日志同时输出到控制台
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

class NpyDataset(Dataset):
    def __init__(self, data_root, files, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = os.path.join(data_root, "gts")
        self.img_path = os.path.join(self.data_root, "imgs")
        self.gt_path_files = files
        self.bbox_shift = bbox_shift
        logging.info(f"Number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(os.path.join(self.img_path, img_name), "r", allow_pickle=True)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        gt = np.load(self.gt_path_files[index], "r", allow_pickle=True)
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

class MedSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def main():
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    sam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    # 加载 MedSAM 预训练权重
    medsam_model.load_state_dict(torch.load(args.medsam_checkpoint), strict=False)

    logging.info(
        "Number of total parameters: %d",
        sum(p.numel() for p in medsam_model.parameters()),
    )
    logging.info(
        "Number of trainable parameters: %d",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )

    img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        medsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    )
    logging.info(
        "Number of image encoder and mask decoder parameters: %d",
        sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),
    )
    
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    # 划分训练和验证数据集
    all_data_files = glob.glob(os.path.join(args.data_path, "gts/**/*.npy"), recursive=True)
    train_files, val_files = train_test_split(all_data_files, test_size=0.2, random_state=42)

    train_dataset = NpyDataset(args.data_path, train_files)
    val_dataset = NpyDataset(args.data_path, val_files)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=72,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=72,
        pin_memory=True,
    )

    best_loss = float('inf')
    losses = []
    val_dice_scores = []
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    logging.info("Number of training samples: %d", len(train_dataset))
    logging.info("Number of validation samples: %d", len(val_dataset))

    for epoch in range(args.num_epochs):
        epoch_loss = 0
        medsam_model.train()
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes.detach().cpu().numpy())
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes.detach().cpu().numpy())
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss}")

        # 验证步骤
        medsam_model.eval()
        # val_dice = 0.0
        with torch.no_grad():
            for image, gt2D, boxes, _ in tqdm(val_dataloader):
                image, gt2D = image.to(device), gt2D.to(device)
                medsam_pred = medsam_model(image, boxes.detach().cpu().numpy())
                dice_metric(y_pred=medsam_pred, y=gt2D)
                # dice_score = seg_loss(medsam_pred, gt2D)
                # val_dice += dice_score.item()
        
        # val_dice /= len(val_dataloader)
        val_dice = dice_metric.aggregate().item()
        val_dice_scores.append(val_dice)
        logging.info(f"Epoch {epoch + 1}/{args.num_epochs}, Validation Dice Score: {val_dice}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(medsam_model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        plt.figure()
        plt.plot(losses, label="Training Loss")
        plt.plot(val_dice_scores, label="Validation Dice")
        plt.title("Training Loss and Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, "loss_dice_plot.png"))
        plt.close()

if __name__ == "__main__":
    main()
