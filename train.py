import argparse
import os.path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel

import nltk
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
nltk.download('punkt')

from utils import EOR, read_json_file
from model import VisionLanguageModel


class CustomDataset(Dataset):
    def __init__(self, data_pairs, image_processor):
        self.data_pairs = data_pairs
        self.image_processor = image_processor

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values']
        return image.squeeze(0)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        data = self.data_pairs[idx]
        image = self.process_image(data['image'])
        seg_image = self.process_image(data['seg_image'])
        return {
            "image": image,
            "seg_image": seg_image,
            "report": data['report']
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    seg_images = torch.stack([item["seg_image"] for item in batch])
    reports = [item["report"] for item in batch]
    return {"image": images, "seg_image": seg_images, "report": reports}


def get_train_data(annotation_file, image_home, seg_image_home):
    data = read_json_file(annotation_file)
    train_data = data['train']

    def convert2pair(data):
        output = []
        for item in data:
            image_path = os.path.join(image_home, item['image_path'][0])
            seg_image_path = os.path.join(seg_image_home, item['image_path'][0])
            # report = data_clean(item['report'])
            # item_id = item['id']
            # target_tensor = target_dict[item_id]
            output.append({
                'image': image_path,
                'seg_image': seg_image_path,
                'report': item['report']
            })

        return output

    train_data = convert2pair(train_data)
    return train_data


def train(args):
    image_processor = ViTFeatureExtractor.from_pretrained(args.vit_model_name)
    if args.use_seg_vit:
        rrg_model = VisionLanguageModel(
            vit_model_name=args.vit_model_name,
            seg_visual_model=args.seg_vit_model_name,
            sentence_pool_dir=args.sentence_pool_dir,
            gamma_r=args.gamma_r,
            gamma_t=args.gamma_t,
            top_k=args.top_k
        )
    else:
        rrg_model = VisionLanguageModel(
            vit_model_name=args.vit_model_name,
            sentence_pool_dir=args.sentence_pool_dir,
            gamma_r=args.gamma_r,
            gamma_t=args.gamma_t,
            top_k=args.top_k
        )
    rrg_model.train_mode()
    if args.frozen_seg_vit:
        rrg_model.frozen_seg_vit()

    # 加载数据
    train_data = get_train_data(args.annotation_file, args.image_home, args.seg_image_home)

    if args.debug:
        train_data = train_data[:1000]

    train_dataset = CustomDataset(
        data_pairs=train_data,
        image_processor=image_processor,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rrg_model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    optimizer = torch.optim.Adam(rrg_model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        rrg_model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            images = batch['image'].to(device)
            seg_images = batch['seg_image'].to(device)
            reports = batch['report']
            o = rrg_model(images, seg_images)  # visual_embeds tensor of shape [batch, dim]

            # Compute training loss using SentencePool
            sentences_batch = [nltk.sent_tokenize(r) for r in reports]
            losses = rrg_model.pool.extract_train(o, sentences_batch)
            batch_loss = losses.mean()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, loss = {avg_loss:.4f}")

    # save final model
    final_model_dir = os.path.join(args.save_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    torch.save(rrg_model.state_dict(), os.path.join(final_model_dir, "model_weights.pth"))
    print(f"Final model saved to {final_model_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="Path to the annotation JSON file")
    parser.add_argument("--image_home", type=str, required=True,
                        help="Path to the directory containing images")
    parser.add_argument("--seg_image_home", type=str, required=True,
                        help="Path to the directory containing images")
    parser.add_argument("--vit_model_name", type=str, default="google/vit-base-patch16-224",
                        help="Name of the pre-trained ViT model")
    parser.add_argument("--seg_vit_model_name", type=str, default="bert-base-uncased",
                        help="Name of the pre-trained BERT model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training")
    parser.add_argument("--save_dir", type=str, default="./model_checkpoints",
                        help="Directory to save the trained model")
    parser.add_argument('--debug', action='store_true',)
    parser.add_argument('--frozen_seg_vit', action='store_true', )
    parser.add_argument('--use_seg_vit', action='store_true', )
    parser.add_argument("--gamma_r", type=float, default=0.5,
                        help="Weight for visual (segmentation) loss")
    parser.add_argument("--gamma_t", type=float, default=0.5,
                        help="Weight for text loss")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top sentences to extract")
    parser.add_argument("--sentence_pool_dir", type=str, required=True,
                        help="Directory containing sentences.txt, sentence_embeddings.npy, image_list.txt, image_embeddings.npy, and sent2img.json")
    args = parser.parse_args()

    train(args)
