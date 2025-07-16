import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, ViTFeatureExtractor, ViTModel
from train import VisionLanguageModel, CustomDataset, collate_fn
import numpy as np
import nltk
from PIL import Image
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import json
from model import SentencePool


def test(args):
    # Load feature extractor and model
    image_processor = ViTFeatureExtractor.from_pretrained(args.vit_model_name)
    model = VisionLanguageModel.from_pretrained(args.model_dir)
    model = model.to(args.device)
    model.eval()
    # Load data
    with open(args.annotation_file, 'r') as f:
        all_data = json.load(f)
    test_data = all_data.get('test', [])
    if args.debug:
        test_data = test_data[:5]

    results = []
    for item in tqdm(test_data, desc='Testing'):
        # Load and preprocess image
        img_path = os.path.join(args.image_home, item['image_path'][0])
        image = Image.open(img_path).convert("RGB")
        image = image_processor(images=[image], return_tensors="pt")['pixel_values'].to(args.device)

        # Optionally load seg image
        if model.use_seg_vit:
            seg_path = os.path.join(args.seg_image_home, item['image_path'][0])
            seg_img = Image.open(seg_path).convert("RGB")
            seg_img = image_processor(images=[seg_img], return_tensors="pt")['pixel_values'].to(args.device)
            with torch.no_grad():
                o = model(image.unsqueeze(0), seg_img.unsqueeze(0))  # (1, dim)
        else:
            with torch.no_grad():
                o = model(image.unsqueeze(0))  # (1, dim)

        # Extract sentences
        preds = model.pool.extract_test(o)  # list of lists
        results.append({
            'id': item['id'],
            'prediction': preds[0]
        })

    # Save predictions
    with open(args.output_data, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} predictions to {args.output_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add command-line arguments
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="Path to the annotation JSON file")
    parser.add_argument("--image_home", type=str, required=True,
                        help="Path to the directory containing images")
    parser.add_argument("--seg_image_home", type=str, required=True,
                        help="Path to the directory containing segmentation images")
    parser.add_argument("--vit_model_name", type=str, default="google/vit-base-patch16-224",
                        help="Name of the pre-trained ViT model")
    # parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased",
    #                     help="Name of the pre-trained BERT model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the saved model")
    parser.add_argument("--output_data", type=str, required=True,
                        help="Directory containing the saved model")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the evaluation on")
    parser.add_argument('--debug', action='store_true',)

    args = parser.parse_args()

    test(args)
