# Mem-ERR

## Repository Layout

```
.
├── README.md                   ← This file
├── train.py                    ← Training script
├── test.py                     ← Inference script
├── model.py                    ← VisionLanguageModel + Memory & SentencePool
├── segment/
│   └── segment_datasets.py     ← Utilities for generating overlaid segmentation maps
└── data/
    └── sentence_pool/          ← Precomputed sentence-pool & cluster files
```

## Files

- **data/sentence_pool/**  
  Contains precomputed:
  - `sentences.txt` (one sentence per line)
  - `sentence_embeddings.npy` (sentence vectors)
  - `image_list.txt` (list of image names)
  - `image_embeddings.npy` (image feature matrices)
  - `sent2img.json` (mapping from sentence to associated image indices)
  - `cluster2sent.json` (clusters of sentence IDs)
  - `cluster_embeddings.npy` (cluster centroids for fast retrieval)
  - **Note:** The contents of `data/sentence_pool/` are provided for example only and do not represent real results.

## Quick Start

### Training

```bash
python train.py \
  --data_file ./data/annotations.json \
  --image_home ./data/images \
  --seg_image_home ./data/segmentations \
  --sentence_pool_dir ./data/sentence_pool \
  --save_dir ./checkpoints/rrg \
  --vit_model_name openai/clip-vit-base-patch32 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --epochs 10 \
  --gamma_r 0.5 \
  --gamma_t 0.5 \
  --top_k 5
```

### Inference

```bash
python test.py \
  --model_dir ./checkpoints/rrg \
  --annotation_file ./data/annotations.json \
  --image_home ./data/images \
  --seg_image_home ./data/segmentations \
  --output_data ./predictions.json \
  --vit_model_name openai/clip-vit-base-patch32
```
