# Mem-ERR

A memory-augmented, retrieval-based radiology report generator built on top of CLIP visual encoders and a sentence-pool indexing mechanism.

## Repository Layout

```
.
├── README.md                   ← This file
├── train.py                    ← Training script
├── test.py                     ← Inference script
├── model.py                    ← VisionLanguageModel + Memory & SentencePool
├── segmentation/
│   ├── segment_datasets.py     ← Utilities for generating overlaid segmentation maps
│   └── …                       ← Additional helpers for segmentation
├── data/
│   └── sentence_pool/          ← Precomputed sentence-pool & cluster files
└── utils.py                    ← General helper functions (I/O, JSON, etc.)
```

## Files

- **train.py**
  Loads annotations and images (and optional segmentation), tokenizes reports into sentences, runs the VisionLanguageModel to get fused visual embeddings, and calls `SentencePool.extract_train` to compute and back-propagate the weighted visual/text loss. Saves model weights and copies the entire `sentence_pool/` directory into the checkpoint folder.

- **test.py**
  Loads a saved model (including the bundled `sentence_pool/`), preprocesses test images, runs the model for visual embeddings, then uses `SentencePool.extract_test` with cluster indexing to retrieve the top‑k sentences. Outputs a JSON of `{ "id": ..., "prediction": [...] }` for each case.

- **model.py**
  Defines:
  - **VisionLanguageModel**: Dual‐stream CLIP visual encoder (raw + segmented) → fusion Transformer → MemoryModule → average pooling.
  - **MemoryModule**: A trainable key/value memory bank that enhances the fused visual representation via cross‑modal attention.
  - **SentencePool**: Loads precomputed sentence and image embeddings plus cluster mappings; provides `extract_train` and `extract_test` methods.

- **segmentation/segment_datasets.py**
  Applies a pre‑trained PSPNet segmentation model to chest X‑rays, overlays colored masks for each anatomical group, and saves the results for downstream processing.

- **data/sentence_pool/**
  Contains precomputed:
  - `sentences.txt` (one sentence per line)
  - `sentence_embeddings.npy` (sentence vectors)
  - `image_list.txt` (list of image names)
  - `image_embeddings.npy` (image feature matrices)
  - `sent2img.json` (mapping from sentence to associated image indices)
  - `cluster2sent.json` (clusters of sentence IDs)
  - `cluster_embeddings.npy` (cluster centroids for fast retrieval)

## Quick Start

### Installation

```bash
pip install torch torchvision transformers torchxrayvision scikit-image opencv-python tqdm nltk
```

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
  --sentence_pool_dir ./checkpoints/rrg/sentence_pool \
  --annotation_file ./data/annotations.json \
  --image_home ./data/images \
  --seg_image_home ./data/segmentations \
  --output_data ./predictions.json \
  --vit_model_name openai/clip-vit-base-patch32 \
  --gamma_r 0.5 \
  --gamma_t 0.5 \
  --top_k 5
```

## License

MIT License
