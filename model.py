import torch.nn as nn
from transformers import ViTModel
import torch.nn.functional as F
from safetensors.torch import load_file
import os, json, numpy as np, torch
import shutil


class SentencePool:
    def __init__(self, pool_dir, gamma_r=0.5, gamma_t=0.5, top_k=5):
        txt_path = os.path.join(pool_dir, "sentences.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f if line.strip()]
        # Build sentence text → index mapping
        self.sentence2id = {s: idx for idx, s in enumerate(self.sentences)}
        sent_emb = np.load(os.path.join(pool_dir, "sentence_embeddings.npy"))
        self.sentence_embeddings = torch.from_numpy(sent_emb)  # CPU tensor
        img_emb = np.load(os.path.join(pool_dir, "image_embeddings.npy"))
        self.image_embeddings = torch.from_numpy(img_emb)      # CPU tensor
        with open(os.path.join(pool_dir, "sent2img.json"), "r", encoding="utf-8") as f:
            self.sent2img = json.load(f)

        # Load cluster to sentences mapping
        cluster_map_path = os.path.join(pool_dir, "cluster2sent.json")
        with open(cluster_map_path, "r", encoding="utf-8") as f:
            self.cluster2sent = json.load(f)
        # Load cluster embeddings
        cluster_emb_path = os.path.join(pool_dir, "cluster_embeddings.npy")
        cluster_emb_np = np.load(cluster_emb_path)
        self.cluster_embeddings = torch.from_numpy(cluster_emb_np)  # CPU tensor

        self.gamma_r = gamma_r
        self.gamma_t = gamma_t
        self.top_k = top_k

    def extract_train(self, visual_embeds: torch.Tensor, gold_sentences_list: list[list[str]]) -> torch.Tensor:
        """
        Compute the combined score for known gold-standard sentences during training.
        visual_embeds: Tensor of shape (batch_size, seq_len, dim) or (batch_size, dim)
        gold_sentences_list: list of length batch_size, each a list of gold sentence texts
        Returns: Tensor of shape (batch_size,) containing the average combined score per example
        """
        batch_size, dim = visual_embeds.shape[0], visual_embeds.shape[-1]
        device = visual_embeds.device
        S = len(self.sentences)
        losses = []

        # For text similarity, average over seq dimension; for visual similarity, keep full matrix
        if visual_embeds.dim() == 3:
            o_mat = visual_embeds         # (batch_size, seq_len, dim)
            o_vec = o_mat.mean(dim=1)     # (batch_size, dim)
        else:
            o_mat = visual_embeds.unsqueeze(1)  # (batch_size, 1, dim)
            o_vec = visual_embeds                # (batch_size, dim)

        for b in range(batch_size):
            o_b_mat = o_mat[b]  # (seq_len, dim)
            o_b_vec = o_vec[b]  # (dim,)
            gold_texts = gold_sentences_list[b]
            # Map gold texts to indices, skipping any not in the pool
            gold_ids = []
            for text in gold_texts:
                idx = self.sentence2id.get(text, None)
                if idx is not None:
                    gold_ids.append(idx)
            if not gold_ids:
                # No valid gold sentences found, skip this example
                losses.append(torch.tensor(0.0, device=device))
                continue
            # Compute text and visual scores for each gold sentence
            combined_scores = []
            for i in gold_ids:
                # Text score
                t_i = self.sentence_embeddings[i].to(device)
                t_diff = t_i - o_b_vec  # (dim,)
                score_t = t_diff.pow(2).sum()
                # Visual score
                img_idx_list = self.sent2img.get(str(i), [])
                # compute minimal squared distance over all associated images
                min_dist = torch.tensor(float('inf'), device=device)
                for img_idx in img_idx_list:
                    if 0 <= img_idx < self.image_embeddings.size(0):
                        R_sij = self.image_embeddings[img_idx].to(device)  # (dim, seq_len)
                        dist = (R_sij - o_b_mat).pow(2).sum()
                        min_dist = torch.min(min_dist, dist)
                score_r = min_dist
                # Combined
                combined = self.gamma_r * score_r + self.gamma_t * score_t
                combined_scores.append(combined)
            # Average over gold sentences
            batch_loss = torch.stack(combined_scores).mean()
            losses.append(batch_loss)
        return torch.stack(losses)

    def extract_test(self, visual_embeds: torch.Tensor) -> list[list[str]]:
        """
        Inference-time extraction with embedding indexing.
        visual_embeds: (batch_size, seq_len, dim) or (batch_size, dim)
        Returns: list of top_k predicted sentences per example.
        """
        # Prepare visual matrix and vector
        if visual_embeds.dim() == 3:
            o_mat = visual_embeds           # (batch, seq_len, dim)
            o_vec = o_mat.mean(dim=1)       # (batch, dim)
        else:
            o_mat = visual_embeds.unsqueeze(1)  # (batch, 1, dim)
            o_vec = visual_embeds              # (batch, dim)

        batch_size = o_vec.size(0)
        device = o_vec.device

        # Move cluster embeddings once
        cluster_embs = self.cluster_embeddings.to(device)  # (num_clusters, D)
        top_k = self.top_k

        results = []
        for b in range(batch_size):
            # Compute o_f = [flatten(o_mat[b]); o_vec[b]]
            o_flat = o_mat[b].flatten()          # (seq_len * dim,)
            o_f = torch.cat([o_flat, o_vec[b]], dim=0)  # (D,)

            # Find nearest cluster
            diffs = cluster_embs - o_f.unsqueeze(0)      # (num_clusters, D)
            dists = diffs.pow(2).sum(dim=1)             # (num_clusters,)
            best_cluster = dists.argmin().item()

            # Candidate sentence IDs
            cand_ids = self.cluster2sent.get(str(best_cluster), [])

            # Score each candidate
            scores = []
            for i in cand_ids:
                # Text score
                t_i = self.sentence_embeddings[i].to(device)
                score_t = (t_i - o_vec[b]).pow(2).sum()
                # Visual score
                img_idxs = self.sent2img.get(str(i), [])
                if img_idxs:
                    dists_r = []
                    for idx in img_idxs:
                        if 0 <= idx < self.image_embeddings.size(0):
                            R_sij = self.image_embeddings[idx].to(device)  # (seq_len, dim)
                            dists_r.append((R_sij - o_mat[b]).pow(2).sum())
                    score_r = torch.stack(dists_r).min() if dists_r else torch.tensor(float('inf'), device=device)
                else:
                    score_r = torch.tensor(float('inf'), device=device)

                scores.append((self.gamma_r * score_r + self.gamma_t * score_t, i))

            # Select top_k sentences by lowest score
            scores.sort(key=lambda x: x[0].item())
            top_ids = [i for _, i in scores[:top_k]]
            results.append([self.sentences[i] for i in top_ids])

        return results


class VisionModel(nn.Module):
    def __init__(self, vit_model_name):
        super(VisionModel, self).__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_dim = self.vit.config.hidden_size
        self.vit.config.name_or_path = vit_model_name

        self.mlp = nn.Sequential(
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim),
        )


    def frozen_vit(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, image, target=None, **kwargs):
        # Step 1: ViT处理图像
        vit_outputs = self.vit(pixel_values=image)
        vit_embeddings = vit_outputs.last_hidden_state  # (batch_size, seq_len_vit, vit_hidden_dim)

        hidden_states = self.mlp(vit_embeddings)  # (batch_size, seq_len_vit, hidden_dim)

        if target is not None:
            # assert target_mask is not None, "target_mask must be provided when target is provided."
            image_embeds = F.normalize(hidden_states, p=2, dim=-1)
            image_embeds = image_embeds.view(-1, self.vit_hidden_dim)
            text_embeds = F.normalize(target, p=2, dim=-1)
            text_embeds = text_embeds.view(-1, self.vit_hidden_dim)
            loss = self.loss_fn(text_embeds, image_embeds)
            # return loss
            return {'loss': loss}
        else:
            return {'hidden_states': hidden_states}

    @classmethod
    def from_pretrained(cls, model_dir):
        config = torch.load(os.path.join(model_dir, "config.pth"))
        vit_model_name = config["vit_model_name"]

        model = cls(vit_model_name)

        model_weights_path = os.path.join(model_dir, "model_weights.pth")
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {model_dir}")

        return model

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        config = {
            "vit_model_name": self.vit.config.name_or_path,
        }
        torch.save(config, os.path.join(save_dir, "config.pth"))

        model_weights_path = os.path.join(save_dir, "model_weights.pth")
        torch.save(self.state_dict(), model_weights_path)

        print(f"Model saved to {save_dir}")



class MemoryModule(nn.Module):
    def __init__(self, memory_size: int, dim: int):
        super().__init__()
        # Memory bank
        self.M = nn.Parameter(torch.randn(memory_size, dim))
        # Key and value projections
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: (batch, seq_len, dim)
        returns: O = H + A, same shape
        """
        # Compute keys and values
        K = self.Wk(self.M)  # (memory_size, dim)
        V = self.Wv(self.M)  # (memory_size, dim)
        # Attention weights: H @ K^T -> (batch, seq_len, memory_size)
        W = torch.softmax(torch.matmul(H, K.t()), dim=-1)
        # Read from values: W @ V -> (batch, seq_len, dim)
        A = torch.matmul(W, V)
        # Enhance H
        return H + A


class VisionLanguageModel(nn.Module):
    def __init__(self, vit_model_name,
                 seg_visual_model=None,
                 sentence_pool_dir=None,
                 gamma_r=0.5,
                 gamma_t=0.5,
                 top_k=5):
        super(VisionLanguageModel, self).__init__()
        self.gamma_r = gamma_r
        self.gamma_t = gamma_t
        self.top_k = top_k

        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.vit_hidden_dim = self.vit.config.hidden_size
        self.vit_model_name = vit_model_name
        self.seg_visual_model = seg_visual_model

        #
        if seg_visual_model is not None:
            self.seg_vit = VisionModel(vit_model_name)
            model_weights_path = os.path.join(seg_visual_model, "model_weights.pth")
            self.seg_vit.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
            print(f"Model loaded from {seg_visual_model}")
            self.use_seg_vit = True
        else:
            self.seg_visual_model = None
            self.use_seg_vit = False

        # Initialize sentence pool
        if sentence_pool_dir is not None:
            self.pool = SentencePool(sentence_pool_dir, gamma_r=self.gamma_r, gamma_t=self.gamma_t, top_k=self.top_k)
            self.sentence_pool_dir = sentence_pool_dir
        else:
            self.pool = None
            self.sentence_pool_dir = None

        self.mlp = nn.Sequential(
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.vit_hidden_dim, self.vit_hidden_dim)
        )

        # Projection and fusion transformer (optional, skip if already present)
        # Memory module for vision-text alignment, size matches number of sentences
        if self.pool is None:
            raise ValueError("SentencePool must be initialized before MemoryModule")
        memory_size = len(self.pool.sentences)
        self.memory = MemoryModule(memory_size=memory_size, dim=self.vit_hidden_dim)

    def frozen_seg_vit(self):
        for param in self.seg_vit.parameters():
            param.requires_grad = False

    def forward(self, image, seg_image=None, **kwargs):
        vit_outputs = self.vit(pixel_values=image)
        vit_embeddings = vit_outputs.last_hidden_state  # (batch_size, seq_len_vit, vit_hidden_dim)

        raw_hidden = self.mlp(vit_embeddings)  # (batch, seq, dim)
        raw_embeds = raw_hidden.mean(dim=1)    # (batch, dim)

        if self.use_seg_vit and seg_image is not None:
            seg_outputs = self.seg_vit(seg_image)
            # adjust if seg_vit returns a dict
            seg_states = seg_outputs.last_hidden_state if hasattr(seg_outputs, 'last_hidden_state') else seg_outputs['hidden_states']
            seg_hidden = self.mlp(seg_states)   # (batch, seq, dim)
            seg_embeds = seg_hidden.mean(dim=1) # (batch, dim)

            fused_hidden = raw_hidden + seg_hidden  # (batch, seq_len, dim)
        else:
            seg_embeds = raw_embeds
            fused_hidden = raw_hidden              # (batch, seq_len, dim)

        # Memory-based cross-modal alignment
        H = self.memory(fused_hidden)  # (batch, seq_len, dim)

        return {
            'raw_embeds': raw_embeds,
            'seg_embeds': seg_embeds,
            'visual_embeds': H
        }

    @classmethod
    def from_pretrained(cls, model_dir):
        """
        Load a pretrained VisionLanguageModel from the specified directory.
        :param model_dir: Directory where the model is saved
        :return: An instance of VisionLanguageModel
        """
        config = torch.load(os.path.join(model_dir, "config.pth"))
        vit_model_name = config["vit_model_name"]
        seg_vit_model_name = config["seg_vit_model_name"]
        use_seg_vit = config.get("use_seg_vit", True)
        gamma_r = config.get("gamma_r", 0.5)
        gamma_t = config.get("gamma_t", 0.5)
        top_k   = config.get("top_k",   5)

        pool_dir = os.path.join(model_dir, "sentence_pool")
        if use_seg_vit:
            model = cls(vit_model_name, seg_vit_model_name, sentence_pool_dir=pool_dir,
                        gamma_r=gamma_r, gamma_t=gamma_t, top_k=top_k)
        else:
            model = cls(vit_model_name, None, sentence_pool_dir=pool_dir,
                        gamma_r=gamma_r, gamma_t=gamma_t, top_k=top_k)

        model_weights_path = os.path.join(model_dir, "model_weights.pth")
        if os.path.exists(model_weights_path):
            weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            model_weights_path = os.path.join(model_dir, "model.safetensors")
            weights = load_file(model_weights_path, device='cpu')
        model.load_state_dict(weights, strict=False)
        print(f"Model loaded from {model_dir}")
        return model

    def save_pretrained(self, save_dir):
        """
        Save the model to the specified directory.
        :param save_dir: Directory in which to save the model
        """
        os.makedirs(save_dir, exist_ok=True)

        # 保存配置
        config = {
            "vit_model_name": self.vit_model_name,
            "seg_vit_model_name": self.seg_visual_model,
            'use_seg_vit': self.use_seg_vit,
            'gamma_r': self.gamma_r,
            'gamma_t': self.gamma_t,
            'top_k':   self.top_k,
        }
        torch.save(config, os.path.join(save_dir, "config.pth"))

        # 保存模型权重
        model_weights_path = os.path.join(save_dir, "model_weights.pth")
        torch.save(self.state_dict(), model_weights_path)

        # Copy sentence pool directory
        if self.sentence_pool_dir:
            dst = os.path.join(save_dir, "sentence_pool")
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(self.sentence_pool_dir, dst)
            print(f"Sentence pool copied to {dst}")

        print(f"Model saved to {save_dir}")


def test_sentence_pool():
    # Example test for SentencePool
    pool_dir = "./data/sentence_pool"
    print(f"Loading SentencePool from {pool_dir}...")
    pool = SentencePool(pool_dir, gamma_r=0.5, gamma_t=0.5, top_k=5)

    # Create a random batch of visual embeddings
    batch_size = 3
    seq_len = 2
    dim = pool.image_embeddings.shape[-1] if pool.image_embeddings.dim() > 1 else pool.sentence_embeddings.shape[1]
    visual_embeds = torch.randn(batch_size, seq_len, dim)

    # Example gold sentences (use first two sentences from pool)
    gold_samples = [pool.sentences[:2] for _ in range(batch_size)]

    # Training-time scoring
    train_losses = pool.extract_train(visual_embeds, gold_samples)
    print("Training-time losses:", train_losses.tolist())

    # Inference-time extraction
    test_preds = pool.extract_test(visual_embeds)
    for i, preds in enumerate(test_preds):
        print(f"Example {i} predictions:")
        for sent in preds:
            print("  -", sent)


def main():
    # Instantiate the model
    model = VisionLanguageModel()
    # Example input
    image = torch.randn(8, 3, 224, 224)
    # Model forward pass
    logits = model(image)
    print("Logits shape:", logits.shape)  # (batch_size, seq_len_vit+seq_len_bert, large_matrix_dim)


# Test harness for SentencePool
if __name__ == "__main__":
    test_sentence_pool()
