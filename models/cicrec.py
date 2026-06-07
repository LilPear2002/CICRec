import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.model_utils import TransformerLayer
from config.configurator import configs
import os


class CICRec(BaseModel):
    def __init__(self, data_handler):
        super(CICRec, self).__init__(data_handler)

        # Basic configuration
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        self.mask_token = self.item_num + 1

        # Transformer config
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.dropout_rate = configs['model']['dropout_rate']
        self.inner_size = 4 * self.emb_size

        # Semantic neighbor config (ColaKG style)
        self.semantic_k = configs['model'].get('semantic_k', 10)
        self.semantic_hid = configs['model'].get('semantic_hid', 32)
        self.dropout_neighbor = configs['model'].get('dropout_neighbor', 0.1)

        # Contrastive learning config
        self.batch_size = configs['train']['batch_size']
        self.cl_weight = configs['model'].get('cl_weight', 0.1)
        self.tau = configs['model'].get('tau', 1.0)

        # Load frozen EasyRec semantic embeddings
        dataset_name = configs['data']['name']
        easyrec_filename = f'{dataset_name}_easyrec_embeddings.pt' if dataset_name != 'mooc' else 'course_easyrec_embeddings.pt'
        easyrec_path = os.path.join(configs['data']['dir'], easyrec_filename)
        if os.path.exists(easyrec_path):
            print(f"Loading EasyRec embeddings from {easyrec_path}...")
            easyrec_emb = torch.load(easyrec_path, map_location='cpu')
            # Keep padding vector zero so it does not participate in semantic computation.
            easyrec_emb[0] = 0.0
            # Add mask token embedding as a zero vector.
            mask_emb = torch.zeros(1, easyrec_emb.shape[1])
            easyrec_emb = torch.cat([easyrec_emb, mask_emb], dim=0)
            self.register_buffer('easyrec_embeddings', easyrec_emb)
            self.use_semantic = True
            print(f"EasyRec embeddings loaded. Shape: {easyrec_emb.shape}")

            # Pre-compute the global top-k semantic neighbor index.
            self._build_semantic_neighbors()
        else:
            print("Warning: EasyRec embeddings not found.")
            self.use_semantic = False

        # Semantic adapter MLP
        if self.use_semantic:
            self.semantic_adapter = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.emb_size),
                nn.LayerNorm(self.emb_size)
            )

            # ColaKG-style attention for neighbor aggregation
            self.W_neighbor = nn.Parameter(torch.empty(size=(1024, self.semantic_hid)))
            nn.init.xavier_uniform_(self.W_neighbor.data, gain=1.414)
            self.a_neighbor = nn.Parameter(torch.empty(size=(2 * self.semantic_hid, 1)))
            nn.init.xavier_uniform_(self.a_neighbor.data, gain=1.414)
            self.leakyrelu = nn.LeakyReLU(0.2)

        # Embedding layers
        self.token_emb = nn.Embedding(self.item_num + 2, self.emb_size, padding_idx=0)
        self.position_emb = nn.Embedding(self.max_len, self.emb_size)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        # Transformer layers for long-term intent
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.emb_size, self.n_heads, self.inner_size, self.dropout_rate)
            for _ in range(self.n_layers)
        ])

        # Gated fusion MLP (long-term & short-term -> scalar gate)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_size, self.emb_size),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1)
        )

        # Loss functions
        self.loss_func = nn.CrossEntropyLoss()
        self.cl_loss_func = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _build_semantic_neighbors(self):
        """Pre-compute global top-k semantic neighbors based on cosine similarity."""
        print(f"Building semantic neighbor index with K={self.semantic_k}...")

        sem_emb = self.easyrec_embeddings  # [item_num+2, 1024]
        sem_emb_norm = F.normalize(sem_emb, dim=1)

        sim_matrix = torch.matmul(sem_emb_norm, sem_emb_norm.T)  # [N, N]
        sim_matrix.fill_diagonal_(-float('inf'))
        sim_matrix[:, 0] = -float('inf')
        sim_matrix[:, -1] = -float('inf')

        _, topk_indices = torch.topk(sim_matrix, self.semantic_k, dim=-1)  # [N, K]

        self.register_buffer('neighbor_indices', topk_indices)
        print(f"Semantic neighbor index built. Shape: {topk_indices.shape}")

    def get_item_emb(self, item_ids):
        """Return fused item embeddings (ID + semantic)."""
        id_emb = self.token_emb(item_ids)
        if self.use_semantic:
            sem_emb = self.semantic_adapter(self.easyrec_embeddings[item_ids])
            return id_emb + sem_emb
        return id_emb

    def long_term_encoder(self, batch_seqs):
        """Transformer + reverse position encoding + masked mean pooling."""
        mask = (batch_seqs > 0).unsqueeze(1).repeat(1, batch_seqs.size(1), 1).unsqueeze(1)

        x = self.get_item_emb(batch_seqs)

        batch_size, seq_len = batch_seqs.shape
        reverse_pos = torch.arange(seq_len - 1, -1, -1, device=batch_seqs.device)
        reverse_pos = reverse_pos.unsqueeze(0).expand(batch_size, -1)
        x = x + self.position_emb(reverse_pos)
        x = self.emb_dropout(x)

        for transformer in self.transformer_layers:
            x = transformer(x, mask)

        valid_mask = (batch_seqs > 0).float().unsqueeze(-1)
        long_term = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        return long_term

    def short_term_encoder(self, batch_seqs):
        """Short-term intent encoder with semantic neighbor enhancement."""
        batch_size, seq_len = batch_seqs.shape
        device = batch_seqs.device

        lengths = (batch_seqs > 0).sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(batch_size, device=device)
        last_items = batch_seqs[batch_indices, lengths]

        last_item_emb = self.get_item_emb(last_items)  # [B, D]

        if not self.use_semantic:
            return last_item_emb

        # ColaKG-style neighbor enhancement
        neighbor_idx = self.neighbor_indices[last_items]  # [B, K]
        neighbor_emb = self.get_item_emb(neighbor_idx)  # [B, K, D]

        last_item_sem = self.easyrec_embeddings[last_items]  # [B, 1024]
        neighbor_sem = self.easyrec_embeddings[neighbor_idx]  # [B, K, 1024]

        Wh_i = torch.matmul(last_item_sem, self.W_neighbor)  # [B, semantic_hid]
        Wh_j = torch.matmul(neighbor_sem, self.W_neighbor)  # [B, K, semantic_hid]

        Wh_i_expanded = Wh_i.unsqueeze(1).expand(-1, self.semantic_k, -1)  # [B, K, semantic_hid]
        W_concat = torch.cat([Wh_i_expanded, Wh_j], dim=-1)  # [B, K, 2*semantic_hid]

        attention = torch.matmul(W_concat, self.a_neighbor).squeeze(-1)  # [B, K]
        attention = self.leakyrelu(attention)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout_neighbor, training=self.training)

        attention = attention.unsqueeze(-1)  # [B, K, 1]
        neighbor_agg = torch.sum(attention * neighbor_emb, dim=1)  # [B, D]

        short_emb = F.elu((last_item_emb + neighbor_agg) / 2)
        return short_emb

    def forward(self, batch_seqs):
        """Forward pass: long-term + short-term gated fusion."""
        long_term = self.long_term_encoder(batch_seqs)
        short_term = self.short_term_encoder(batch_seqs)

        gate_input = torch.cat([long_term, short_term], dim=-1)
        alpha = torch.sigmoid(self.gate_mlp(gate_input))
        final_emb = alpha * long_term + (1.0 - alpha) * short_term
        return final_emb

    # ==================== CL4SRec augmentation ====================
    def _cl4srec_aug(self, batch_seqs):
        """CL4SRec augmentations: crop, mask, reorder."""
        def item_crop(seq, length, eta=0.6):
            num_left = max(math.floor(length * eta), 1)
            croped_item_seq = np.zeros_like(seq)
            croped_item_seq[-num_left:] = seq[-num_left:]
            return croped_item_seq.tolist(), num_left

        def item_mask(seq, length, gamma=0.3):
            num_mask = math.floor(length * gamma)
            mask_index = random.sample(range(1, length), k=num_mask) if num_mask > 0 else []
            masked_item_seq = seq[:]
            mask_index = [-i-1 for i in mask_index]
            masked_item_seq[mask_index] = self.mask_token
            return masked_item_seq.tolist(), length

        def item_reorder(seq, length, beta=0.6):
            num_reorder = math.floor(length * beta)
            reorder_begin = random.randint(0, length - num_reorder - 1)
            reordered_item_seq = seq[:]
            shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
            random.shuffle(shuffle_index)
            shuffle_index_neg = [-(length - i) for i in shuffle_index]
            target_start = -(length - reorder_begin)
            target_end = -(length - reorder_begin - num_reorder) if (length - reorder_begin - num_reorder) > 0 else None
            reordered_item_seq[target_start:target_end] = [reordered_item_seq[i] for i in shuffle_index_neg]
            return reordered_item_seq.tolist(), length

        seqs = batch_seqs.tolist()
        lengths = batch_seqs.count_nonzero(dim=1).tolist()

        aug_seq1, aug_seq2 = [], []
        for seq, length in zip(seqs, lengths):
            seq = np.asarray(seq.copy(), dtype=np.int64)
            if length > 1:
                switch = random.sample(range(3), k=2)
            else:
                switch = [3, 3]

            if switch[0] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[0] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[0] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
            else:
                aug_seq, aug_len = seq.tolist(), length
            aug_seq1.append(aug_seq if aug_len > 0 else seq.tolist())

            if switch[1] == 0:
                aug_seq, aug_len = item_crop(seq, length)
            elif switch[1] == 1:
                aug_seq, aug_len = item_mask(seq, length)
            elif switch[1] == 2:
                aug_seq, aug_len = item_reorder(seq, length)
            else:
                aug_seq, aug_len = seq.tolist(), length
            aug_seq2.append(aug_seq if aug_len > 0 else seq.tolist())

        aug_seq1 = torch.tensor(aug_seq1, dtype=torch.long, device=batch_seqs.device)
        aug_seq2 = torch.tensor(aug_seq2, dtype=torch.long, device=batch_seqs.device)
        return aug_seq1, aug_seq2

    # ==================== CL4SRec contrastive learning ====================
    def mask_correlated_samples(self, batch_size):
        """Build a mask that removes self-pairs and positive pairs."""
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def cl4srec_loss(self, aug_final1, aug_final2):
        """Pure CL4SRec InfoNCE loss."""
        batch_size = aug_final1.shape[0]
        device = aug_final1.device

        aug1_norm = F.normalize(aug_final1, dim=1)
        aug2_norm = F.normalize(aug_final2, dim=1)

        N = 2 * batch_size
        z = torch.cat([aug1_norm, aug2_norm], dim=0)  # [2B, D]
        sim = torch.mm(z, z.T) / self.tau  # [2B, 2B]

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size).to(device)
        negative_samples = sim[mask].reshape(N, -1)

        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=device)

        cl_loss = self.cl_loss_func(logits, labels)
        return cl_loss

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_last_items = batch_data

        final_emb = self.forward(batch_seqs)

        # Recommendation loss
        test_item_emb = self.get_item_emb(torch.arange(self.item_num + 1, device=batch_seqs.device))
        logits = torch.matmul(final_emb, test_item_emb.T)
        rec_loss = self.loss_func(logits, batch_last_items)

        # Contrastive loss
        aug_seq1, aug_seq2 = self._cl4srec_aug(batch_seqs)
        aug_final1 = self.forward(aug_seq1)
        aug_final2 = self.forward(aug_seq2)
        cl_loss = self.cl4srec_loss(aug_final1, aug_final2)

        total_loss = rec_loss + self.cl_weight * cl_loss

        loss_dict = {
            'rec': rec_loss.item(),
            'cl': cl_loss.item(),
            'total': total_loss.item()
        }
        return total_loss, loss_dict

    def full_predict(self, batch_data):
        batch_user, batch_seqs, _ = batch_data
        final_emb = self.forward(batch_seqs)
        test_item_emb = self.get_item_emb(torch.arange(self.item_num + 1, device=batch_seqs.device))
        scores = torch.matmul(final_emb, test_item_emb.T)
        return scores
