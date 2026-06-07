import torch.utils.data as data
from config.configurator import configs
import torch


class SequentialDataset(data.Dataset):
    def __init__(self, user_seqs, mode='train', user_seqs_aug=None):
        self.mode = mode
        self.max_seq_len = configs['model']['max_seq_len']
        self.user_history_lists = {user: seq for user, seq in zip(user_seqs["uid"], user_seqs["item_seq"])}
        
        if user_seqs_aug is not None:
            self.uids = user_seqs_aug["uid"]
            self.seqs = user_seqs_aug["item_seq"]
            self.last_items = user_seqs_aug["item_id"]
        else:
            self.uids = user_seqs["uid"]
            self.seqs = user_seqs["item_seq"]
            self.last_items = user_seqs["item_id"]

        if mode == 'test':
            self.test_users = self.uids
            self.user_pos_lists = {uid: [item] for uid, item in zip(self.uids, self.last_items)}

    def _pad_seq(self, seq):
        if len(seq) >= self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        else:
            # pad at the head
            seq = [0] * (self.max_seq_len - len(seq)) + seq
        return seq

    def sample_negs(self):
        # Placeholder for negative sampling if needed
        pass

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        seq_i = self.seqs[idx]
        return self.uids[idx], torch.LongTensor(self._pad_seq(seq_i)), self.last_items[idx]
