"""
Build a MoCo model with: a query encoder, a key encoder, and a queue
https://arxiv.org/abs/1911.05722
"""
import torch
import torch.nn as nn


class MoCo(nn.Module):

    def __init__(self, base_encoder,
                 dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.modelname = self._get_name()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()#num_classes=dim)
        self.encoder_k = base_encoder()#num_classes=dim)

        if isinstance(self.encoder_q.decoder, nn.Sequential):
            dim_mlp = self.encoder_q.decoder[0].weight.shape[1]
        else:
            dim_mlp = self.encoder_q.decoder.weight.shape[1]

        if mlp:  # hack: brute-force replacement
            self.modelname += 'V2'
            self.encoder_q.decoder = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_k.decoder = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        else:
            self.encoder_q.decoder = nn.Linear(dim_mlp, dim)
            self.encoder_k.decoder = nn.Linear(dim_mlp, dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size = x[0].shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return [x_[idx_shuffle] for x_ in x], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """

        return x[idx_unshuffle]

    def forward(self, data_q, data_k, use_doy=False):
        """
        Input:
            data_k: a batch of query time series
            data_k: a batch of key  time series
        Output:
            logits, targets
        """

        # compute query features
        if use_doy:
            q = self.encoder_q(data_q, use_doy=True)  # queries: NxC
        else:
            q = self.encoder_q(data_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            data_k, idx_unshuffle = self._batch_shuffle_ddp(data_k)

            if use_doy:
                k = self.encoder_k(data_k, use_doy=True)  # keys: NxC
            else:
                k = self.encoder_k(data_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

