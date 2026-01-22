"""Utility classes for SCM-based prior generation."""

from __future__ import annotations

import random
import numpy as np

import torch
from torch import nn


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to the input tensor."""
    
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, X):
        return X + torch.normal(torch.zeros_like(X), self.std)


class XSampler:
    """Input sampler for generating features for prior datasets."""

    def __init__(self, seq_len, num_features, pre_stats=False, sampling="mixed", device="cpu"):
        self.seq_len = seq_len
        self.num_features = num_features
        self.pre_stats = pre_stats
        self.sampling = sampling
        self.device = device

        if pre_stats:
            self._pre_stats()

    def _pre_stats(self):
        means = np.random.normal(0, 1, self.num_features)
        stds = np.abs(np.random.normal(0, 1, self.num_features) * means)
        self.means = torch.tensor(means, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)
        self.stds = torch.tensor(stds, dtype=torch.float, device=self.device).unsqueeze(0).repeat(self.seq_len, 1)

    def sample(self, return_numpy=False):
        samplers = {"normal": self.sample_normal_all, "mixed": self.sample_mixed, "uniform": self.sample_uniform}
        if self.sampling not in samplers:
            raise ValueError(f"Invalid sampling method: {self.sampling}")
        X = samplers[self.sampling]()
        return X.cpu().numpy() if return_numpy else X

    def sample_normal_all(self):
        if self.pre_stats:
            X = torch.normal(self.means, self.stds.abs()).float()
        else:
            X = torch.normal(0.0, 1.0, (self.seq_len, self.num_features), device=self.device).float()
        return X

    def sample_uniform(self):
        return torch.rand((self.seq_len, self.num_features), device=self.device)

    def sample_normal(self, n=None):
        if self.pre_stats:
            return torch.normal(self.means[:, n], self.stds[:, n].abs()).float()
        else:
            return torch.normal(0.0, 1.0, (self.seq_len,), device=self.device).float()

    def sample_multinomial(self):
        n_categories = random.randint(2, 20)
        probs = torch.rand(n_categories, device=self.device)
        x = torch.multinomial(probs, self.seq_len, replacement=True)
        x = x.float()
        return (x - x.mean()) / x.std()

    def sample_zipf(self):
        x = np.random.zipf(2.0 + random.random() * 2, (self.seq_len,))
        x = torch.tensor(x, device=self.device).clamp(max=10)
        x = x.float()
        return x - x.mean()

    def sample_mixed(self):
        X = []
        zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
        for n in range(self.num_features):
            if random.random() > normal_p:
                x = self.sample_normal(n)
            elif random.random() > multi_p:
                x = self.sample_multinomial()
            elif random.random() > zipf_p:
                x = self.sample_zipf()
            else:
                x = torch.rand((self.seq_len,), device=self.device)
            X.append(x)
        return torch.stack(X, -1)
