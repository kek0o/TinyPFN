import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import MultiheadAttention, Linear, LayerNorm

# -------------------------
# Small Adapter (bottleneck)
# -------------------------
class Adapter(nn.Module):
    def __init__(self, embedding_size: int, bottleneck: int = 32):
        super().__init__()
        self.down = nn.Linear(embedding_size, bottleneck)
        self.up = nn.Linear(bottleneck, embedding_size)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x):
        # x: (..., E)
        z = F.relu(self.down(x))
        return x + self.up(z)

# -------------------------
# Prompt pool (learnable)
# -------------------------
class PromptPool(nn.Module):
    def __init__(self, num_prompts: int, prompt_length: int, embedding_size: int):
        super().__init__()
        # prompts shape: (num_prompts, prompt_length, embedding_size)
        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_length, embedding_size) * 0.02)
        self.num_prompts = num_prompts
        self.prompt_length = prompt_length

    def get_prompts(self, indices=None, device=None, batch_size: int = 1):
        # return (batch_size, L_total, 1, E) to match x_src shape (B,R,C,E) where C includes feature columns
        if indices is None:
            p = self.prompts  # (P, L, E)
        else:
            p = self.prompts[indices]  # (k, L, E)
        p = p.reshape(-1, p.shape[-1])  # (L_total, E)
        # build (B, L_total, 1, E)
        if device is None:
            device = p.device
        p = p.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1).to(device)
        return p  # (B, L_total, 1, E)

# -------------------------
# Core TinyPFN Model
# -------------------------
class TinyPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int,
                 num_layers: int, num_outputs: int,
                 use_adapters: bool = True, adapter_bottleneck: int = 32,
                 use_prompts: bool = False, prompt_config: dict = None):
        """ Initializes the feature/target encoder, transformer stack and decoder """
        super().__init__()
        self.embedding_size = embedding_size
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList()
        self.use_adapters = use_adapters
        self.adapters = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(embedding_size, num_attention_heads, mlp_hidden_size)
            )
            if use_adapters:
                self.adapters.append(Adapter(embedding_size, bottleneck=adapter_bottleneck))
            else:
                self.adapters.append(nn.Identity())
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
        self.use_prompts = use_prompts
        if use_prompts:
            assert prompt_config is not None
            self.prompt_pool = PromptPool(
                prompt_config.get("num_prompts", 4),
                prompt_config.get("prompt_length", 1),
                embedding_size
            )
        else:
            self.prompt_pool = None

        # small hook stats (for monitoring)
        self._last_act_norm = None

    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int,
                prompt_indices=None) -> torch.Tensor:
        x_src, y_src = src
        #labels are expected to look like (batches, num_train_datapoints, 1)
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        # embed features
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]

        # optionally prepend prompts (they act as extra pseudo-rows)
        if self.use_prompts and (self.prompt_pool is not None):
            batch_size = x_src.shape[0]
            p = self.prompt_pool.get_prompts(indices=prompt_indices, device=x_src.device, batch_size=batch_size)
            # p shape => (B, L_total, 1, E)
            x_src = torch.cat([p, x_src], dim=1)
            # adjust split index (prompts are considered before train rows)
            train_test_split_index = train_test_split_index + p.shape[1]

        # pad & embed targets
        y_src = self.target_encoder(y_src, num_rows if not self.use_prompts else x_src.shape[1])
        src = torch.cat([x_src, y_src], 2)  # (B, R, C, E)

        # pass through transformer blocks + adapters
        for i, block in enumerate(self.transformer_blocks):
            src = block(src, train_test_split_index=train_test_split_index)
            if self.use_adapters:
                # apply adapter on last dim
                B, R, C, E = src.shape
                src = src.view(-1, E)
                src = self.adapters[i](src)
                src = src.view(B, R, C, E)

        # simple monitoring hook: store last activation norm
        self._last_act_norm = src.norm(dim=-1).mean().item()

        # select target embeddings and decode
        output = src[:, train_test_split_index:, -1, :]  # (B, num_targets, E)
        output = self.decoder(output)  # (B, num_targets, num_outputs)
        return output

    # -------------- convenience methods for CL --------------
    def freeze_base(self, freeze_feature_encoder=True, freeze_transformer=True, freeze_decoder=False):
        if freeze_feature_encoder:
            for p in self.feature_encoder.parameters():
                p.requires_grad = False
        if freeze_transformer:
            for block in self.transformer_blocks:
                for p in block.parameters():
                    p.requires_grad = False
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False
        # adapters and prompts stay trainable by default unless explicitly frozen

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def get_trainable_param_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def last_activation_norm(self):
        return self._last_act_norm

    def extract_feature_embeddings(self, x_numpy: np.ndarray, device: torch.device):
        """Return encoder embeddings (no prompts) for a numpy X (shape: [N_rows, num_features])"""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(x_numpy).unsqueeze(0).to(torch.float).to(device)
            # dummy small y (shape 1) to let encoder compute stats
            y_dummy = torch.zeros((1, 1)).to(device)
            e = self.feature_encoder(x, train_test_split_index=1)  # (1, R, C, E)
            return e.cpu().numpy()  # (1, R, C, E)

    def add_prompt_vectors(self, new_prompts: torch.Tensor):
        if self.prompt_pool is None:
            raise RuntimeError("Prompts not enabled")
        with torch.no_grad():
            # new_prompts shape (k, L, E)
            p = self.prompt_pool.prompts.data
            self.prompt_pool.prompts.data = torch.cat([p, new_prompts.to(p.device)], dim=0)

# -------------------------
# Encoders
# -------------------------
class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        # normalize features based on the mean and standard deviation
        x = (x - mean) / std
        # clip to avoid data spikes
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)

class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train, dim=1, keepdim=True)
        # pad y_train using the mean to the length of y as the model doesn't see the complete y
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)
        self.self_attention_between_features = MultiheadAttention(embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype)

        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)

        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        # attention between features
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        # attention between datapoints
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        # as seen on the original paper, training data attends to itself, and test data also to the training data
        src_left = self.self_attention_between_datapoints(src[:, :train_test_split_index], src[:, :train_test_split_index], src[:, :train_test_split_index])[0]
        src_right = self.self_attention_between_datapoints(src[:, train_test_split_index:], src[:, :train_test_split_index], src[:, :train_test_split_index])[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        # Simple 2 layer MLP
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src

class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))

# -------------------------
# Classifier wrapper (scikit-learn fashion)
# -------------------------
class TinyPFNClassifier():
    def __init__(self, model: TinyPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.array, y_train: np.array):
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = int(max(set(y_train)) + 1)

    def predict_proba(self, X_test: np.array) -> np.array:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(0)
            # cut out classes not seen in the dataset
            out = out[:, :self.num_classes]
            # softmax to get a probability distribution
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.array) -> np.array:
        return self.predict_proba(X_test).argmax(axis=1)
