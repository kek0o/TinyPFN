import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import Linear, LayerNorm
import math


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism with O(n) complexity instead of O(n²).
    Based on "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    Uses feature map approximation: softmax(QK^T)V ≈ φ(Q)(φ(K)^TV)
    
    Supports cached computation for efficient train/test split attention,
    where test queries attend to training keys/values without recomputation.
    """
    def __init__(self, embedding_size: int, nhead: int, feature_map_dim: int = 256, 
                 batch_first: bool = True, device=None, dtype=None):
        super().__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.head_dim = embedding_size // nhead
        self.feature_map_dim = feature_map_dim
        self.batch_first = batch_first
        
        assert embedding_size % nhead == 0, "embedding_size must be divisible by nhead"
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(embedding_size, embedding_size, device=device, dtype=dtype)
        self.k_proj = Linear(embedding_size, embedding_size, device=device, dtype=dtype)
        self.v_proj = Linear(embedding_size, embedding_size, device=device, dtype=dtype)
        self.out_proj = Linear(embedding_size, embedding_size, device=device, dtype=dtype)
        
        # Precompute scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature map using ReLU for stable kernel approximation.
        Applied after scaling for better numerical stability.
        
        Args:
            x: Input tensor (already scaled)
            
        Returns:
            Transformed tensor with same shape
        """
        return F.relu(x) + 1e-6
    
    def _project_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project inputs to Q, K, V and reshape for multi-head attention.
        
        Args:
            query: (B, T_q, E)
            key: (B, T_k, E)
            value: (B, T_k, E)
            
        Returns:
            Q, K, V each of shape (B, H, T, D) with feature map applied
        """
        B, T_q, E = query.shape
        _, T_k, _ = key.shape
        
        # Project
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape: (B, T, E) -> (B, H, T, D)
        Q = Q.view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scale and apply feature map
        Q = self.feature_map(Q * self.scale)
        K = self.feature_map(K * self.scale)
        
        return Q, K, V
    
    def _project_q_only(self, query: torch.Tensor) -> torch.Tensor:
        """
        Project only the query (for cached forward pass).
        
        Args:
            query: (B, T_q, E)
            
        Returns:
            Q of shape (B, H, T_q, D) with feature map applied
        """
        B, T_q, E = query.shape
        
        Q = self.q_proj(query)
        Q = Q.view(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        Q = self.feature_map(Q * self.scale)
        
        return Q
    
    def _compute_output(self, Q: torch.Tensor, KV: torch.Tensor, K_sum: torch.Tensor
                        ) -> torch.Tensor:
        """
        Compute attention output from Q and cached KV accumulator.
        
        Args:
            Q: Query tensor (B, H, T_q, D)
            KV: Cached K^T V accumulator (B, H, D, D)
            K_sum: Cached sum of K (B, H, D)
            
        Returns:
            Output tensor (B, T_q, E)
        """
        B = Q.shape[0]
        T_q = Q.shape[2]
        
        # Normalizer: Z = Q @ K_sum
        Z = torch.einsum('bhnd,bhd->bhn', Q, K_sum) + 1e-8
        
        # Output: Q @ KV
        out = torch.einsum('bhnd,bhde->bhne', Q, KV)
        
        # Normalize
        out = out / Z.unsqueeze(-1)
        
        # Reshape: (B, H, T_q, D) -> (B, T_q, E)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.embedding_size)
        
        # Final projection
        out = self.out_proj(out)
        
        return out
    
    def forward_with_cache(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns cached KV accumulator and K sum for reuse.
        Used for train-to-train attention where K, V will be reused by test queries.
        
        Computational cost:
            - Q, K, V projections: 6 * B * T * E^2
            - Feature maps: 6 * B * T * E
            - KV accumulator: 2 * B * T * E^2 / H
            - K sum: B * T * E (negligible)
            - Q @ KV: 2 * B * T * E^2 / H
            - Normalization: 4 * B * T * E
            - Output projection: 2 * B * T * E^2
        
        Args:
            query: (B, T, E) if batch_first else (T, B, E)
            key: (B, T, E) if batch_first else (T, B, E)
            value: (B, T, E) if batch_first else (T, B, E)
            
        Returns:
            output: Attention output (B, T, E)
            KV_cache: Accumulated K^T V matrix (B, H, D, D) for reuse
            K_sum_cache: Sum of K vectors (B, H, D) for normalization reuse
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project and apply feature maps
        Q, K, V = self._project_qkv(query, key, value)
        
        # Compute KV accumulator: K^T @ V -> (B, H, D, D)
        KV = torch.einsum('bhnd,bhne->bhde', K, V)
        
        # Compute K sum for normalization: (B, H, D)
        K_sum = K.sum(dim=2)
        
        # Compute output
        out = self._compute_output(Q, KV, K_sum)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
        
        return out, KV, K_sum
    
    def forward_from_cache(self, query: torch.Tensor, 
                           kv_cache: torch.Tensor, 
                           k_sum_cache: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using pre-computed KV accumulator and K sum.
        Used for test-to-train attention where test queries attend to cached training K, V.
        
        Computational cost (significant savings vs full forward):
            - Q projection only: 2 * B * T_test * E^2
            - Feature map (Q only): 3 * B * T_test * E
            - Q @ KV (reuses cache): 2 * B * T_test * E^2 / H
            - Normalization (reuses K_sum): 3 * B * T_test * E
            - Output projection: 2 * B * T_test * E^2
            
        Savings: Avoids recomputing K, V projections (4 * B * T_train * E^2),
                 K feature map (3 * B * T_train * E), and KV accumulator (2 * B * T_train * E^2 / H)
        
        Args:
            query: Test queries (B, T_test, E) if batch_first
            kv_cache: Pre-computed K^T V from training data (B, H, D, D)
            k_sum_cache: Pre-computed sum of K from training data (B, H, D)
            
        Returns:
            output: Attention output (B, T_test, E)
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
        
        # Project only Q (K, V are cached)
        Q = self._project_q_only(query)
        
        # Compute output using cached KV and K_sum
        out = self._compute_output(Q, kv_cache, k_sum_cache)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
        
        return out
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: torch.Tensor = None) -> tuple[torch.Tensor, None]:
        """
        Standard forward pass (maintains compatibility with nn.MultiheadAttention interface).
        
        For optimized train/test split computation, use forward_with_cache() 
        and forward_from_cache() instead.
        
        Args:
            query: (B, T_q, E) if batch_first else (T_q, B, E)
            key: (B, T_k, E) if batch_first else (T_k, B, E)
            value: (B, T_k, E) if batch_first else (T_k, B, E)
            attn_mask: Not used in linear attention (kept for interface compatibility)
            
        Returns:
            output: (B, T_q, E) if batch_first else (T_q, B, E)
            None: Linear attention doesn't return attention weights
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project and apply feature maps
        Q, K, V = self._project_qkv(query, key, value)
        
        # Compute KV accumulator
        KV = torch.einsum('bhnd,bhne->bhde', K, V)
        
        # Compute K sum
        K_sum = K.sum(dim=2)
        
        # Compute output
        out = self._compute_output(Q, KV, K_sum)
        
        if not self.batch_first:
            out = out.transpose(0, 1)
        
        return out, None


class TinyPFNModel(nn.Module):
    def __init__(self, embedding_size: int, num_attention_heads: int, mlp_hidden_size: int, 
                 num_layers: int, num_outputs: int, use_linear_attention: bool = True):
        """
        TinyPFN: A lightweight Prior-Fitted Network for tabular classification on edge devices.
        
        Args:
            embedding_size: Dimension of embeddings (E)
            num_attention_heads: Number of attention heads (H)
            mlp_hidden_size: Hidden dimension of MLP blocks (M)
            num_layers: Number of transformer layers (L)
            num_outputs: Maximum number of output classes
            use_linear_attention: Use O(n) linear attention vs O(n²) standard attention
        """
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.use_linear_attention = use_linear_attention
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                TransformerEncoderLayer(
                    embedding_size, 
                    num_attention_heads, 
                    mlp_hidden_size,
                    use_linear_attention=use_linear_attention
                )
            )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)
    
    def forward(self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int) -> torch.Tensor:
        """
        Forward pass for in-context learning.
        
        Args:
            src: Tuple of (X, y_train) where:
                - X: Features for all samples (B, R, C)
                - y_train: Labels for training samples only (B, R_train)
            train_test_split_index: Number of training samples (R_train)
            
        Returns:
            Logits for test samples (B, R_test, num_outputs)
        """
        x_src, y_src = src
        
        # Ensure y has correct shape (B, R_train, 1)
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        
        # Encode features: (B, R, C) -> (B, R, C, E)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        
        # Encode targets with mean-padding: (B, R_train, 1) -> (B, R, 1, E)
        y_src = self.target_encoder(y_src, num_rows)
        
        # Concatenate to form full table: (B, R, C+1, E)
        src = torch.cat([x_src, y_src], 2)
        
        # Apply transformer layers
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        
        # Extract test target embeddings: (B, R_test, E)
        output = src[:, train_test_split_index:, -1, :]
        
        # Decode to class logits: (B, R_test, num_outputs)
        output = self.decoder(output)
        return output


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """
        Encodes scalar features into embedding vectors.
        
        Args:
            embedding_size: Output embedding dimension (E)
        """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
    
    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Normalize features using training statistics and embed.
        
        Args:
            x: Input features (B, R, C)
            train_test_split_index: Number of training samples for normalization
            
        Returns:
            Embedded features (B, R, C, E)
        """
        x = x.unsqueeze(-1)
        
        # Compute statistics from training data only (no data leakage)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        
        # Normalize all data using training statistics
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        """
        Encodes target labels into embedding vectors with mean-padding for test samples.
        
        Args:
            embedding_size: Output embedding dimension (E)
        """
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
    
    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        """
        Pad training labels to full length using mean, then embed.
        
        Args:
            y_train: Training labels (B, R_train, 1)
            num_rows: Total number of rows (R = R_train + R_test)
            
        Returns:
            Embedded targets (B, R, 1, E)
        """
        # Pad test positions with mean of training labels
        mean = torch.mean(y_train, dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer layer with separate attention over features and datapoints.
    Supports both standard O(n²) and linear O(n) attention mechanisms.
    
    For linear attention, uses optimized caching to avoid redundant computation
    when test queries attend to training keys/values.
    """
    def __init__(self, embedding_size: int, nhead: int, mlp_hidden_size: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 use_linear_attention: bool = True, device=None, dtype=None):
        super().__init__()
        self.use_linear_attention = use_linear_attention
        
        if use_linear_attention:
            self.self_attention_between_datapoints = LinearAttention(
                embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
            )
            self.self_attention_between_features = LinearAttention(
                embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
            )
        else:
            from torch.nn.modules.transformer import MultiheadAttention
            self.self_attention_between_datapoints = MultiheadAttention(
                embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
            )
            self.self_attention_between_features = MultiheadAttention(
                embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
            )
        
        self.linear1 = Linear(embedding_size, mlp_hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(mlp_hidden_size, embedding_size, device=device, dtype=dtype)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm3 = LayerNorm(embedding_size, eps=layer_norm_eps, device=device, dtype=dtype)
    
    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        """
        Apply self-attention between features, then between datapoints, then MLP.
        
        Args:
            src: Input embeddings (B, R, C, E)
            train_test_split_index: Number of training samples (R_train)
            
        Returns:
            Transformed embeddings (B, R, C, E)
        """
        batch_size, rows_size, col_size, embedding_size = src.shape
        
        # ============================================================
        # ATTENTION BETWEEN FEATURES
        # Each datapoint attends across its features
        # Reshape: (B, R, C, E) -> (B*R, C, E)
        # ============================================================
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        
        # ============================================================
        # ATTENTION BETWEEN DATAPOINTS
        # Each feature column attends across datapoints
        # Reshape: (B, R, C, E) -> (B*C, R, E)
        # ============================================================
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        
        src_train = src[:, :train_test_split_index]
        src_test = src[:, train_test_split_index:]
        
        if self.use_linear_attention:
            # OPTIMIZED: Compute train attention and cache KV accumulator
            # Train attends to train, caching K^T V and sum(K)
            src_left, kv_cache, k_sum_cache = self.self_attention_between_datapoints.forward_with_cache(
                src_train, src_train, src_train
            )
            
            # Test attends to train using cached values (avoids recomputing K, V projections)
            src_right = self.self_attention_between_datapoints.forward_from_cache(
                src_test, kv_cache, k_sum_cache
            )
        else:
            # Standard attention: no caching optimization
            src_left = self.self_attention_between_datapoints(
                src_train, src_train, src_train
            )[0]
            src_right = self.self_attention_between_datapoints(
                src_test, src_train, src_train
            )[0]
        
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        
        # ============================================================
        # FEEDFORWARD MLP
        # ============================================================
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        """
        Decodes target embeddings to class logits.
        
        Args:
            embedding_size: Input embedding dimension (E)
            mlp_hidden_size: Hidden layer dimension (M)
            num_outputs: Number of output classes
        """
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP to get class logits.
        
        Args:
            x: Target embeddings (B, R_test, E)
            
        Returns:
            Class logits (B, R_test, num_outputs)
        """
        return self.linear2(F.gelu(self.linear1(x)))


class TinyPFNClassifier:
    """
    Scikit-learn compatible interface for TinyPFN.
    
    Usage:
        classifier = TinyPFNClassifier(model, device)
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
    """
    def __init__(self, model: TinyPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.X_train = None
        self.y_train = None
        self.num_classes = None
    
    def fit(self, X_train: np.array, y_train: np.array):
        """
        Store training data for in-context learning (no actual training occurs).
        
        Args:
            X_train: Training features (n_train, n_features)
            y_train: Training labels (n_train,)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1
    
    def predict_proba(self, X_test: np.array) -> np.array:
        """
        Predict class probabilities for test samples.
        
        Args:
            X_test: Test features (n_test, n_features)
            
        Returns:
            Class probabilities (n_test, n_classes)
        """
        # Concatenate train and test features
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        
        with torch.no_grad():
            # Add batch dimension and convert to tensors
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            
            # Forward pass
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(0)
            
            # Truncate to actual number of classes in dataset
            out = out[:, :self.num_classes]
            
            # Convert logits to probabilities
            probabilities = F.softmax(out, dim=1)
            
            return probabilities.to("cpu").numpy()
    
    def predict(self, X_test: np.array) -> np.array:
        """
        Predict class labels for test samples.
        
        Args:
            X_test: Test features (n_test, n_features)
            
        Returns:
            Predicted class labels (n_test,)
        """
        predicted_probabilities = self.predict_proba(X_test)
        return predicted_probabilities.argmax(axis=1)