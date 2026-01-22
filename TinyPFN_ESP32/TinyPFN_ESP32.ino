/*
 * TinyPFN - Pure C Implementation for ESP32
 * REDUCED VERSION FOR ESP32-WROOM (no PSRAM)
 * 
 * Configuration:
 * - 5 features (SelectKBest from breast cancer)
 * - 50 train + 50 test = 100 samples
 * - 2 classes
 * 
 * Memory estimate: ~69 KB buffers (fits in ESP32-WROOM)
 */

#include <Arduino.h>
#include <math.h>
#include "tinypfn_weights.h"

// ============================================================================
// Configuration - REDUCED FOR ESP32 MEMORY
// ============================================================================

#define NUM_FEATURES 5
#define NUM_TRAIN 50
#define NUM_TEST 50
#define NUM_ROWS (NUM_TRAIN + NUM_TEST)
#define MAX_COLS (NUM_FEATURES + 1)  // features + target column

// ============================================================================
// Helper Functions
// ============================================================================

void linear(const float* W, const float* b, const float* x, float* y, 
            int in_features, int out_features) {
  for (int i = 0; i < out_features; i++) {
    float sum = b[i];
    for (int j = 0; j < in_features; j++) {
      sum += W[i * in_features + j] * x[j];
    }
    y[i] = sum;
  }
}

void linear_batched(const float* W, const float* b, const float* input, float* output,
                    int batch, int in_features, int out_features) {
  for (int i = 0; i < batch; i++) {
    linear(W, b, &input[i * in_features], &output[i * out_features], 
           in_features, out_features);
  }
}

float gelu(float x) {
  return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

void gelu_inplace(float* x, int n) {
  for (int i = 0; i < n; i++) {
    x[i] = gelu(x[i]);
  }
}

void layer_norm(const float* gamma, const float* beta, float* x, int size) {
  float mean = 0.0f;
  for (int i = 0; i < size; i++) mean += x[i];
  mean /= size;
  
  float var = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = x[i] - mean;
    var += diff * diff;
  }
  var /= size;
  
  float inv_std = 1.0f / sqrtf(var + 1e-5f);
  for (int i = 0; i < size; i++) {
    x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
  }
}

/**
 * Feature map for linear attention: ReLU(x) + epsilon
 * Note: Scaling is applied separately before calling this function
 */
void feature_map(float* x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = (x[i] > 0.0f ? x[i] : 0.0f) + 1e-6f;
  }
}

// ============================================================================
// Linear Attention
// ============================================================================

static float attn_Q[NUM_ROWS * EMBEDDING_SIZE];
static float attn_K[NUM_ROWS * EMBEDDING_SIZE];
static float attn_V[NUM_ROWS * EMBEDDING_SIZE];
static float attn_KV[EMBEDDING_SIZE * EMBEDDING_SIZE];
static float attn_out[NUM_ROWS * EMBEDDING_SIZE];

/**
 * Linear Attention mechanism with O(n) complexity
 * 
 * Computes: Output = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))
 * where phi(x) = ReLU(x * scale) + epsilon
 * 
 * @param input_q Query input (seq_q, E)
 * @param input_k Key input (seq_k, E)
 * @param input_v Value input (seq_k, E)
 * @param output Output buffer (seq_q, E)
 * @param seq_q Number of query positions
 * @param seq_k Number of key/value positions
 * @param W_q, b_q Query projection weights and bias
 * @param W_k, b_k Key projection weights and bias
 * @param W_v, b_v Value projection weights and bias
 * @param W_o, b_o Output projection weights and bias
 */
void linear_attention(const float* input_q, const float* input_k, const float* input_v,
                      float* output, int seq_q, int seq_k,
                      const float* W_q, const float* b_q,
                      const float* W_k, const float* b_k,
                      const float* W_v, const float* b_v,
                      const float* W_o, const float* b_o) {
  
  // Project Q, K, V
  linear_batched(W_q, b_q, input_q, attn_Q, seq_q, EMBEDDING_SIZE, EMBEDDING_SIZE);
  linear_batched(W_k, b_k, input_k, attn_K, seq_k, EMBEDDING_SIZE, EMBEDDING_SIZE);
  linear_batched(W_v, b_v, input_v, attn_V, seq_k, EMBEDDING_SIZE, EMBEDDING_SIZE);
  
  // Apply scale to both Q and K before feature map
  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  for (int i = 0; i < seq_q * EMBEDDING_SIZE; i++) {
    attn_Q[i] *= scale;
  }
  for (int i = 0; i < seq_k * EMBEDDING_SIZE; i++) {
    attn_K[i] *= scale;
  }
  
  // Apply feature map: phi(x) = ReLU(x) + epsilon
  feature_map(attn_Q, seq_q * EMBEDDING_SIZE);
  feature_map(attn_K, seq_k * EMBEDDING_SIZE);
  
  // Compute KV accumulator: K^T @ V -> (E, E)
  // KV[i][j] = sum_n K[n][i] * V[n][j]
  for (int i = 0; i < EMBEDDING_SIZE; i++) {
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      float sum = 0.0f;
      for (int n = 0; n < seq_k; n++) {
        sum += attn_K[n * EMBEDDING_SIZE + i] * attn_V[n * EMBEDDING_SIZE + j];
      }
      attn_KV[i * EMBEDDING_SIZE + j] = sum;
    }
  }
  
  // Compute K_sum for normalization: sum_n K[n][d]
  float K_sum[EMBEDDING_SIZE] = {0};
  for (int n = 0; n < seq_k; n++) {
    for (int d = 0; d < EMBEDDING_SIZE; d++) {
      K_sum[d] += attn_K[n * EMBEDDING_SIZE + d];
    }
  }
  
  // Compute output for each query position
  for (int q = 0; q < seq_q; q++) {
    // Normalizer: Z = Q[q] @ K_sum
    float Z = 1e-8f;
    for (int d = 0; d < EMBEDDING_SIZE; d++) {
      Z += attn_Q[q * EMBEDDING_SIZE + d] * K_sum[d];
    }
    
    // Output: (Q[q] @ KV) / Z
    for (int d = 0; d < EMBEDDING_SIZE; d++) {
      float sum = 0.0f;
      for (int k = 0; k < EMBEDDING_SIZE; k++) {
        sum += attn_Q[q * EMBEDDING_SIZE + k] * attn_KV[k * EMBEDDING_SIZE + d];
      }
      attn_out[q * EMBEDDING_SIZE + d] = sum / Z;
    }
  }
  
  // Output projection
  linear_batched(W_o, b_o, attn_out, output, seq_q, EMBEDDING_SIZE, EMBEDDING_SIZE);
}

// ============================================================================
// TinyPFN Forward Pass
// ============================================================================

static float embeddings[NUM_ROWS * MAX_COLS * EMBEDDING_SIZE];
static float temp_buffer[NUM_ROWS * MAX_COLS * EMBEDDING_SIZE];
static float mlp_temp[NUM_ROWS * MAX_COLS * EMBEDDING_SIZE];

/**
 * TinyPFN forward pass for in-context learning
 * 
 * @param X Feature matrix (num_rows, num_features)
 * @param y_train Training labels (num_train,)
 * @param num_train Number of training samples
 * @param num_test Number of test samples
 * @param num_features Number of input features
 * @param output Output logits (num_test, NUM_OUTPUTS)
 */
void tinypfn_forward(const float* X, const float* y_train,
                     int num_train, int num_test, int num_features,
                     float* output) {
  
  int num_rows = num_train + num_test;
  int num_cols = num_features + 1;
  
  // =========================================================================
  // Feature Encoding
  // Normalize using training statistics only (no data leakage)
  // =========================================================================
  float feat_mean[NUM_FEATURES] = {0};
  float feat_std[NUM_FEATURES] = {0};
  
  // Compute mean over training data
  for (int f = 0; f < num_features; f++) {
    for (int i = 0; i < num_train; i++) {
      feat_mean[f] += X[i * num_features + f];
    }
    feat_mean[f] /= num_train;
  }
  
  // Compute std over training data
  for (int f = 0; f < num_features; f++) {
    for (int i = 0; i < num_train; i++) {
      float diff = X[i * num_features + f] - feat_mean[f];
      feat_std[f] += diff * diff;
    }
    feat_std[f] = sqrtf(feat_std[f] / num_train) + 1e-20f;
  }
  
  // Normalize all data and embed
  for (int row = 0; row < num_rows; row++) {
    for (int f = 0; f < num_features; f++) {
      float x_norm = (X[row * num_features + f] - feat_mean[f]) / feat_std[f];
      x_norm = fmaxf(-100.0f, fminf(100.0f, x_norm));
      
      int emb_idx = (row * num_cols + f) * EMBEDDING_SIZE;
      for (int e = 0; e < EMBEDDING_SIZE; e++) {
        embeddings[emb_idx + e] = feature_enc_weight[e] * x_norm + feature_enc_bias[e];
      }
    }
  }
  
  // =========================================================================
  // Target Encoding
  // Use mean padding for test samples
  // =========================================================================
  float y_mean = 0.0f;
  for (int i = 0; i < num_train; i++) {
    y_mean += y_train[i];
  }
  y_mean /= num_train;
  
  for (int row = 0; row < num_rows; row++) {
    float y_val = (row < num_train) ? y_train[row] : y_mean;
    
    int emb_idx = (row * num_cols + num_features) * EMBEDDING_SIZE;
    for (int e = 0; e < EMBEDDING_SIZE; e++) {
      embeddings[emb_idx + e] = target_enc_weight[e] * y_val + target_enc_bias[e];
    }
  }
  
  // =========================================================================
  // Transformer Block - Attention between features
  // Each row attends across its columns (features)
  // =========================================================================
  for (int row = 0; row < num_rows; row++) {
    float* row_emb = &embeddings[row * num_cols * EMBEDDING_SIZE];
    float* row_out = &temp_buffer[row * num_cols * EMBEDDING_SIZE];
    
    linear_attention(row_emb, row_emb, row_emb, row_out, num_cols, num_cols,
                     layer0_feat_q_weight, layer0_feat_q_bias,
                     layer0_feat_k_weight, layer0_feat_k_bias,
                     layer0_feat_v_weight, layer0_feat_v_bias,
                     layer0_feat_out_weight, layer0_feat_out_bias);
    
    // Residual connection
    for (int i = 0; i < num_cols * EMBEDDING_SIZE; i++) {
      row_emb[i] = row_out[i] + row_emb[i];
    }
    
    // Layer normalization
    for (int col = 0; col < num_cols; col++) {
      layer_norm(layer0_norm1_weight, layer0_norm1_bias,
                 &row_emb[col * EMBEDDING_SIZE], EMBEDDING_SIZE);
    }
  }
  
  // =========================================================================
  // Transformer Block - Attention between datapoints
  // Each column attends across its rows (datapoints)
  // Training data attends to itself, test data attends to training data
  // =========================================================================
  
  // Transpose: (row, col, E) -> (col, row, E)
  for (int col = 0; col < num_cols; col++) {
    for (int row = 0; row < num_rows; row++) {
      int src_idx = (row * num_cols + col) * EMBEDDING_SIZE;
      int dst_idx = (col * num_rows + row) * EMBEDDING_SIZE;
      for (int e = 0; e < EMBEDDING_SIZE; e++) {
        temp_buffer[dst_idx + e] = embeddings[src_idx + e];
      }
    }
  }
  
  for (int col = 0; col < num_cols; col++) {
    float* col_emb = &temp_buffer[col * num_rows * EMBEDDING_SIZE];
    float* col_out = &mlp_temp[col * num_rows * EMBEDDING_SIZE];
    
    // Train-to-train attention
    float* train_emb = col_emb;
    float train_out[NUM_ROWS * EMBEDDING_SIZE];
    
    linear_attention(train_emb, train_emb, train_emb, train_out, 
                     num_train, num_train,
                     layer0_dp_q_weight, layer0_dp_q_bias,
                     layer0_dp_k_weight, layer0_dp_k_bias,
                     layer0_dp_v_weight, layer0_dp_v_bias,
                     layer0_dp_out_weight, layer0_dp_out_bias);
    
    // Test-to-train attention (test queries attend to training keys/values)
    float* test_emb = &col_emb[num_train * EMBEDDING_SIZE];
    float test_out[NUM_ROWS * EMBEDDING_SIZE];
    
    linear_attention(test_emb, train_emb, train_emb, test_out,
                     num_test, num_train,
                     layer0_dp_q_weight, layer0_dp_q_bias,
                     layer0_dp_k_weight, layer0_dp_k_bias,
                     layer0_dp_v_weight, layer0_dp_v_bias,
                     layer0_dp_out_weight, layer0_dp_out_bias);
    
    // Residual connections
    for (int i = 0; i < num_train * EMBEDDING_SIZE; i++) {
      col_out[i] = train_out[i] + col_emb[i];
    }
    for (int i = 0; i < num_test * EMBEDDING_SIZE; i++) {
      col_out[num_train * EMBEDDING_SIZE + i] = test_out[i] + col_emb[num_train * EMBEDDING_SIZE + i];
    }
    
    // Copy back for layer norm
    for (int i = 0; i < num_rows * EMBEDDING_SIZE; i++) {
      col_emb[i] = col_out[i];
    }
    
    // Layer normalization
    for (int row = 0; row < num_rows; row++) {
      layer_norm(layer0_norm2_weight, layer0_norm2_bias,
                 &col_emb[row * EMBEDDING_SIZE], EMBEDDING_SIZE);
    }
  }
  
  // Transpose back: (col, row, E) -> (row, col, E)
  for (int col = 0; col < num_cols; col++) {
    for (int row = 0; row < num_rows; row++) {
      int src_idx = (col * num_rows + row) * EMBEDDING_SIZE;
      int dst_idx = (row * num_cols + col) * EMBEDDING_SIZE;
      for (int e = 0; e < EMBEDDING_SIZE; e++) {
        embeddings[dst_idx + e] = temp_buffer[src_idx + e];
      }
    }
  }
  
  // =========================================================================
  // Feed-Forward Network (MLP)
  // =========================================================================
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      int idx = (row * num_cols + col) * EMBEDDING_SIZE;
      float* emb = &embeddings[idx];
      float hidden[MLP_HIDDEN];
      
      // First linear layer
      linear(layer0_mlp1_weight, layer0_mlp1_bias, emb, hidden, 
             EMBEDDING_SIZE, MLP_HIDDEN);
      
      // GELU activation
      gelu_inplace(hidden, MLP_HIDDEN);
      
      // Second linear layer
      float mlp_out[EMBEDDING_SIZE];
      linear(layer0_mlp2_weight, layer0_mlp2_bias, hidden, mlp_out,
             MLP_HIDDEN, EMBEDDING_SIZE);
      
      // Residual connection
      for (int e = 0; e < EMBEDDING_SIZE; e++) {
        emb[e] = mlp_out[e] + emb[e];
      }
      
      // Layer normalization
      layer_norm(layer0_norm3_weight, layer0_norm3_bias, emb, EMBEDDING_SIZE);
    }
  }
  
  // =========================================================================
  // Decoder
  // Extract test target embeddings and decode to class logits
  // =========================================================================
  for (int t = 0; t < num_test; t++) {
    int row = num_train + t;
    int emb_idx = (row * num_cols + num_features) * EMBEDDING_SIZE;
    float* emb = &embeddings[emb_idx];
    
    float hidden[MLP_HIDDEN];
    
    // First linear layer
    linear(decoder_linear1_weight, decoder_linear1_bias, emb, hidden,
           EMBEDDING_SIZE, MLP_HIDDEN);
    
    // GELU activation
    gelu_inplace(hidden, MLP_HIDDEN);
    
    // Second linear layer (output logits)
    linear(decoder_linear2_weight, decoder_linear2_bias, hidden, &output[t * NUM_OUTPUTS],
           MLP_HIDDEN, NUM_OUTPUTS);
  }
}

// ============================================================================
// BENCHMARK DATA - 50 train + 50 test (reduced for ESP32 memory)
// Same 5 features as full benchmark
// ============================================================================

const float X_train[NUM_TRAIN * NUM_FEATURES] = {
    73.590000f, 0.015020f, 12.680000f, 82.690000f, 0.055090f,
    71.490000f, 0.022570f, 12.020000f, 77.800000f, 0.064130f,
    89.780000f, 0.029440f, 15.660000f, 101.200000f, 0.074530f,
    143.700000f, 0.160400f, 23.370000f, 170.300000f, 0.250800f,
    142.700000f, 0.149600f, 26.680000f, 176.500000f, 0.290300f,
    113.400000f, 0.088110f, 22.510000f, 141.200000f, 0.206600f,
    116.000000f, 0.140100f, 22.250000f, 152.400000f, 0.255000f,
    64.120000f, 0.070380f, 10.600000f, 69.470000f, 0.107500f,
    102.500000f, 0.109700f, 18.790000f, 125.000000f, 0.182700f,
    82.010000f, 0.001852f, 14.000000f, 88.180000f, 0.009259f,
    96.730000f, 0.073640f, 17.460000f, 124.100000f, 0.171200f,
    72.170000f, 0.021730f, 12.360000f, 79.290000f, 0.062030f,
    87.020000f, 0.051020f, 14.550000f, 99.480000f, 0.112600f,
    73.000000f, 0.047960f, 11.920000f, 76.530000f, 0.086110f,
    70.150000f, 0.059410f, 10.850000f, 76.510000f, 0.146500f,
    98.170000f, 0.031570f, 17.380000f, 113.700000f, 0.103500f,
    65.310000f, 0.005495f, 11.250000f, 71.120000f, 0.023810f,
    91.120000f, 0.064630f, 17.040000f, 113.900000f, 0.182700f,
    93.630000f, 0.055980f, 15.890000f, 116.200000f, 0.144700f,
    94.370000f, 0.066180f, 15.740000f, 106.400000f, 0.177200f,
    130.700000f, 0.097400f, 25.280000f, 159.800000f, 0.250700f,
    96.220000f, 0.037810f, 16.110000f, 104.600000f, 0.084850f,
    62.110000f, 0.007937f, 10.920000f, 68.810000f, 0.023810f,
    114.600000f, 0.041780f, 19.820000f, 127.100000f, 0.083410f,
    73.300000f, 0.063670f, 13.240000f, 91.760000f, 0.252400f,
    78.290000f, 0.015270f, 13.340000f, 88.830000f, 0.056900f,
    93.860000f, 0.064950f, 15.800000f, 103.100000f, 0.106900f,
    130.000000f, 0.094980f, 27.320000f, 186.800000f, 0.238800f,
    82.020000f, 0.028640f, 13.750000f, 89.040000f, 0.047730f,
    59.200000f, 0.021800f, 10.010000f, 65.590000f, 0.050870f,
    79.010000f, 0.019210f, 13.290000f, 85.560000f, 0.084420f,
    126.200000f, 0.096640f, 23.720000f, 159.800000f, 0.187200f,
    103.700000f, 0.083990f, 18.490000f, 126.300000f, 0.201400f,
    135.900000f, 0.150400f, 25.300000f, 171.100000f, 0.268500f,
    53.270000f, 0.021680f, 9.092000f, 58.080000f, 0.078790f,
    84.180000f, 0.018830f, 14.130000f, 96.310000f, 0.066080f,
    75.210000f, 0.021570f, 13.350000f, 87.000000f, 0.081200f,
    98.220000f, 0.063000f, 16.460000f, 114.100000f, 0.110800f,
    65.850000f, 0.024380f, 10.830000f, 71.080000f, 0.083330f,
    114.400000f, 0.075070f, 20.390000f, 137.900000f, 0.152800f,
    61.640000f, 0.022920f, 10.750000f, 71.250000f, 0.081200f,
    102.400000f, 0.091130f, 21.200000f, 142.100000f, 0.212100f,
    74.680000f, 0.032390f, 13.010000f, 84.420000f, 0.109900f,
    54.420000f, 0.000000f, 9.262000f, 58.360000f, 0.000000f,
    152.100000f, 0.100300f, 30.790000f, 211.500000f, 0.226400f,
    155.100000f, 0.141000f, 30.670000f, 202.400000f, 0.208900f,
    124.400000f, 0.119800f, 20.800000f, 149.600000f, 0.196400f,
    78.310000f, 0.020270f, 14.290000f, 93.850000f, 0.088290f,
    88.100000f, 0.053810f, 14.800000f, 97.330000f, 0.145300f,
    73.160000f, 0.055500f, 12.840000f, 84.930000f, 0.131800f
};

const float y_train_arr[NUM_TRAIN] = {
    1, 1, 1, 0, 0, 0, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
    1, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 0, 0, 1, 1, 1
};

const float X_test[NUM_TEST * NUM_FEATURES] = {
    73.810000f, 0.022330f, 13.110000f, 84.530000f, 0.061270f,
    106.900000f, 0.084880f, 19.280000f, 129.800000f, 0.158300f,
    73.280000f, 0.020690f, 12.970000f, 83.120000f, 0.065440f,
    108.100000f, 0.102800f, 20.960000f, 136.800000f, 0.207300f,
    65.670000f, 0.027380f, 10.840000f, 69.570000f, 0.091270f,
    94.660000f, 0.025410f, 15.610000f, 101.700000f, 0.079550f,
    137.800000f, 0.132200f, 24.300000f, 160.200000f, 0.214800f,
    82.690000f, 0.050740f, 14.380000f, 95.290000f, 0.140700f,
    81.250000f, 0.021070f, 13.710000f, 88.700000f, 0.056020f,
    69.140000f, 0.013690f, 11.370000f, 72.420000f, 0.031940f,
    94.490000f, 0.059800f, 18.330000f, 117.900000f, 0.183800f,
    69.280000f, 0.026420f, 11.880000f, 78.280000f, 0.079260f,
    91.560000f, 0.091760f, 19.200000f, 128.500000f, 0.201300f,
    97.030000f, 0.048190f, 16.250000f, 109.100000f, 0.148900f,
    111.800000f, 0.099340f, 20.990000f, 143.200000f, 0.182700f,
    97.400000f, 0.090290f, 17.310000f, 114.600000f, 0.161400f,
    182.100000f, 0.187800f, 33.120000f, 220.800000f, 0.268800f,
    61.060000f, 0.005769f, 11.150000f, 71.110000f, 0.025000f,
    64.600000f, 0.037160f, 11.260000f, 73.070000f, 0.099100f,
    90.430000f, 0.050690f, 16.570000f, 110.300000f, 0.138300f,
    68.010000f, 0.016150f, 12.250000f, 77.980000f, 0.061360f,
    77.870000f, 0.037910f, 13.240000f, 92.200000f, 0.115500f,
    93.600000f, 0.080250f, 15.030000f, 108.800000f, 0.220800f,
    140.100000f, 0.152000f, 25.740000f, 184.600000f, 0.265000f,
    94.250000f, 0.049380f, 16.210000f, 108.400000f, 0.122500f,
    80.640000f, 0.028800f, 14.180000f, 95.230000f, 0.098040f,
    81.780000f, 0.019240f, 13.500000f, 88.540000f, 0.063430f,
    137.200000f, 0.086320f, 29.170000f, 188.000000f, 0.200900f,
    94.150000f, 0.042230f, 16.670000f, 111.400000f, 0.141400f,
    98.000000f, 0.065530f, 18.510000f, 121.200000f, 0.152600f,
    60.110000f, 0.015040f, 10.410000f, 67.030000f, 0.065170f,
    79.470000f, 0.021660f, 13.160000f, 85.130000f, 0.080880f,
    87.210000f, 0.017230f, 14.670000f, 96.080000f, 0.085860f,
    63.780000f, 0.078570f, 11.020000f, 71.040000f, 0.157100f,
    73.720000f, 0.055880f, 11.860000f, 78.270000f, 0.093140f,
    82.630000f, 0.018670f, 14.400000f, 91.630000f, 0.056010f,
    95.540000f, 0.028990f, 14.990000f, 95.540000f, 0.028990f,
    118.400000f, 0.077620f, 21.530000f, 143.400000f, 0.148900f,
    77.610000f, 0.042740f, 12.980000f, 84.480000f, 0.120200f,
    104.100000f, 0.066380f, 19.770000f, 128.800000f, 0.152000f,
    186.900000f, 0.168900f, 36.040000f, 251.200000f, 0.262500f,
    69.500000f, 0.028670f, 11.690000f, 76.510000f, 0.086000f,
    71.800000f, 0.022300f, 12.330000f, 78.000000f, 0.069610f,
    124.800000f, 0.124400f, 23.150000f, 160.500000f, 0.184800f,
    129.700000f, 0.086910f, 25.730000f, 170.300000f, 0.182000f,
    54.660000f, 0.009259f, 9.565000f, 62.060000f, 0.027780f,
    188.500000f, 0.159500f, 28.110000f, 188.500000f, 0.159500f,
    87.320000f, 0.052660f, 14.830000f, 94.940000f, 0.133500f,
    73.930000f, 0.028540f, 12.580000f, 87.160000f, 0.121800f,
    123.600000f, 0.079510f, 24.860000f, 165.900000f, 0.178900f
};

const int y_test_true[NUM_TEST] = {
    1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 1, 1, 0,
    1, 1, 0, 0, 0, 1, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    0, 1, 1, 0, 0, 1, 0, 1, 1, 0
};

// Class distribution:
//   Train: 20 malignant, 30 benign
//   Test:  22 malignant, 28 benign

// ============================================================================
// Arduino Interface
// ============================================================================

float X_data[NUM_ROWS * NUM_FEATURES];

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n");
  Serial.println("╔══════════════════════════════════════════════════════╗");
  Serial.println("║     TinyPFN for ESP32-WROOM                          ║");
  Serial.println("║     5 features, 50 train + 50 test = 100 samples     ║");
  Serial.println("╚══════════════════════════════════════════════════════╝");
  Serial.println();
  
  Serial.printf("Model: embedding=%d, heads=%d, mlp=%d, layers=%d, outputs=%d\n",
                EMBEDDING_SIZE, NUM_HEADS, MLP_HIDDEN, NUM_LAYERS, NUM_OUTPUTS);
  Serial.printf("Free heap: %d bytes (%.1f KB)\n", ESP.getFreeHeap(), ESP.getFreeHeap()/1024.0f);
  Serial.println();
  
  runBenchmark();
}

void runBenchmark() {
  Serial.println("========================================================");
  Serial.println("Breast Cancer Wisconsin - Reduced for ESP32 Memory");
  Serial.println("Features: mean perimeter, mean concave points,");
  Serial.println("          worst radius, worst perimeter, worst concave points");
  Serial.println("========================================================\n");
  
  // Combine train and test data
  for (int i = 0; i < NUM_TRAIN * NUM_FEATURES; i++) {
    X_data[i] = X_train[i];
  }
  for (int i = 0; i < NUM_TEST * NUM_FEATURES; i++) {
    X_data[NUM_TRAIN * NUM_FEATURES + i] = X_test[i];
  }
  
  Serial.printf("Training samples: %d\n", NUM_TRAIN);
  Serial.printf("Test samples: %d\n", NUM_TEST);
  Serial.printf("Features: %d\n", NUM_FEATURES);
  Serial.println();
  
  float output[NUM_TEST * NUM_OUTPUTS];
  
  unsigned long start = micros();
  tinypfn_forward(X_data, y_train_arr, NUM_TRAIN, NUM_TEST, NUM_FEATURES, output);
  unsigned long duration = micros() - start;
  
  Serial.printf("Inference time: %lu us (%.2f ms)\n\n", duration, duration/1000.0f);
  
  Serial.println("Predictions:");
  Serial.println("------------------------------------------------------------");
  
  int correct = 0;
  int tp = 0, tn = 0, fp = 0, fn = 0;
  
  for (int t = 0; t < NUM_TEST; t++) {
    float* logits = &output[t * NUM_OUTPUTS];
    
    // Softmax with numerical stability
    float max_l = logits[0] > logits[1] ? logits[0] : logits[1];
    float probs[2];
    float sum = 0;
    for (int c = 0; c < 2; c++) {
      probs[c] = expf(logits[c] - max_l);
      sum += probs[c];
    }
    for (int c = 0; c < 2; c++) probs[c] /= sum;
    
    int pred = probs[1] > probs[0] ? 1 : 0;
    int actual = y_test_true[t];
    
    if (pred == actual) {
      correct++;
      if (pred == 1) tn++; else tp++;
    } else {
      if (pred == 1) fn++; else fp++;
    }
    
    // Print first 10 and last 5 predictions
    if (t < 10 || t >= NUM_TEST - 5) {
      Serial.printf("Test %2d: P(M)=%.3f P(B)=%.3f | True:%-6s Pred:%-6s [%s]\n",
                    t, probs[0], probs[1], 
                    actual == 0 ? "Malig" : "Benign",
                    pred == 0 ? "Malig" : "Benign",
                    pred == actual ? "OK" : "MISS");
    } else if (t == 10) {
      Serial.println("... (skipping middle results) ...");
    }
  }
  
  float accuracy = (float)correct / NUM_TEST;
  float sensitivity = tp > 0 ? (float)tp / (tp + fn) : 0;
  float specificity = tn > 0 ? (float)tn / (tn + fp) : 0;
  float balanced_acc = (sensitivity + specificity) / 2;
  
  Serial.println();
  Serial.println("============================================================");
  Serial.println("                      RESULTS SUMMARY");
  Serial.println("============================================================");
  Serial.printf("Accuracy:          %d/%d = %.2f%%\n", correct, NUM_TEST, accuracy * 100.0f);
  Serial.printf("Balanced Accuracy: %.2f%%\n", balanced_acc * 100.0f);
  Serial.println("------------------------------------------------------------");
  Serial.printf("True Positives:  %d (Malignant correctly identified)\n", tp);
  Serial.printf("True Negatives:  %d (Benign correctly identified)\n", tn);
  Serial.printf("False Positives: %d (Benign misclassified as Malignant)\n", fp);
  Serial.printf("False Negatives: %d (Malignant misclassified as Benign)\n", fn);
  Serial.println("============================================================");
  Serial.println();
  
  // Memory report
  Serial.println("MEMORY USAGE:");
  int weights_mem = 890 * 4;  // 890 params * 4 bytes (FP32)
  int emb_size = sizeof(embeddings);
  int temp_size = sizeof(temp_buffer);
  int mlp_size = sizeof(mlp_temp);
  int attn_size = sizeof(attn_Q) + sizeof(attn_K) + sizeof(attn_V) + sizeof(attn_KV) + sizeof(attn_out);
  int total_act = emb_size + temp_size + mlp_size + attn_size;
  
  Serial.printf("  Weights:     %d bytes (%.2f KB)\n", weights_mem, weights_mem/1024.0f);
  Serial.printf("  Activations: %d bytes (%.2f KB)\n", total_act, total_act/1024.0f);
  Serial.printf("  TOTAL:       %d bytes (%.2f KB)\n", weights_mem + total_act, (weights_mem + total_act)/1024.0f);
  Serial.printf("  Free heap:   %d bytes (%.2f KB)\n", ESP.getFreeHeap(), ESP.getFreeHeap()/1024.0f);
  Serial.println();
  
  Serial.println("Commands: 'run' to repeat, 'info' for memory details");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toLowerCase();
    
    if (cmd == "run") {
      runBenchmark();
    } else if (cmd == "info") {
      Serial.println("\n=== Memory Details ===");
      Serial.printf("embeddings: %d bytes\n", sizeof(embeddings));
      Serial.printf("temp_buffer: %d bytes\n", sizeof(temp_buffer));
      Serial.printf("mlp_temp: %d bytes\n", sizeof(mlp_temp));
      Serial.printf("attention buffers: %d bytes\n", 
                    sizeof(attn_Q) + sizeof(attn_K) + sizeof(attn_V) + sizeof(attn_KV) + sizeof(attn_out));
      Serial.printf("Free heap: %d bytes\n\n", ESP.getFreeHeap());
    }
  }
  delay(10);
}