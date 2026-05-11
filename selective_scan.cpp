#include <torch/extension.h>
#include <vector>
#include <cmath>

// Forward pass for Selective Scan
// x: (batch, seq_len, d_model)
// dt: (batch, seq_len, d_model)
// B_td: (batch, seq_len, d_state)
// C_td: (batch, seq_len, d_state)
// A: (d_model, d_state)
// D: (d_model)

torch::Tensor selective_scan_forward(
    torch::Tensor x,
    torch::Tensor dt,
    torch::Tensor B_td,
    torch::Tensor C_td,
    torch::Tensor A,
    torch::Tensor D) {
    
    // Ensure all tensors are contiguous float32 on CPU
    x = x.contiguous();
    dt = dt.contiguous();
    B_td = B_td.contiguous();
    C_td = C_td.contiguous();
    A = A.contiguous();
    D = D.contiguous();

    auto batch = x.size(0);
    auto seq_len = x.size(1);
    auto d_model = x.size(2);
    auto d_state = A.size(1);
    
    auto out = torch::empty_like(x);
    auto h = torch::zeros({batch, d_model, d_state}, x.options());

    // Raw pointers for fast access
    float* x_ptr = x.data_ptr<float>();
    float* dt_ptr = dt.data_ptr<float>();
    float* B_ptr = B_td.data_ptr<float>();
    float* C_ptr = C_td.data_ptr<float>();
    float* A_ptr = A.data_ptr<float>();
    float* D_ptr = D.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    float* h_ptr = h.data_ptr<float>();

    // Parallelize over batches and channels (d_model)
    #pragma omp parallel for collapse(2)
    for (long b = 0; b < batch; ++b) {
        for (long d = 0; d < d_model; ++d) {
            float D_val = D_ptr[d];
            
            // Loop over sequence elements sequentially for this batch and channel
            for (long t = 0; t < seq_len; ++t) {
                float x_td = x_ptr[b * seq_len * d_model + t * d_model + d];
                float delta_td = dt_ptr[b * seq_len * d_model + t * d_model + d];
                float yt_d = 0.0f;

                for (long s = 0; s < d_state; ++s) {
                    float b_ts = B_ptr[b * seq_len * d_state + t * d_state + s];
                    float c_ts = C_ptr[b * seq_len * d_state + t * d_state + s];
                    float a_ds = A_ptr[d * d_state + s];
                    
                    float dA_ds = std::exp(delta_td * a_ds);
                    float dB_ds = delta_td * b_ts;
                    
                    long h_idx = b * d_model * d_state + d * d_state + s;
                    
                    // h = dA * h + dB * x
                    float h_val = dA_ds * h_ptr[h_idx] + dB_ds * x_td;
                    h_ptr[h_idx] = h_val;
                    
                    yt_d += h_val * c_ts;
                }
                
                yt_d += D_val * x_td;
                out_ptr[b * seq_len * d_model + t * d_model + d] = yt_d;
            }
        }
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &selective_scan_forward, "Selective Scan Forward (CPU)");
}
