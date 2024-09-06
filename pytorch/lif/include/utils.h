#include <torch/extension.h>

#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CPU(x);        \
  CHECK_CONTIGUOUS(x)

torch::Tensor lif_fw(
    torch::Tensor input,
    float thresh_,
    float beta_);

std::vector<torch::Tensor> lifee_fw(
    torch::Tensor input,
    torch::Tensor prev_spike_sum,
    float thresh_,
    float beta_,    
    float tee);

torch::Tensor lifee_onnx_fw(
    torch::Tensor input,
    float thresh_,
    float beta_,
    float tee);