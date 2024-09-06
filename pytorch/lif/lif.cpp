#include "include/utils.h"


torch::Tensor
lif_fw(
    torch::Tensor input,
    float thresh_,
    float beta_)
{
  CHECK_INPUT(input);
  
  float *data = input.data_ptr<float>();

  float thresh = thresh_;
  float beta = beta_;
  // float thresh = 0.5;
  // float beta = 0.75;
  
  auto sizes = input.sizes();
  int CIN = sizes[1];
  int T = sizes[2];

  // ASSUME THAT THE INPUT HAS ONLY ONE BATCH
  float mem[CIN] = {0};
  float spike_out[CIN * T] = {0};
  float spike = 0;
  for (int i = 0; i < T; i++)
  {
    for (int c = 0; c < CIN; c++)
    {
      mem[c] *= beta;
      mem[c] += data[c * T + i];
      spike = mem[c] - thresh;
      if (spike > 0)
      {
        mem[c] -= thresh;
        spike_out[c * T + i] = 1;
      }
    }
  }

  torch::Tensor output = torch::zeros(sizes, input.options());
  output.copy_(torch::from_blob(spike_out, sizes, input.options()));
  return output;
}

std::vector<torch::Tensor> lifee_fw(
    torch::Tensor input,
    torch::Tensor prev_spike_sum,
    float thresh_,
    float beta_,    
    float tee)
{
  CHECK_INPUT(input);
  CHECK_INPUT(prev_spike_sum);

  bool early_exit = false;
  float *data = input.data_ptr<float>();
  int *prev = prev_spike_sum.data_ptr<int>(); // FOR Early Exit

  float thresh = thresh_;
  float beta = beta_;

  auto sizes = input.sizes();
  int CIN = sizes[1];
  int T = sizes[2];

  // ASSUME THAT THE INPUT HAS ONLY ONE BATCH
  float mem[CIN] = {0};
  float spike_out[CIN * T] = {0};
  int spike_sum[T] = {0}; // FOR Early Exit
  float distance = 0;
  for (int i = 0; i < T; i++)
  {
    for (int c = 0; c < CIN; c++)
    {
      mem[c] *= beta;
      mem[c] += data[c * T + i];
      if (mem[c] > thresh)
      {
        mem[c] -= thresh;
        spike_out[c * T + i] = 1;
        spike_sum[i] += 1;
      }
    }
    distance += abs(spike_sum[i] - prev[i]);
  }

  torch::Tensor output = torch::zeros(sizes, input.options());
  output.copy_(torch::from_blob(spike_out, sizes, input.options()));

  // FOR Early Exit
  torch::Tensor output_spike_sum = torch::zeros({T}, prev_spike_sum.options());
  output_spike_sum.copy_(torch::from_blob(spike_sum, {T}, prev_spike_sum.options()));

  torch::Tensor ee = torch::tensor((distance < tee) ? 1 : 0);
  return {output, output_spike_sum, ee};
}

// This function does NOT work properly, just for ONNX model export.
torch::Tensor lifee_onnx_fw(
    torch::Tensor input,
    float thresh_,
    float beta_,    
    float ee_thresh)
{
  CHECK_INPUT(input);
  auto sizes = input.sizes();
  torch::Tensor output = torch::zeros(sizes, input.options());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("lif_fw", &lif_fw);
  m.def("lifee_fw", &lifee_fw);
  m.def("lifee_onnx_fw", &lifee_onnx_fw);
}