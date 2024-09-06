#include <algorithm> // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <type_traits>
#include <chrono>
// ORT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_lite_custom_op.h>

// NEON
// #define NEON
#ifdef NEON
#include <arm_neon.h>
#endif

// EELog
#define EELOG
#ifdef EELOG
std::vector<int> ee_res;
#endif

using std::cout;
using std::endl;
using namespace Ort::Custom;

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v)
{
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v)
{
  int total = 1;
  for (auto &i : v)
    total *= i;
  if (total < 0)
  {
    total *= -1;
  }
  return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T> &data, const std::vector<std::int64_t> &shape)
{
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}

template <typename T>
void load_file(std::string param_path, std::vector<T> &param)
{
  std::ifstream in(param_path);
  std::string line;
  if (std::is_same<T, float>::value)
  {
    while (getline(in, line))
    {
      param.push_back(std::stof(line));
    }
  }
  else
  {
    while (getline(in, line))
    {
      param.push_back(std::stoi(line));
    }
  }
}

int find_argmax(float *arr, int size)
{
  float max = arr[0];
  int argmax = 0;
  for (int i = 1; i < size; i++)
  {
    if (max < arr[i])
    {
      max = arr[i];
      argmax = i;
    }
  }
  return argmax;
}

struct Lif
{
  float thresh;
  float beta;
  Lif(const OrtApi *ort_api, const OrtKernelInfo *info)
  {
    auto ret = ort_api->KernelInfoGetAttribute_float(info, "thresh", &thresh);
    ret = ort_api->KernelInfoGetAttribute_float(info, "beta", &beta);
  }

  void Compute(const Ort::Custom::Tensor<float> &X,
               Ort::Custom::Tensor<float> &Z)
  {
    auto input_shape = X.Shape();
    auto x_raw = X.Data(); // ASSUME THAT THE INPUT HAS ONLY ONE BATCH
    auto z_raw = Z.Allocate(input_shape);
    int CIN = input_shape[1];
    int T = input_shape[2];
    int idx = 0;
    for (int c = 0; c < CIN; c++)
    {
      float mem_ = 0;
      for (int t = 0; t < T; t++)
      {
        mem_ = mem_ * beta + x_raw[idx];
        if (mem_ > thresh)
        {
          mem_ -= thresh;
          z_raw[idx] = 1;
        }
        else
        {
          z_raw[idx] = 0;
        }
        idx++;
      }
    }
  }
};
int ee_point = 0;
struct Lifee
{
  float thresh;
  float beta;
  float tee;
  Lifee(const OrtApi *ort_api, const OrtKernelInfo *info)
  {
    auto ret = ort_api->KernelInfoGetAttribute_float(info, "thresh", &thresh);
    ret = ort_api->KernelInfoGetAttribute_float(info, "beta", &beta);
    ret = ort_api->KernelInfoGetAttribute_float(info, "ee_thresh", &tee);
  }
  int prev_spike_sum[128] = {0};
  void Compute(
      const Ort::Custom::Tensor<float> &X,
      Ort::Custom::Tensor<float> &Z)
  {
    auto input_shape = X.Shape();
    auto x_raw = X.Data();
    auto z_raw = Z.Allocate(input_shape);
    int CIN = input_shape[1];
    int T = input_shape[2];

    // ASSUME THAT THE INPUT HAS ONLY ONE BATCH
    int spike_sum[T] = {0}; // FOR Early Exit
    float distance = 0;

    int idx = 0;
    
    for (int c = 0; c < CIN; c++)
    {
      float mem_ = 0;
      for (int t = 0; t < T; t++)
      {
        mem_ = mem_ * beta + x_raw[idx];
        if (mem_ > thresh)
        {
          mem_ -= thresh;
          z_raw[idx] = 1;
          spike_sum[t] += 1;
        }
        else
        {
          z_raw[idx] = 0;
        }
        idx++;
      }
    }
#ifdef NEON
    int t;
    for (t = 0; t + 4 <= T; t += 4)
    {
      int32x4_t diff = vsubq_s32(vld1q_s32(spike_sum + t), vld1q_s32(prev_spike_sum + t));
      int32x4_t abs_diff = vabsq_s32(diff);
      distance += vaddvq_s32(abs_diff);
    }
    for (; t < T; ++t)
    {
      distance += abs(spike_sum[t] - prev_spike_sum[t]);
    }
#else
    for (int t = 0; t < T; t++)
    {
      distance += abs(spike_sum[t] - prev_spike_sum[t]);
    }
#endif

    if (distance < (tee))
    {
#ifdef EELOG
      ee_res.push_back(ee_point);
#endif
      ORT_CXX_API_THROW("earlyexit", ORT_INVALID_GRAPH);
    }
#ifdef EELOG
    ee_point++;
#endif

    memcpy(prev_spike_sum, spike_sum, T * sizeof(int));
  }
};

int inference(Ort::Session &session,
              std::vector<const char *> &input_names_char,
              std::vector<Ort::Value> &input_tensors,
              std::vector<const char *> &output_names_char)
{
  std::vector<Ort::Value> output_tensors;

  try
  {
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                 input_names_char.size(), output_names_char.data(), output_names_char.size());
    // Get Prediction
    int pred = 0;
    for (int i = 0; i < session.GetOutputCount(); i++)
    {
      auto out = output_tensors[i].GetTensorMutableData<float>();
      auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      auto total_number_outs = calculate_product(output_shapes);
      pred = find_argmax(out, total_number_outs);
    }
    return pred;
  }
  catch (const Ort::Exception &exception)
  {
    // std::cout << "ERROR running model inference: " << exception.what() << std::endl;
    // exit(-1);
  }

  return -1;
}

int main(int argc, ORTCHAR_T *argv[])
{
  cout << "Revision 240508" << endl;
  // Get the ONNX Runtime version
  std::string onnxruntime_version = Ort::GetVersionString();
  std::cout << "ONNX Runtime version: " << onnxruntime_version << std::endl;
  std::basic_string<ORTCHAR_T> model_file = argv[1];
  std::basic_string<ORTCHAR_T> sample_file = argv[2];
  std::basic_string<ORTCHAR_T> target_file = argv[3];
  int repeats = std::atoi(argv[4]);

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Model:");
  Ort::SessionOptions session_options;

  // Execution provider
  // session_options.AppendExecutionProvider("FPGA");

  // Add Custom Op
  Ort::CustomOpDomain custom_domain{"custom"};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> custom_op_lif{Ort::Custom::CreateLiteCustomOp<Lif>("Lif", "CPUExecutionProvider")};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> custom_op_lifee{Ort::Custom::CreateLiteCustomOp<Lifee>("Lifee", "CPUExecutionProvider")};
  custom_domain.Add(custom_op_lif.get());
  custom_domain.Add(custom_op_lifee.get());
  session_options.Add(custom_domain);
  session_options.SetLogSeverityLevel(4);
  // session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

  // Create Session
  Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::int64_t> input_shapes;
  std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetInputCount(); i++)
  {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  int class_number;
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++)
  {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    class_number = output_shapes[1];
  }

  // pass data through model
  std::vector<const char *> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  std::vector<const char *> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                 [&](const std::string &str)
                 { return str.c_str(); });

  // Load Test Sample & Target
  std::vector<float> input_dataloader_values;
  load_file(sample_file, input_dataloader_values);
  std::vector<int> target_dataloader_values;
  load_file(target_file, target_dataloader_values);

  auto input_shape = input_shapes;
  auto total_number_elements = calculate_product(input_shape); // per sample length
  int confusion_matrix[class_number][class_number];
  memset(confusion_matrix, 0, sizeof(confusion_matrix));

  float average_repeats_inference_time = 0;
  float repeats_ = repeats;
  while (repeats--)
  {
    // Inference
    int pred;
    int prev_pred;
    auto input_dataloader_index = input_dataloader_values.begin();
    int total_number_inference = target_dataloader_values.size();
    int total_number_earlyexit = 0;
    float total_number_correct = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < total_number_inference; i++)
    {
      std::vector<Ort::Value> input_tensors;
      std::vector<float> input_tensor_values(input_dataloader_index, input_dataloader_index + total_number_elements);
      input_dataloader_index += total_number_elements;
      input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));
      pred = inference(session, input_names_char, input_tensors, output_names_char);
      if (pred == -1)
      {
        pred = prev_pred;
        total_number_earlyexit++;
      }
      else
      {
        prev_pred = pred;
      }
      if (pred == target_dataloader_values[i])
      {
        total_number_correct++;
      }
      confusion_matrix[target_dataloader_values[i]][pred] += 1;
#ifdef EELOG
      ee_point = 0;
#endif
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<float> ts = t1 - t0;
    float duration = ts.count() * 1000;
    printf("Accuracy: %.2f %% \n", (total_number_correct / total_number_inference) * 100);
    printf("Inference with early exit: %d \n", total_number_earlyexit);
    printf("Total inference sample: %d, with time: %.0f ms\n", total_number_inference, duration);
#ifdef EELOG
    int ee_point_sum[] = {0,0,0};
    for (int i = 0; i < ee_res.size(); i++)
    {
      int res = ee_res[i];
      // printf("%d ", res);
      ee_point_sum[res]++;
    }
    printf("\n");
    printf("Eearly Exit at L1: %d, L2: %d, L3: %d, None: %d.\n", ee_point_sum[0], ee_point_sum[1], ee_point_sum[2], (int)(total_number_inference-ee_res.size()));
#endif
    average_repeats_inference_time += duration;
  };
  average_repeats_inference_time /= repeats_;
  printf("---\n");
  printf("Average <%.0f> inference time: %.0f ms\n", repeats_, average_repeats_inference_time);
  printf("--- Confusion Matrix\n");
  printf("[");
  for (int i = 0; i < class_number; i++)
  {
    printf("[");
    for (int j = 0; j < class_number; j++)
    {
      if (j == class_number-1)
        printf("%d ", confusion_matrix[i][j]);
      else
        printf("%d, ", confusion_matrix[i][j]);
    }
    
    if (i == class_number-1)
      printf("]]\n");
    else
      printf("],\n");
  }
}