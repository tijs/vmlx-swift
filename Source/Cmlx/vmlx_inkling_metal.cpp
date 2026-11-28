#include "vmlx_inkling_metal.h"
#include "mlx/c/error.h"

#ifdef _METAL_
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/primitives.h"

void* vmlx_inkling_paged_prefill_pipeline(bool align_query, bool align_keys) {
  using namespace mlx::core;
  try {
    auto& device = metal::device(Device::gpu);
    bool mask = true;
    bool no = false;
    int window = -1;
    const metal::MTLFCList constants = {
        {&align_query, MTL::DataType::DataTypeBool, 200},
        {&align_keys, MTL::DataType::DataTypeBool, 201},
        {&mask, MTL::DataType::DataTypeBool, 300},
        {&no, MTL::DataType::DataTypeBool, 301},
        {&no, MTL::DataType::DataTypeBool, 302},
        {&no, MTL::DataType::DataTypeBool, 303},
        {&window, MTL::DataType::DataTypeInt, 304},
    };
    const std::string name = "inkling_paged_steel_bf16";
    const auto cache_key =
        name + (align_query ? "_q_aligned" : "_q_tail") +
        (align_keys ? "_k_aligned" : "_k_tail");
    // Use the same compiler configuration as resident STEEL attention.
    auto* library = device.get_library(name, [] {
      std::string source = metal::utils();
      source += "\n#define VMLX_PAGED_STEEL_ATTENTION\n";
      source += metal::steel_attention();
      source += R"(
instantiate_kernel("inkling_paged_steel_bf16", attention, bfloat16_t,
                   32, 16, 128, 128, 4, 1, bfloat16_t, float);
)";
      return source;
    });
    return device.get_kernel(name, library, cache_key, constants);
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return nullptr;
  }
}
namespace {
using namespace mlx::core;

class InklingPagedVectorMerge final : public UnaryPrimitive {
 public:
  using UnaryPrimitive::UnaryPrimitive;

  const char* name() const override {
    return "InklingPagedVectorMerge";
  }

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("Inkling paged vector reduction requires Metal");
  }

  void eval_gpu(const std::vector<array>& inputs, array& out) override {
    if (!inputs[0].flags().row_contiguous || !inputs[1].flags().row_contiguous ||
        !inputs[2].flags().row_contiguous) {
      throw std::invalid_argument("Inkling paged vector state must be contiguous");
    }
    const auto& s = stream();
    auto& device = metal::device(s.device);
    auto& encoder = metal::get_command_encoder(s);
    out.set_data(allocator::malloc(out.nbytes()));
    encoder.set_compute_pipeline_state(
        device.get_kernel("sdpa_vector_2pass_2_bfloat16_t_128"));
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(inputs[0].shape(2), 4);
    encoder.dispatch_threadgroups(
        MTL::Size(out.shape(1), out.shape(2), 1), MTL::Size(1024, 1, 1));
  }
};
} // namespace

int vmlx_inkling_paged_vector_merge(
    mlx_array* result, mlx_array partials, mlx_array sums, mlx_array maxima,
    mlx_stream stream) {
  using namespace mlx::core;
  try {
    const auto& p = mlx_array_get_(partials);
    const auto& s = mlx_array_get_(sums);
    const auto& m = mlx_array_get_(maxima);
    if (p.ndim() != 4 || p.shape(3) != 128 || p.dtype() != bfloat16 ||
        p.shape(2) <= 0 || p.shape(2) % 32 != 0) {
      throw std::invalid_argument("Invalid Inkling paged vector partials");
    }
    const Shape reduction_shape{p.shape(0), p.shape(1), p.shape(2)};
    if (s.shape() != reduction_shape || m.shape() != reduction_shape ||
        s.dtype() != float32 || m.dtype() != float32) {
      throw std::invalid_argument("Invalid Inkling paged vector reduction state");
    }
    mlx_array_set_(*result, array(
        {1, p.shape(0), p.shape(1), p.shape(3)}, bfloat16,
        std::make_shared<InklingPagedVectorMerge>(mlx_stream_get_(stream)),
        {p, s, m}));
    return 0;
  } catch (const std::exception& error) {
    mlx_error(error.what());
    return 1;
  }
}
#else
void* vmlx_inkling_paged_prefill_pipeline(bool, bool) {
  return nullptr;
}

int vmlx_inkling_paged_vector_merge(
    mlx_array*, mlx_array, mlx_array, mlx_array, mlx_stream) {
  mlx_error("Inkling paged vector reduction requires Metal");
  return 1;
}
#endif
