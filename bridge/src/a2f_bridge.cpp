// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// C bridge implementation for NVIDIA Audio2Face-3D-SDK.

#include "a2f_bridge.h"

#include "audio2face/audio2face.h"
#include "audio2emotion/audio2emotion.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/tensor_float.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Verify binary compatibility of C param structs with SDK C++ structs.
// Size + first/last field offset checks catch layout divergence.
// ---------------------------------------------------------------------------
static_assert(sizeof(A2FSkinParams)   == sizeof(nva2f::AnimatorSkinParams),   "A2FSkinParams size mismatch");
static_assert(offsetof(A2FSkinParams, lowerFaceSmoothing) == offsetof(nva2f::AnimatorSkinParams, lowerFaceSmoothing), "A2FSkinParams first field offset mismatch");
static_assert(offsetof(A2FSkinParams, blinkOffset) == offsetof(nva2f::AnimatorSkinParams, blinkOffset), "A2FSkinParams last field offset mismatch");

static_assert(sizeof(A2FTongueParams) == sizeof(nva2f::AnimatorTongueParams), "A2FTongueParams size mismatch");
static_assert(offsetof(A2FTongueParams, tongueStrength) == offsetof(nva2f::AnimatorTongueParams, tongueStrength), "A2FTongueParams first field offset mismatch");
static_assert(offsetof(A2FTongueParams, tongueDepthOffset) == offsetof(nva2f::AnimatorTongueParams, tongueDepthOffset), "A2FTongueParams last field offset mismatch");

static_assert(sizeof(A2FTeethParams)  == sizeof(nva2f::AnimatorTeethParams),  "A2FTeethParams size mismatch");
static_assert(offsetof(A2FTeethParams, lowerTeethStrength) == offsetof(nva2f::AnimatorTeethParams, lowerTeethStrength), "A2FTeethParams first field offset mismatch");
static_assert(offsetof(A2FTeethParams, lowerTeethDepthOffset) == offsetof(nva2f::AnimatorTeethParams, lowerTeethDepthOffset), "A2FTeethParams last field offset mismatch");

static_assert(sizeof(A2FEyesParams)   == sizeof(nva2f::AnimatorEyesParams),   "A2FEyesParams size mismatch");
static_assert(offsetof(A2FEyesParams, eyeballsStrength) == offsetof(nva2f::AnimatorEyesParams, eyeballsStrength), "A2FEyesParams first field offset mismatch");
static_assert(offsetof(A2FEyesParams, saccadeSeed) == offsetof(nva2f::AnimatorEyesParams, saccadeSeed), "A2FEyesParams last field offset mismatch");

static_assert(sizeof(A2FSolverParams) == sizeof(nva2f::BlendshapeSolverParams), "A2FSolverParams size mismatch");
static_assert(offsetof(A2FSolverParams, L1Reg) == offsetof(nva2f::BlendshapeSolverParams, L1Reg), "A2FSolverParams first field offset mismatch");
static_assert(offsetof(A2FSolverParams, tolerance) == offsetof(nva2f::BlendshapeSolverParams, tolerance), "A2FSolverParams last field offset mismatch");

// ---------------------------------------------------------------------------
// Thread-local error string
// ---------------------------------------------------------------------------
static thread_local std::string g_last_error;

static void set_error(const char* msg) {
  g_last_error = msg ? msg : "";
}

static void set_error(const std::string& msg) {
  g_last_error = msg;
}

static void set_error_from_ec(std::error_code ec) {
  g_last_error = ec.message();
}

// ---------------------------------------------------------------------------
// SdkPtr: RAII wrapper calling Destroy() — like the SDK samples' UniquePtr.
// ---------------------------------------------------------------------------
struct Destroyer {
  template <typename T> void operator()(T* obj) const {
    if (obj) obj->Destroy();
  }
};
template <typename T> using SdkPtr = std::unique_ptr<T, Destroyer>;

// ---------------------------------------------------------------------------
// Internal session structure
// ---------------------------------------------------------------------------
struct A2FSession_t {
  // Core blendshape bundle (always present)
  SdkPtr<nva2f::IBlendshapeExecutorBundle> bs_bundle;

  // Optional emotion pipeline (when a2e_model_path provided)
  SdkPtr<nva2e::IClassifierModel::IEmotionModelInfo> a2e_model_info;
  SdkPtr<nva2e::IEmotionExecutor> emotion_executor;
  SdkPtr<nva2e::IEmotionBinder>   emotion_binder;

  // Model info pointers (for pose queries; only one pair is non-null)
  SdkPtr<nva2f::IRegressionModel::IGeometryModelInfo>         reg_geo_info;
  SdkPtr<nva2f::IRegressionModel::IBlendshapeSolveModelInfo>  reg_bs_info;
  SdkPtr<nva2f::IDiffusionModel::IGeometryModelInfo>          diff_geo_info;
  SdkPtr<nva2f::IDiffusionModel::IBlendshapeSolveModelInfo>   diff_bs_info;

  A2FModelType model_type;
  size_t       nb_tracks;
  bool         has_emotion;

  // User callback
  A2FBlendshapeCallback user_callback{nullptr};
  void*                 user_data{nullptr};

  // D2H buffer for GPU solver results
  SdkPtr<nva2x::IHostTensorFloat> d2h_buffer;

  // Frame counter for a2f_process()
  size_t frames_produced{0};

  // Callback error propagation: set by internal callbacks, checked by a2f_process()
  std::error_code last_callback_error;
};

// ---------------------------------------------------------------------------
// Helper: get the blendshape executor reference from a session.
// ---------------------------------------------------------------------------
static nva2f::IBlendshapeExecutor& get_bs_executor(A2FSession_t& s) {
  return s.bs_bundle->GetExecutor();
}

// Helper: get the face executor reference (IBlendshapeExecutor inherits IFaceExecutor).
static nva2f::IFaceExecutor& get_face_executor(A2FSession_t& s) {
  return s.bs_bundle->GetExecutor();
}

// Helper: get an IBlendshapeSolver* for a given track and solver type.
static A2FResult get_solver(A2FSession_t& s, size_t track, A2FSolverType solver_type, nva2f::IBlendshapeSolver** out) {
  std::error_code ec;
  if (solver_type == A2F_SOLVER_SKIN) {
    ec = nva2f::GetExecutorSkinSolver(get_face_executor(s), track, out);
  } else if (solver_type == A2F_SOLVER_TONGUE) {
    ec = nva2f::GetExecutorTongueSolver(get_face_executor(s), track, out);
  } else {
    set_error("invalid solver type");
    return A2F_ERROR_INVALID_ARG;
  }
  if (ec) {
    set_error_from_ec(ec);
    return A2F_ERROR_SDK;
  }
  if (!*out) {
    set_error("solver not available for this track");
    return A2F_ERROR_NOT_SUPPORTED;
  }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Macro: validate session handle
// ---------------------------------------------------------------------------
#define CHECK_SESSION(s)      \
  do {                        \
    if (!(s)) {               \
      set_error("null session handle"); \
      return A2F_ERROR_NULL_HANDLE;     \
    }                         \
  } while (0)

#define CHECK_TRACK(s, t)     \
  do {                        \
    if ((t) >= (s)->nb_tracks) { \
      set_error("track index out of range"); \
      return A2F_ERROR_INVALID_TRACK;        \
    }                         \
  } while (0)

// ---------------------------------------------------------------------------
// Bridge internal callbacks installed on the SDK executor
// ---------------------------------------------------------------------------

// Host-solver callback: HostResults already on CPU — forward directly.
static void host_results_cb(void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code ec) {
  auto* s = static_cast<A2FSession_t*>(userdata);
  if (ec) {
    s->last_callback_error = ec;
    set_error_from_ec(ec);
    return;
  }
  s->frames_produced++;
  if (s->user_callback) {
    s->user_callback(
      s->user_data,
      results.trackIndex,
      results.timeStampCurrentFrame,
      results.timeStampNextFrame,
      results.weights.Data(),
      results.weights.Size()
    );
  }
}

// Device-solver callback: DeviceResults on GPU — do D2H copy then forward.
static bool device_results_cb(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
  auto* s = static_cast<A2FSession_t*>(userdata);

  // Ensure D2H buffer is large enough
  const size_t count = results.weights.Size();
  if (!s->d2h_buffer || s->d2h_buffer->Size() < count) {
    s->d2h_buffer.reset();
    s->d2h_buffer.reset(nva2x::CreateHostPinnedTensorFloat(count));
  }

  // Synchronous D2H copy
  nva2x::HostTensorFloatView dst = *s->d2h_buffer;
  auto ec = nva2x::CopyDeviceToHost(
    dst.View(0, count),
    results.weights,
    results.cudaStream
  );
  if (ec) {
    s->last_callback_error = ec;
    set_error_from_ec(ec);
    return true; // continue processing; error is propagated via last_callback_error
  }

  // Synchronize so the data is available on host
  cudaStreamSynchronize(results.cudaStream);

  s->frames_produced++;
  if (s->user_callback) {
    s->user_callback(
      s->user_data,
      results.trackIndex,
      results.timeStampCurrentFrame,
      results.timeStampNextFrame,
      s->d2h_buffer->Data(),
      count
    );
  }
  return true; // continue
}

// ---------------------------------------------------------------------------
// a2f_get_last_error
// ---------------------------------------------------------------------------
const char* a2f_get_last_error(void) {
  return g_last_error.c_str();
}

// ---------------------------------------------------------------------------
// a2f_create_session
// ---------------------------------------------------------------------------
A2FResult a2f_create_session(
    A2FSession*  out,
    A2FModelType model_type,
    const char*  model_json_path,
    size_t       nb_tracks,
    int          use_gpu_solver,
    size_t       fps_num,
    size_t       fps_den,
    size_t       identity_idx,
    int          constant_noise,
    int          cuda_device,
    const char*  a2e_model_path
) {
  if (!out) { set_error("out is NULL"); return A2F_ERROR_INVALID_ARG; }
  if (!model_json_path) { set_error("model_json_path is NULL"); return A2F_ERROR_INVALID_ARG; }
  if (nb_tracks == 0) { set_error("nb_tracks must be >= 1"); return A2F_ERROR_INVALID_ARG; }

  *out = nullptr;

  // Set CUDA device
  {
    auto ec = nva2x::SetCudaDeviceIfNeeded(cuda_device);
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  }

  auto session = std::make_unique<A2FSession_t>();
  session->model_type = model_type;
  session->nb_tracks  = nb_tracks;
  session->has_emotion = (a2e_model_path != nullptr);

  // Create the blendshape executor bundle
  const auto exec_option = nva2f::IGeometryExecutor::ExecutionOption::All;
  const bool gpu_solver = (use_gpu_solver != 0);

  if (model_type == A2F_MODEL_REGRESSION) {
    nva2f::IRegressionModel::IGeometryModelInfo* geo_info_raw = nullptr;
    nva2f::IRegressionModel::IBlendshapeSolveModelInfo* bs_info_raw = nullptr;

    auto* bundle = nva2f::ReadRegressionBlendshapeSolveExecutorBundle(
      nb_tracks, model_json_path, exec_option, gpu_solver,
      fps_num, fps_den,
      &geo_info_raw, &bs_info_raw
    );
    if (!bundle) {
      set_error("failed to create regression blendshape executor bundle");
      return A2F_ERROR_SDK;
    }
    session->bs_bundle.reset(bundle);
    session->reg_geo_info.reset(geo_info_raw);
    session->reg_bs_info.reset(bs_info_raw);

  } else if (model_type == A2F_MODEL_DIFFUSION) {
    nva2f::IDiffusionModel::IGeometryModelInfo* geo_info_raw = nullptr;
    nva2f::IDiffusionModel::IBlendshapeSolveModelInfo* bs_info_raw = nullptr;

    auto* bundle = nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
      nb_tracks, model_json_path, exec_option, gpu_solver,
      identity_idx, (constant_noise != 0),
      &geo_info_raw, &bs_info_raw
    );
    if (!bundle) {
      set_error("failed to create diffusion blendshape executor bundle");
      return A2F_ERROR_SDK;
    }
    session->bs_bundle.reset(bundle);
    session->diff_geo_info.reset(geo_info_raw);
    session->diff_bs_info.reset(bs_info_raw);

  } else {
    set_error("unknown model_type");
    return A2F_ERROR_INVALID_ARG;
  }

  // Install internal SDK callback
  auto& bs_exec = get_bs_executor(*session);
  if (bs_exec.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::HOST) {
    auto ec = bs_exec.SetResultsCallback(host_results_cb, session.get());
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  } else {
    auto ec = bs_exec.SetResultsCallback(device_results_cb, session.get());
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  }

  // Setup emotion pipeline
  if (session->has_emotion) {
    // Load A2E model
    auto* model_info_raw = nva2e::ReadClassifierModelInfo(a2e_model_path);
    if (!model_info_raw) {
      set_error("failed to load a2e model");
      return A2F_ERROR_SDK;
    }
    session->a2e_model_info.reset(model_info_raw);

    // Create emotion executor sharing the bundle's audio accumulators
    auto cuda_stream = session->bs_bundle->GetCudaStream().Data();

    nva2e::EmotionExecutorCreationParameters emo_params;
    emo_params.cudaStream = cuda_stream;
    emo_params.nbTracks = nb_tracks;

    // Build array of audio accumulator pointers
    std::vector<const nva2x::IAudioAccumulator*> audio_accs(nb_tracks);
    for (size_t i = 0; i < nb_tracks; ++i) {
      audio_accs[i] = &session->bs_bundle->GetAudioAccumulator(i);
    }
    emo_params.sharedAudioAccumulators = audio_accs.data();

    // Get creation parameters from model (bufferLength=60000, 30fps, skip=30)
    auto classifier_params = session->a2e_model_info->GetExecutorCreationParameters(
      60000, 30, 1, 30
    );

    auto* emo_exec_raw = nva2e::CreateClassifierEmotionExecutor(emo_params, classifier_params);
    if (!emo_exec_raw) {
      set_error("failed to create emotion executor");
      return A2F_ERROR_SDK;
    }
    session->emotion_executor.reset(emo_exec_raw);

    // Create emotion binder to connect emotion output -> emotion accumulators
    std::vector<nva2x::IEmotionAccumulator*> emo_accs(nb_tracks);
    for (size_t i = 0; i < nb_tracks; ++i) {
      emo_accs[i] = &session->bs_bundle->GetEmotionAccumulator(i);
    }

    auto* binder_raw = nva2e::CreateEmotionBinder(
      *session->emotion_executor, emo_accs.data(), nb_tracks
    );
    if (!binder_raw) {
      set_error("failed to create emotion binder");
      return A2F_ERROR_SDK;
    }
    session->emotion_binder.reset(binder_raw);

  } else {
    // No emotion model: fill emotion accumulators with zeros and close them.
    // Follows the AddDefaultEmotion pattern from sample-a2f-executor.
    for (size_t i = 0; i < nb_tracks; ++i) {
      auto& emo_acc = session->bs_bundle->GetEmotionAccumulator(i);
      const size_t emo_size = emo_acc.GetEmotionSize();
      std::vector<float> zeros(emo_size, 0.0f);
      auto ec = emo_acc.Accumulate(
        0,
        nva2x::HostTensorFloatConstView{zeros.data(), zeros.size()},
        session->bs_bundle->GetCudaStream().Data()
      );
      if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
      ec = emo_acc.Close();
      if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
    }
  }

  *out = session.release();
  g_last_error.clear();
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_destroy_session
// ---------------------------------------------------------------------------
A2FResult a2f_destroy_session(A2FSession session) {
  // Destroy order matters: binder before executor, executor before bundle
  if (session) {
    session->emotion_binder.reset();
    session->emotion_executor.reset();
    session->a2e_model_info.reset();
    session->d2h_buffer.reset();
    session->reg_bs_info.reset();
    session->reg_geo_info.reset();
    session->diff_bs_info.reset();
    session->diff_geo_info.reset();
    session->bs_bundle.reset();
    delete session;
  }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_set_blendshape_callback
// ---------------------------------------------------------------------------
A2FResult a2f_set_blendshape_callback(A2FSession session, A2FBlendshapeCallback callback, void* userdata) {
  CHECK_SESSION(session);
  session->user_callback = callback;
  session->user_data = userdata;
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_push_audio
// ---------------------------------------------------------------------------
A2FResult a2f_push_audio(A2FSession session, size_t track, const float* samples, size_t count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!samples && count > 0) { set_error("samples is NULL"); return A2F_ERROR_INVALID_ARG; }

  auto& acc = session->bs_bundle->GetAudioAccumulator(track);
  auto ec = acc.Accumulate(
    nva2x::HostTensorFloatConstView{samples, count},
    session->bs_bundle->GetCudaStream().Data()
  );
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_close_audio
// ---------------------------------------------------------------------------
A2FResult a2f_close_audio(A2FSession session, size_t track) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);

  auto ec = session->bs_bundle->GetAudioAccumulator(track).Close();
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_process
// ---------------------------------------------------------------------------
A2FResult a2f_process(A2FSession session, size_t* out_frames) {
  CHECK_SESSION(session);

  session->frames_produced = 0;
  session->last_callback_error = {};
  auto& bs_exec = get_bs_executor(*session);

  // Low-latency loop: prefer blendshape execution, fall back to emotion.
  while (true) {
    // Try blendshape first
    if (nva2x::GetNbReadyTracks(bs_exec) > 0) {
      auto ec = bs_exec.Execute(nullptr);
      if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
      continue;
    }

    // Try emotion (if available)
    if (session->has_emotion && session->emotion_executor) {
      if (nva2x::GetNbReadyTracks(*session->emotion_executor) > 0) {
        auto ec = session->emotion_executor->Execute(nullptr);
        if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
        continue;
      }

      // Check if all audio is closed → close emotion accumulators
      bool all_audio_closed = true;
      for (size_t i = 0; i < session->nb_tracks; ++i) {
        if (!session->bs_bundle->GetAudioAccumulator(i).IsClosed()) {
          all_audio_closed = false;
          break;
        }
      }

      if (all_audio_closed) {
        bool any_emo_open = false;
        for (size_t i = 0; i < session->nb_tracks; ++i) {
          auto& emo_acc = session->bs_bundle->GetEmotionAccumulator(i);
          if (!emo_acc.IsClosed()) {
            // Only close if no more emotion executions are available
            bool still_has_work = false;
            for (size_t t = 0; t < session->nb_tracks; ++t) {
              if (session->emotion_executor->GetNbAvailableExecutions(t) > 0) {
                still_has_work = true;
                break;
              }
            }
            if (!still_has_work) {
              emo_acc.Close();
            } else {
              any_emo_open = true;
            }
          }
        }
        if (any_emo_open) continue;

        // After closing emotion, blendshape might have more work
        if (nva2x::GetNbReadyTracks(bs_exec) > 0) continue;
      }
    }

    break;
  }

  if (out_frames) *out_frames = session->frames_produced;

  // Propagate any error that occurred inside a callback
  if (session->last_callback_error) {
    set_error_from_ec(session->last_callback_error);
    return A2F_ERROR_SDK;
  }

  return A2F_OK;
}

// ---------------------------------------------------------------------------
// a2f_wait
// ---------------------------------------------------------------------------
A2FResult a2f_wait(A2FSession session, size_t track) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);

  auto ec = get_bs_executor(*session).Wait(track);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Info queries
// ---------------------------------------------------------------------------
A2FResult a2f_get_weight_count(A2FSession session, size_t* out_count) {
  CHECK_SESSION(session);
  if (!out_count) { set_error("out_count is NULL"); return A2F_ERROR_INVALID_ARG; }
  *out_count = get_bs_executor(*session).GetWeightCount();
  return A2F_OK;
}

A2FResult a2f_get_sampling_rate(A2FSession session, size_t* out_rate) {
  CHECK_SESSION(session);
  if (!out_rate) { set_error("out_rate is NULL"); return A2F_ERROR_INVALID_ARG; }
  *out_rate = get_bs_executor(*session).GetSamplingRate();
  return A2F_OK;
}

A2FResult a2f_get_frame_rate(A2FSession session, size_t* out_numerator, size_t* out_denominator) {
  CHECK_SESSION(session);
  if (!out_numerator || !out_denominator) { set_error("output pointer is NULL"); return A2F_ERROR_INVALID_ARG; }
  get_bs_executor(*session).GetFrameRate(*out_numerator, *out_denominator);
  return A2F_OK;
}

A2FResult a2f_get_nb_tracks(A2FSession session, size_t* out_nb_tracks) {
  CHECK_SESSION(session);
  if (!out_nb_tracks) { set_error("out_nb_tracks is NULL"); return A2F_ERROR_INVALID_ARG; }
  *out_nb_tracks = session->nb_tracks;
  return A2F_OK;
}

A2FResult a2f_get_pose_name(A2FSession session, size_t track, A2FSolverType solver, size_t index, const char** out_name) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_name) { set_error("out_name is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* s = nullptr;
  A2FResult r = get_solver(*session, track, solver, &s);
  if (r != A2F_OK) return r;

  *out_name = s->GetPoseName(index);
  return A2F_OK;
}

A2FResult a2f_get_num_poses(A2FSession session, size_t track, A2FSolverType solver, size_t* out_count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_count) { set_error("out_count is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* s = nullptr;
  A2FResult r = get_solver(*session, track, solver, &s);
  if (r != A2F_OK) return r;

  *out_count = static_cast<size_t>(s->NumBlendshapePoses());
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Track management
// ---------------------------------------------------------------------------
A2FResult a2f_reset_track(A2FSession session, size_t track) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);

  // Reset audio accumulator
  auto ec = session->bs_bundle->GetAudioAccumulator(track).Reset();
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  // Reset emotion accumulator
  ec = session->bs_bundle->GetEmotionAccumulator(track).Reset();
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  // Reset executor for this track
  ec = get_bs_executor(*session).Reset(track);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  // Reset emotion executor for this track (if present)
  if (session->has_emotion && session->emotion_executor) {
    ec = session->emotion_executor->Reset(track);
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  }

  // If no emotion model, re-apply default emotion
  if (!session->has_emotion) {
    auto& emo_acc = session->bs_bundle->GetEmotionAccumulator(track);
    const size_t emo_size = emo_acc.GetEmotionSize();
    std::vector<float> zeros(emo_size, 0.0f);
    ec = emo_acc.Accumulate(
      0,
      nva2x::HostTensorFloatConstView{zeros.data(), zeros.size()},
      session->bs_bundle->GetCudaStream().Data()
    );
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
    ec = emo_acc.Close();
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  }

  return A2F_OK;
}

A2FResult a2f_drop_processed_data(A2FSession session, size_t track) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);

  auto& bs_exec = get_bs_executor(*session);

  // Drop emotions
  auto& emo_acc = session->bs_bundle->GetEmotionAccumulator(track);
  if (!emo_acc.IsEmpty()) {
    auto ts_to_read = get_face_executor(*session).GetNextEmotionTimestampToRead(track);
    auto last_ts = emo_acc.LastAccumulatedTimestamp();
    auto ts_to_drop = std::min(ts_to_read, last_ts);
    auto ec = emo_acc.DropEmotionsBefore(ts_to_drop);
    if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  }

  // Drop audio samples
  auto sample_bs = get_face_executor(*session).GetNextAudioSampleToRead(track);
  size_t sample_to_drop = sample_bs;

  if (session->has_emotion && session->emotion_executor) {
    auto sample_emo = session->emotion_executor->GetNextAudioSampleToRead(track);
    sample_to_drop = std::min(sample_bs, sample_emo);
  }

  auto ec = session->bs_bundle->GetAudioAccumulator(track).DropSamplesBefore(sample_to_drop);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Animator parameter control
// ---------------------------------------------------------------------------
A2FResult a2f_get_skin_params(A2FSession session, size_t track, A2FSkinParams* out_params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_params) { set_error("out_params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorSkinParams params;
  auto ec = nva2f::GetExecutorSkinParameters(get_face_executor(*session), track, params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  std::memcpy(out_params, &params, sizeof(params));
  return A2F_OK;
}

A2FResult a2f_set_skin_params(A2FSession session, size_t track, const A2FSkinParams* params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!params) { set_error("params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorSkinParams sdk_params;
  std::memcpy(&sdk_params, params, sizeof(sdk_params));
  auto ec = nva2f::SetExecutorSkinParameters(get_face_executor(*session), track, sdk_params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_get_tongue_params(A2FSession session, size_t track, A2FTongueParams* out_params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_params) { set_error("out_params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorTongueParams params;
  auto ec = nva2f::GetExecutorTongueParameters(get_face_executor(*session), track, params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  std::memcpy(out_params, &params, sizeof(params));
  return A2F_OK;
}

A2FResult a2f_set_tongue_params(A2FSession session, size_t track, const A2FTongueParams* params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!params) { set_error("params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorTongueParams sdk_params;
  std::memcpy(&sdk_params, params, sizeof(sdk_params));
  auto ec = nva2f::SetExecutorTongueParameters(get_face_executor(*session), track, sdk_params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_get_teeth_params(A2FSession session, size_t track, A2FTeethParams* out_params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_params) { set_error("out_params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorTeethParams params;
  auto ec = nva2f::GetExecutorTeethParameters(get_face_executor(*session), track, params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  std::memcpy(out_params, &params, sizeof(params));
  return A2F_OK;
}

A2FResult a2f_set_teeth_params(A2FSession session, size_t track, const A2FTeethParams* params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!params) { set_error("params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorTeethParams sdk_params;
  std::memcpy(&sdk_params, params, sizeof(sdk_params));
  auto ec = nva2f::SetExecutorTeethParameters(get_face_executor(*session), track, sdk_params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_get_eyes_params(A2FSession session, size_t track, A2FEyesParams* out_params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_params) { set_error("out_params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorEyesParams params;
  auto ec = nva2f::GetExecutorEyesParameters(get_face_executor(*session), track, params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  std::memcpy(out_params, &params, sizeof(params));
  return A2F_OK;
}

A2FResult a2f_set_eyes_params(A2FSession session, size_t track, const A2FEyesParams* params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!params) { set_error("params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::AnimatorEyesParams sdk_params;
  std::memcpy(&sdk_params, params, sizeof(sdk_params));
  auto ec = nva2f::SetExecutorEyesParameters(get_face_executor(*session), track, sdk_params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Input strength
// ---------------------------------------------------------------------------
A2FResult a2f_get_input_strength(A2FSession session, float* out_strength) {
  CHECK_SESSION(session);
  if (!out_strength) { set_error("out_strength is NULL"); return A2F_ERROR_INVALID_ARG; }

  auto ec = nva2f::GetExecutorInputStrength(get_face_executor(*session), *out_strength);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_set_input_strength(A2FSession session, float strength) {
  CHECK_SESSION(session);

  auto ec = nva2f::SetExecutorInputStrength(get_face_executor(*session), strength);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Blendshape solver parameter control
// ---------------------------------------------------------------------------
A2FResult a2f_get_solver_params(A2FSession session, size_t track, A2FSolverType solver_type, A2FSolverParams* out_params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_params) { set_error("out_params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  const auto& params = solver->GetParameters();
  std::memcpy(out_params, &params, sizeof(params));
  return A2F_OK;
}

A2FResult a2f_set_solver_params(A2FSession session, size_t track, A2FSolverType solver_type, const A2FSolverParams* params) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!params) { set_error("params is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  nva2f::BlendshapeSolverParams sdk_params;
  std::memcpy(&sdk_params, params, sizeof(sdk_params));
  auto ec = solver->SetParameters(sdk_params);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  ec = solver->Prepare();
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }

  return A2F_OK;
}

// ---------------------------------------------------------------------------
// Per-pose multipliers and offsets
// ---------------------------------------------------------------------------
A2FResult a2f_get_multiplier(A2FSession session, size_t track, A2FSolverType solver_type, const char* pose_name, float* out_value) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!pose_name || !out_value) { set_error("pose_name or out_value is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  auto ec = solver->GetMultiplier(pose_name, *out_value);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_set_multiplier(A2FSession session, size_t track, A2FSolverType solver_type, const char* pose_name, float value) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!pose_name) { set_error("pose_name is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  auto ec = solver->SetMultiplier(pose_name, value);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_get_offset(A2FSession session, size_t track, A2FSolverType solver_type, const char* pose_name, float* out_value) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!pose_name || !out_value) { set_error("pose_name or out_value is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  auto ec = solver->GetOffset(pose_name, *out_value);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_set_offset(A2FSession session, size_t track, A2FSolverType solver_type, const char* pose_name, float value) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!pose_name) { set_error("pose_name is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  auto ec = solver->SetOffset(pose_name, value);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

// Batch multipliers
A2FResult a2f_get_multipliers(A2FSession session, size_t track, A2FSolverType solver_type, float* out_values, size_t count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_values) { set_error("out_values is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  nva2x::HostTensorFloatView view(out_values, count);
  auto ec = solver->GetMultipliers(view);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_set_multipliers(A2FSession session, size_t track, A2FSolverType solver_type, const float* values, size_t count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!values) { set_error("values is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  nva2x::HostTensorFloatConstView view(values, count);
  auto ec = solver->SetMultipliers(view);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_get_offsets(A2FSession session, size_t track, A2FSolverType solver_type, float* out_values, size_t count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!out_values) { set_error("out_values is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  nva2x::HostTensorFloatView view(out_values, count);
  auto ec = solver->GetOffsets(view);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}

A2FResult a2f_set_offsets(A2FSession session, size_t track, A2FSolverType solver_type, const float* values, size_t count) {
  CHECK_SESSION(session);
  CHECK_TRACK(session, track);
  if (!values) { set_error("values is NULL"); return A2F_ERROR_INVALID_ARG; }

  nva2f::IBlendshapeSolver* solver = nullptr;
  A2FResult r = get_solver(*session, track, solver_type, &solver);
  if (r != A2F_OK) return r;

  nva2x::HostTensorFloatConstView view(values, count);
  auto ec = solver->SetOffsets(view);
  if (ec) { set_error_from_ec(ec); return A2F_ERROR_SDK; }
  return A2F_OK;
}
