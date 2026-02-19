// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// C bridge for NVIDIA Audio2Face-3D-SDK.
// Provides flat extern "C" functions and opaque handles for P/Invoke consumption.
//
// Compilable as both C (gcc -c -x c) and C++ (g++ -c -x c++).

#ifndef A2F_BRIDGE_H
#define A2F_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// DLL export / import
// ---------------------------------------------------------------------------
#ifdef _WIN32
  #ifdef A2F_BRIDGE_EXPORTS
    #define A2F_API __declspec(dllexport)
  #else
    #define A2F_API __declspec(dllimport)
  #endif
#else
  #define A2F_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Result codes
// ---------------------------------------------------------------------------
typedef int32_t A2FResult;

#define A2F_OK                   0
#define A2F_ERROR_INVALID_ARG   -1
#define A2F_ERROR_SDK           -2
#define A2F_ERROR_NULL_HANDLE   -3
#define A2F_ERROR_INVALID_TRACK -4
#define A2F_ERROR_NOT_SUPPORTED -5

// ---------------------------------------------------------------------------
// Opaque session handle
// ---------------------------------------------------------------------------
typedef struct A2FSession_t* A2FSession;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------
typedef enum {
  A2F_MODEL_REGRESSION = 0,
  A2F_MODEL_DIFFUSION  = 1
} A2FModelType;

typedef enum {
  A2F_SOLVER_SKIN   = 0,
  A2F_SOLVER_TONGUE = 1
} A2FSolverType;

// ---------------------------------------------------------------------------
// Callback type
// ---------------------------------------------------------------------------

// Called once per blendshape frame produced during a2f_process().
// weights points to weight_count floats (valid only for the duration of the
// callback).  For GPU-solver sessions the bridge performs D2H copy before
// invoking this callback.
typedef void (*A2FBlendshapeCallback)(
    void*        userdata,
    size_t       track_index,
    int64_t      ts_current,
    int64_t      ts_next,
    const float* weights,
    size_t       weight_count
);

// ---------------------------------------------------------------------------
// Parameter structs  (binary-compatible with SDK C++ structs)
// ---------------------------------------------------------------------------

typedef struct {
  float lowerFaceSmoothing;
  float upperFaceSmoothing;
  float lowerFaceStrength;
  float upperFaceStrength;
  float faceMaskLevel;
  float faceMaskSoftness;
  float skinStrength;
  float blinkStrength;
  float eyelidOpenOffset;
  float lipOpenOffset;
  float blinkOffset;
} A2FSkinParams;

typedef struct {
  float tongueStrength;
  float tongueHeightOffset;
  float tongueDepthOffset;
} A2FTongueParams;

typedef struct {
  float lowerTeethStrength;
  float lowerTeethHeightOffset;
  float lowerTeethDepthOffset;
} A2FTeethParams;

typedef struct {
  float eyeballsStrength;
  float saccadeStrength;
  float rightEyeballRotationOffsetX;
  float rightEyeballRotationOffsetY;
  float leftEyeballRotationOffsetX;
  float leftEyeballRotationOffsetY;
  float saccadeSeed;
} A2FEyesParams;

typedef struct {
  float L1Reg;
  float L2Reg;
  float SymmetryReg;
  float TemporalReg;
  float templateBBSize;
  float tolerance;
} A2FSolverParams;

// ---------------------------------------------------------------------------
// Error string  (thread-local, valid until next bridge call on same thread)
// ---------------------------------------------------------------------------

A2F_API const char* a2f_get_last_error(void);

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------

// Create a session wrapping an IBlendshapeExecutorBundle.
//
// model_type        Regression or Diffusion.
// model_json_path   Path to the SDK's model.json.
// nb_tracks         Number of audio tracks (>= 1).
// use_gpu_solver    Non-zero to use GPU blendshape solver.
// fps_num           Frame rate numerator  (regression only, e.g. 60).
// fps_den           Frame rate denominator (regression only, e.g. 1).
// identity_idx      Identity index (diffusion only, ignored for regression).
// constant_noise    Non-zero for constant noise (diffusion only).
// cuda_device       CUDA device ordinal (e.g. 0).
// a2e_model_path    Path to Audio2Emotion model.json, or NULL to use default
//                   (zero) emotions.
A2F_API A2FResult a2f_create_session(
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
);

// Destroy a session and all SDK objects it owns.
A2F_API A2FResult a2f_destroy_session(A2FSession session);

// ---------------------------------------------------------------------------
// Blendshape results callback
// ---------------------------------------------------------------------------

// Install a callback that receives blendshape weight frames produced during
// a2f_process().  Must be called before the first a2f_process().
A2F_API A2FResult a2f_set_blendshape_callback(
    A2FSession             session,
    A2FBlendshapeCallback  callback,
    void*                  userdata
);

// ---------------------------------------------------------------------------
// Audio input
// ---------------------------------------------------------------------------

// Push PCM float audio samples (16 kHz mono, range [-1,1]) for a track.
A2F_API A2FResult a2f_push_audio(
    A2FSession   session,
    size_t       track,
    const float* samples,
    size_t       count
);

// Signal that no more audio will be pushed for a track.
A2F_API A2FResult a2f_close_audio(
    A2FSession session,
    size_t     track
);

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// Run the low-latency processing loop: emotion Execute() then blendshape
// Execute(), repeating until no tracks are ready.  Fires the blendshape
// callback for each frame produced.
// out_frames receives the number of frames produced (may be NULL).
A2F_API A2FResult a2f_process(
    A2FSession session,
    size_t*    out_frames
);

// Block until all asynchronously scheduled work for a track completes.
A2F_API A2FResult a2f_wait(
    A2FSession session,
    size_t     track
);

// ---------------------------------------------------------------------------
// Info queries
// ---------------------------------------------------------------------------

A2F_API A2FResult a2f_get_weight_count(
    A2FSession session,
    size_t*    out_count
);

A2F_API A2FResult a2f_get_sampling_rate(
    A2FSession session,
    size_t*    out_rate
);

A2F_API A2FResult a2f_get_frame_rate(
    A2FSession session,
    size_t*    out_numerator,
    size_t*    out_denominator
);

A2F_API A2FResult a2f_get_nb_tracks(
    A2FSession session,
    size_t*    out_nb_tracks
);

// Get the name of a blendshape pose from the specified solver.
// out_name receives a pointer to an internal string (valid for session lifetime).
A2F_API A2FResult a2f_get_pose_name(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    size_t        index,
    const char**  out_name
);

// Get the number of blendshape poses from the specified solver.
A2F_API A2FResult a2f_get_num_poses(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    size_t*       out_count
);

// ---------------------------------------------------------------------------
// Track management
// ---------------------------------------------------------------------------

// Reset a track so it can process new audio.  Resets accumulators and executor.
A2F_API A2FResult a2f_reset_track(
    A2FSession session,
    size_t     track
);

// Drop already-processed audio/emotion data to free memory.
A2F_API A2FResult a2f_drop_processed_data(
    A2FSession session,
    size_t     track
);

// ---------------------------------------------------------------------------
// Animator parameter control
// ---------------------------------------------------------------------------

A2F_API A2FResult a2f_get_skin_params(
    A2FSession    session,
    size_t        track,
    A2FSkinParams* out_params
);

A2F_API A2FResult a2f_set_skin_params(
    A2FSession          session,
    size_t              track,
    const A2FSkinParams* params
);

A2F_API A2FResult a2f_get_tongue_params(
    A2FSession      session,
    size_t          track,
    A2FTongueParams* out_params
);

A2F_API A2FResult a2f_set_tongue_params(
    A2FSession            session,
    size_t                track,
    const A2FTongueParams* params
);

A2F_API A2FResult a2f_get_teeth_params(
    A2FSession     session,
    size_t         track,
    A2FTeethParams* out_params
);

A2F_API A2FResult a2f_set_teeth_params(
    A2FSession           session,
    size_t               track,
    const A2FTeethParams* params
);

A2F_API A2FResult a2f_get_eyes_params(
    A2FSession     session,
    size_t         track,
    A2FEyesParams* out_params
);

A2F_API A2FResult a2f_set_eyes_params(
    A2FSession           session,
    size_t               track,
    const A2FEyesParams* params
);

// ---------------------------------------------------------------------------
// Input strength
// ---------------------------------------------------------------------------

A2F_API A2FResult a2f_get_input_strength(
    A2FSession session,
    float*     out_strength
);

A2F_API A2FResult a2f_set_input_strength(
    A2FSession session,
    float      strength
);

// ---------------------------------------------------------------------------
// Blendshape solver parameter control
// ---------------------------------------------------------------------------

A2F_API A2FResult a2f_get_solver_params(
    A2FSession      session,
    size_t          track,
    A2FSolverType   solver,
    A2FSolverParams* out_params
);

A2F_API A2FResult a2f_set_solver_params(
    A2FSession            session,
    size_t                track,
    A2FSolverType         solver,
    const A2FSolverParams* params
);

// ---------------------------------------------------------------------------
// Per-pose multipliers and offsets (via IBlendshapeSolver)
// ---------------------------------------------------------------------------

// Named access (by blendshape pose name).
A2F_API A2FResult a2f_get_multiplier(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const char*   pose_name,
    float*        out_value
);

A2F_API A2FResult a2f_set_multiplier(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const char*   pose_name,
    float         value
);

A2F_API A2FResult a2f_get_offset(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const char*   pose_name,
    float*        out_value
);

A2F_API A2FResult a2f_set_offset(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const char*   pose_name,
    float         value
);

// Batch access (all poses at once).
// The buffer must hold at least a2f_get_num_poses() floats.
A2F_API A2FResult a2f_get_multipliers(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    float*        out_values,
    size_t        count
);

A2F_API A2FResult a2f_set_multipliers(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const float*  values,
    size_t        count
);

A2F_API A2FResult a2f_get_offsets(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    float*        out_values,
    size_t        count
);

A2F_API A2FResult a2f_set_offsets(
    A2FSession    session,
    size_t        track,
    A2FSolverType solver,
    const float*  values,
    size_t        count
);

#ifdef __cplusplus
}
#endif

#endif /* A2F_BRIDGE_H */
