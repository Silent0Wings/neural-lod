using UnityEngine;
using Unity.InferenceEngine;

// RLPolicyController
// Stage 4 — Closed-loop RL bias controller.
//
// At each inference step:
//   1. Read 11-dim normalized state from RLFeatureExtractor.
//   2. Run ONNX policy → raw bias delta in [-0.20, +0.20].
//   3. Apply stability guardrails: dead zone → dwell guard → bias clamp.
//   4. Write applied delta to RLRolloutLogger.LastActionDelta for CSV logging.
//
// Guardrails (VAR_*):
//   Dead zone   : |delta| < 0.02 → no update (prevents micro-oscillation on GPU noise).
//   Dwell frames: min frames between consecutive changes (default 5, range 5–10).
//   Bias clamp  : [0.30, 2.00] — below 0.30 the scene becomes unrecognisable.
//   Action range: [-0.20, +0.20] clamped from raw model output.
//
// Fallback policy (no ONNX assigned):
//   Rule-based baseline matching the project spec:
//     if fps < 45 → decrease bias by 0.05 (clamped to guardrails)
//     else        → increase bias by 0.05 (clamped to guardrails)
//   Use for initial data collection before any ONNX policy is trained.
//
// Note: Training uses a custom PyTorch REINFORCE notebook (not Unity ML-Agents CLI).
//       The ONNX exported from that notebook is loaded here via Unity Inference Engine.

[RequireComponent(typeof(RLFeatureExtractor))]
[RequireComponent(typeof(RLRolloutLogger))]
public class RLPolicyController : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("Model (leave null for rule-based fallback)")]
    public ModelAsset onnxAsset;

    [Header("Inference")]
    [Tooltip("Run inference every N frames.")]
    public int inferenceInterval = 2;

    [Header("Stability Guardrails")]
    [Tooltip("Minimum |delta_bias| to apply an update. Below this the action is discarded.")]
    [Range(0.01f, 0.10f)]
    public float deadZone = 0.02f;           // VAR_DEAD_ZONE

    [Tooltip("Minimum frames between consecutive bias changes.")]
    [Range(5, 10)]
    public int dwellFrames = 5;              // VAR_DWELL_FRAMES

    [Tooltip("Maximum bias delta magnitude per step.")]
    [Range(0.01f, 0.20f)]
    public float maxActionDelta = 0.20f;     // VAR_ACTION_RANGE upper bound

    [Tooltip("Absolute minimum lodBias. Below 0.30 the scene becomes visually broken.")]
    public float biasMin = 0.30f;            // VAR_BIAS_CLAMP lower

    [Tooltip("Absolute maximum lodBias.")]
    public float biasMax = 2.00f;            // VAR_BIAS_CLAMP upper

    [Header("Debug")]
    public bool logActions = false;

    [Header("Status (read-only)")]
    [SerializeField] private float _currentBias  = 1.0f;
    [SerializeField] private float _rawDelta     = 0f;
    [SerializeField] private string _activeMode  = "uninitialized";

    // ── Private State ──────────────────────────────────────────────────────

    private RLFeatureExtractor _extractor;
    private RLRolloutLogger    _logger;

    private Worker _worker;
    private Model  _model;

    private int _lastSwitchFrame;

    private const int INPUT_DIM = RLFeatureExtractor.FEATURE_COUNT; // 11

    // ── Lifecycle ──────────────────────────────────────────────────────────

    void Awake()
    {
        _extractor = GetComponent<RLFeatureExtractor>();
        _logger    = GetComponent<RLRolloutLogger>();

        if (onnxAsset != null)
        {
            _model  = ModelLoader.Load(onnxAsset);
            _worker = new Worker(_model, BackendType.CPU);
            _activeMode = "neural";
            Debug.Log("[RLPolicyController] ONNX policy loaded (neural mode). INPUT_DIM=" + INPUT_DIM);
        }
        else
        {
            _activeMode = "rule-based";
            Debug.LogWarning("[RLPolicyController] No ONNX asset assigned. " +
                             "Using rule-based fallback (fps < 45 → reduce bias). " +
                             "Assign onnxAsset after training to switch to neural mode.");
        }

        _currentBias     = QualitySettings.lodBias;
        _lastSwitchFrame = Time.frameCount;
    }

    void Update()
    {
        if (!_extractor.IsReady) return;
        if (Time.frameCount % inferenceInterval != 0) return;

        float gpuMs = _extractor.GpuFrameTime;
        float cpuMs = _extractor.CpuFrameTime;
        if (gpuMs <= 0f || cpuMs <= 0f) return; // skip invalid frames

        float delta = _activeMode == "neural"
            ? GetNeuralDelta()
            : GetRuleBasedDelta();

        _rawDelta = delta;
        ApplyWithGuardrails(delta);
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }

    // ── Episode Reset ──────────────────────────────────────────────────────

    /// <summary>
    /// Restore lodBias to 1.0 and flush per-episode counters.
    /// Called by RLRolloutLogger.EndEpisode() or externally at episode boundary.
    /// </summary>
    public void ResetEpisode()
    {
        QualitySettings.lodBias = 1.0f;
        _currentBias            = 1.0f;
        _lastSwitchFrame        = Time.frameCount;

        if (_logger != null)
            _logger.LastActionDelta = 0f;

        Debug.Log("[RLPolicyController] Episode reset. lodBias → 1.0.");
    }

    // ── Neural Policy ──────────────────────────────────────────────────────

    private float GetNeuralDelta()
    {
        float[] features = _extractor.NormalizedFeatures;
        if (features == null || features.Length != INPUT_DIM)
        {
            Debug.LogWarning("[RLPolicyController] Feature array null or wrong length " +
                             $"({features?.Length ?? 0} vs {INPUT_DIM}). Skipping inference.");
            return 0f;
        }

        // Allocate input tensor with pre-built feature array
        using var inputTensor = new Tensor<float>(new TensorShape(1, INPUT_DIM), features);
        _worker.Schedule(inputTensor);

        var rawOutput = _worker.PeekOutput();
        if (rawOutput == null)
        {
            Debug.LogError("[RLPolicyController] Output tensor is null.");
            return 0f;
        }

        if (rawOutput is not Tensor<float> outputTensor)
        {
            Debug.LogError("[RLPolicyController] Output is not Tensor<float>: " + rawOutput.GetType().Name);
            return 0f;
        }

        float rawDelta;
        try
        {
            using var cpuTensor = outputTensor.ReadbackAndClone();
            // Index 0 = mu (action mean). Policy may also output log_sigma at index 1;
            // at inference time we use the mean for deterministic execution.
            rawDelta = cpuTensor[0];
        }
        catch (System.Exception ex)
        {
            Debug.LogError("[RLPolicyController] Readback failed: " + ex.Message);
            return 0f;
        }

        if (float.IsNaN(rawDelta) || float.IsInfinity(rawDelta))
        {
            Debug.LogWarning("[RLPolicyController] NaN/Inf in policy output. Discarding.");
            return 0f;
        }

        // Clamp to VAR_ACTION_RANGE [-0.20, +0.20]
        return Mathf.Clamp(rawDelta, -maxActionDelta, maxActionDelta);
    }

    // ── Rule-Based Fallback ────────────────────────────────────────────────

    private float GetRuleBasedDelta()
    {
        // Spec baseline: "if fps < 45 → decrease bias (simple threshold)"
        float fps = _extractor.RawFeatures != null ? _extractor.RawFeatures[2] : 60f;
        float step = 0.05f;
        return fps < 45f ? -step : step;
    }

    // ── Stability Guardrails ───────────────────────────────────────────────

    private void ApplyWithGuardrails(float delta)
    {
        // 1. Dead zone — discard actions too small to matter.
        //    Without this, the policy exploits GPU timing noise to produce
        //    rapid micro-oscillations that score well on return but are visually broken.
        if (Mathf.Abs(delta) < deadZone)
        {
            if (_logger != null) _logger.LastActionDelta = 0f;
            return;
        }

        // 2. Dwell guard — enforce minimum frames between consecutive changes.
        if (Time.frameCount - _lastSwitchFrame < dwellFrames)
        {
            if (_logger != null) _logger.LastActionDelta = 0f;
            return;
        }

        // 3. Compute candidate bias and hard-clamp to [biasMin, biasMax].
        float current  = QualitySettings.lodBias;
        float newBias  = Mathf.Clamp(current + delta, biasMin, biasMax);
        float applied  = newBias - current; // actual delta after clamp

        // If clamp collapsed the delta to below dead zone, skip the update.
        if (Mathf.Abs(applied) < deadZone)
        {
            if (_logger != null) _logger.LastActionDelta = 0f;
            return;
        }

        // 4. Apply
        QualitySettings.lodBias = newBias;
        _currentBias            = newBias;
        _lastSwitchFrame        = Time.frameCount;

        // 5. Hand off to logger (must be set before LateUpdate reads it)
        if (_logger != null)
            _logger.LastActionDelta = applied;

        if (logActions)
            Debug.Log($"[RLPolicyController] Bias {current:F3} → {newBias:F3} " +
                      $"(delta={delta:F3} applied={applied:F3} mode={_activeMode})");
    }
}
