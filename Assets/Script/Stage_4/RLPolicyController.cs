using UnityEngine;
using Unity.InferenceEngine;

// RLPolicyController
// Stage 4 — Closed-loop RL bias controller.
//
// At each inference step:
//   1. Read normalized state from RLFeatureExtractor.
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
    // Loaded from scaler JSON through RLFeatureExtractor.
    private int inferenceInterval => RLFeatureExtractor.InferenceInterval;

    [Header("Stability Guardrails")]
    [Tooltip("Loaded from scaler JSON at runtime. Minimum |delta_bias| to apply an update.")]
    private float deadZone => RLFeatureExtractor.DeadZone;

    [Tooltip("Loaded from scaler JSON at runtime. Minimum frames between consecutive bias changes.")]
    private int dwellFrames => RLFeatureExtractor.DwellFrames;

    [Tooltip("Loaded from scaler JSON at runtime. Maximum bias delta magnitude per step.")]
    private float maxActionDelta => RLFeatureExtractor.MaxActionDelta;

    private float biasMin => RLFeatureExtractor.BiasMin;

    private float biasMax => RLFeatureExtractor.BiasMax;

    [Header("Cumulative Recovery Assist")]
    [Tooltip("Only accumulate upward recovery force while lodBias is at or below this value.")]
    public float recoveryEligibleBias = 0.85f;

    [Tooltip("Only accumulate downward correction force while lodBias is at or above this value.")]
    public float correctionEligibleBias = 1.45f;

    [Tooltip("Treat positive actions below this threshold as too weak to count as real recovery.")]
    [Range(0.0001f, 0.10f)]
    public float recoveryGrowthThreshold = 0.02f;

    [Tooltip("Number of consecutive negative or too-weak outputs before the upward recovery scalar grows.")]
    [Range(1, 30)]
    public int recoveryTriggerFrames = 5;

    [Tooltip("Multiplier applied to the cumulative upward recovery scalar each time the trigger fires.")]
    [Range(1.0f, 3.0f)]
    public float recoveryForceMultiplier = 1.35f;

    [Tooltip("Maximum value for the cumulative upward recovery scalar.")]
    [Range(1.0f, 10.0f)]
    public float recoveryForceMax = 4.0f;

    [Tooltip("Base positive delta injected when cumulative recovery assist is active.")]
    [Range(0.0001f, 0.10f)]
    public float recoveryBoostBase = 0.02f;

    [Tooltip("Reset the cumulative upward recovery scalar once GPU time reaches target minus this margin.")]
    [Range(0.0f, 1.0f)]
    public float recoveryBudgetResetMargin = 0.10f;

    [Header("Debug")]
    public bool logActions = false;

    [Header("Status (read-only)")]
    [SerializeField] private float _currentBias  = 1.0f;
    [SerializeField] private float _rawDelta     = 0f;
    [SerializeField] private string _activeMode  = "uninitialized";
    [SerializeField] private int _consecutiveWeakUpwardFrames = 0;
    [SerializeField] private float _upwardRecoveryScalar = 1.0f;
    [SerializeField] private int _consecutiveWeakDownwardFrames = 0;
    [SerializeField] private float _downwardCorrectionScalar = 1.0f;

    // ── Private State ──────────────────────────────────────────────────────

    private RLFeatureExtractor _extractor;
    private RLRolloutLogger    _logger;

    private Worker _worker;
    private Model  _model;

    private int _lastSwitchFrame;
    private bool _targetLogged = false;

    private const int INPUT_DIM = RLFeatureExtractor.FEATURE_COUNT;
    public bool UsesRuleBasedFallback => onnxAsset == null;

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
        _consecutiveWeakUpwardFrames = 0;
        _upwardRecoveryScalar = 1.0f;
        _consecutiveWeakDownwardFrames = 0;
        _downwardCorrectionScalar = 1.0f;
    }

    void Update()
    {
        // Reset logger handshakes every frame.
        // LastActionDelta must return to 0 on frames where no action is applied, and
        // ShouldLogDecisionFrame gates the rollout logger so it records only actual
        // controller evaluation points instead of every render frame.
        if (_logger != null)
        {
            _logger.LastActionDelta = 0f;
            _logger.ShouldLogDecisionFrame = false;
        }

        if (!_extractor.IsReady) return;
        LogTargetContractOnce();
        if (Time.frameCount % inferenceInterval != 0) return;

        float gpuMs = _extractor.GpuFrameTime;
        float cpuMs = _extractor.CpuFrameTime;
        if (gpuMs <= 0f || cpuMs <= 0f) return; // skip invalid frames

        if (_logger != null) _logger.ShouldLogDecisionFrame = true;

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
        _consecutiveWeakUpwardFrames = 0;
        _upwardRecoveryScalar = 1.0f;
        _consecutiveWeakDownwardFrames = 0;
        _downwardCorrectionScalar = 1.0f;

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
        float gpuMs = _extractor != null ? _extractor.GpuFrameTime : 0f;
        float assistedDelta = ApplyCumulativeRecoveryAssist(delta, gpuMs, QualitySettings.lodBias);

        // 1. Dead zone — discard actions too small to matter.
        //    Without this, the policy exploits GPU timing noise to produce
        //    rapid micro-oscillations that score well on return but are visually broken.
        if (Mathf.Abs(assistedDelta) < deadZone)
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
        float newBias  = Mathf.Clamp(current + assistedDelta, biasMin, biasMax);
        float applied  = newBias - current; // actual delta after clamp

        // If clamp collapsed the delta to below dead zone, skip the update.
        if (Mathf.Abs(applied) < deadZone)
        {
            if (_logger != null) _logger.LastActionDelta = 0f;
            return;
        }

        // 4. Apply
        // Snapshot bias BEFORE the action is applied (for accurate lod_bias_before_action in CSV)
        if (_logger != null)
            _logger.LastBiasBeforeAction = current;

        QualitySettings.lodBias = newBias;
        _currentBias            = newBias;
        _lastSwitchFrame        = Time.frameCount;

        // 5. Hand off to logger (must be set before LateUpdate reads it)
        if (_logger != null)
            _logger.LastActionDelta = applied;

        if (logActions)
            Debug.Log($"[RLPolicyController] Bias {current:F3} → {newBias:F3} " +
                      $"(raw={delta:F3} assisted={assistedDelta:F3} applied={applied:F3} " +
                      $"upScalar={_upwardRecoveryScalar:F2} upWeak={_consecutiveWeakUpwardFrames} " +
                      $"downScalar={_downwardCorrectionScalar:F2} downWeak={_consecutiveWeakDownwardFrames} mode={_activeMode})");
    }

    private float ApplyCumulativeRecoveryAssist(float rawDelta, float gpuMs, float currentBias)
    {
        float targetMs = SelectedTargetMs;
        float upwardResetGpuThreshold = targetMs - recoveryBudgetResetMargin;
        float downwardResetGpuThreshold = targetMs + recoveryBudgetResetMargin;
        bool upwardBudgetReached = gpuMs >= upwardResetGpuThreshold;
        bool downwardBudgetReached = gpuMs <= downwardResetGpuThreshold;
        bool lowBias = currentBias <= recoveryEligibleBias;
        bool highBias = currentBias >= correctionEligibleBias;

        if (upwardBudgetReached || !lowBias)
        {
            _consecutiveWeakUpwardFrames = 0;
            _upwardRecoveryScalar = 1.0f;
        }
        else
        {
            bool weakUpwardSignal = rawDelta < recoveryGrowthThreshold;
            if (weakUpwardSignal)
            {
                _consecutiveWeakUpwardFrames++;
                if (_consecutiveWeakUpwardFrames >= recoveryTriggerFrames)
                {
                    _upwardRecoveryScalar = Mathf.Min(
                        recoveryForceMax,
                        _upwardRecoveryScalar * recoveryForceMultiplier
                    );
                    _consecutiveWeakUpwardFrames = 0;
                }
            }
            else
            {
                _consecutiveWeakUpwardFrames = 0;
            }
        }

        if (downwardBudgetReached || !highBias)
        {
            _consecutiveWeakDownwardFrames = 0;
            _downwardCorrectionScalar = 1.0f;
        }
        else
        {
            bool weakDownwardSignal = rawDelta > -recoveryGrowthThreshold;
            if (weakDownwardSignal)
            {
                _consecutiveWeakDownwardFrames++;
                if (_consecutiveWeakDownwardFrames >= recoveryTriggerFrames)
                {
                    _downwardCorrectionScalar = Mathf.Min(
                        recoveryForceMax,
                        _downwardCorrectionScalar * recoveryForceMultiplier
                    );
                    _consecutiveWeakDownwardFrames = 0;
                }
            }
            else
            {
                _consecutiveWeakDownwardFrames = 0;
            }
        }

        float assistedDelta = rawDelta;

        if (_upwardRecoveryScalar > 1.0f)
        {
            float positiveAssist = recoveryBoostBase * _upwardRecoveryScalar;
            assistedDelta = Mathf.Max(assistedDelta, 0f) + positiveAssist;
        }

        if (_downwardCorrectionScalar > 1.0f)
        {
            float negativeAssist = recoveryBoostBase * _downwardCorrectionScalar;
            assistedDelta = Mathf.Min(assistedDelta, 0f) - negativeAssist;
        }

        return Mathf.Clamp(assistedDelta, -maxActionDelta, maxActionDelta);
    }

    private void LogTargetContractOnce()
    {
        if (_targetLogged) return;
        _targetLogged = true;

        Debug.Log($"[RLPolicyController] Target contract mode={_activeMode} " +
                  $"selected_source={SelectedTargetSource} selected_target={SelectedTargetMs:F3}ms " +
                  $"json_target={RLFeatureExtractor.TTargetMs:F3}ms scene_target={RLFeatureExtractor.SceneTTargetMs:F3}ms " +
                  $"scene_ready={RLFeatureExtractor.SceneTargetReady}.");
    }

    private float SelectedTargetMs => _extractor != null ? _extractor.SelectedTargetMs : RLFeatureExtractor.TTargetMs;
    private string SelectedTargetSource => _extractor != null ? _extractor.SelectedTargetSource : "json_training";
}
