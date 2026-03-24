using UnityEngine;
using Unity.InferenceEngine;

// NeuralLODController
// Runs inference using the trained MLP and applies the predicted LOD bias
// to QualitySettings with stability controls.
// Requires LODFeatureExtractor on the same GameObject.
// CRITICAL fix list applied: FAULT-19, FAULT-22, CONTRA-02, CONTRA-03,
// CONTRA-04, CRITICAL-02, LOGIC-01, LOGIC-02, LOGIC-04, LOGIC-05

public class NeuralLODController : MonoBehaviour
{
    // inspector

    [Header("Model")]
    public ModelAsset onnxAsset;

    [Header("Inference Settings")]
    [Tooltip("Run inference every N frames")]
    public int inferenceInterval = 2; // LOGIC-01 fix: was 5 frames, too slow for P95 control

    [Header("Stability Controls")]
    [Tooltip("Min bias increase to trigger an upward switch")]
    public float hysteresisUp   = 0.05f; // LOGIC-02 fix: was 0.10, too conservative
    [Tooltip("Min bias decrease to trigger a downward switch")]
    public float hysteresisDown = 0.08f; // LOGIC-02 fix: was 0.15, too conservative
    [Tooltip("Min frames between bias changes")]
    public int   dwellFrames    = 10;    // CONTRA-04 fix: was 30 frames, too slow for P95
    [Tooltip("Max bias change per update")]
    public float maxDelta       = 0.25f;

    [Header("Debug")]
    public bool logPredictions = false;

    // read-only status

    [Header("Status (read-only)")]
    [SerializeField] private float _currentBias   = 1.0f;
    [SerializeField] private float _predictedBias = 1.0f;

    // private

    private Worker              _worker;
    private Model               _model;
    private LODFeatureExtractor _extractor;

    private int _lastSwitchFrame;

    // CONTRA-02 fix: INPUT_DIM must match v4 feature count (12 not 20)
    private const int INPUT_DIM = 12;

    private int _underBudgetFrames = 0;
    private int _overBudgetFrames  = 0;

    private const int   RecoveryFramesRequired   = 60;
    private const int   OverBudgetFramesRequired = 10;
    private const float BiasStep                 = 0.25f;

    // FAULT-19: track how many frames extractor has not been ready
    private int  _notReadyFrames   = 0;
    private const int NotReadyWarnInterval = 120; // warn every 2 seconds at 60fps

    // CRITICAL-02: flag set at Awake to confirm this controller owns bias control
    private bool _controllerActive = false;

    // lifecycle

    void Awake()
    {
        _extractor = GetComponent<LODFeatureExtractor>();

        if (_extractor == null)
        {
            Debug.LogError("[NeuralLODController] LODFeatureExtractor not found.");
            enabled = false;
            return;
        }

        if (onnxAsset == null)
        {
            Debug.LogError("[NeuralLODController] No ONNX asset assigned.");
            enabled = false;
            return;
        }

        // CRITICAL-02: check for conflicting fixed-bias scripts on same GameObject
        var fixedBias = GetComponent<FixedLODBiasController>();
        if (fixedBias != null && fixedBias.enabled)
        {
            Debug.LogError("[NeuralLODController] FixedLODBiasController is active on the same " +
                           "GameObject. Disable it before enabling NeuralLODController.");
            enabled = false;
            return;
        }

        _model  = ModelLoader.Load(onnxAsset);
        _worker = new Worker(_model, BackendType.CPU);

        _currentBias = QualitySettings.lodBias;

        // LOGIC-04 fix: init to current frame so dwell guard does not fire immediately
        _lastSwitchFrame = Time.frameCount;

        _controllerActive = true;
        Debug.Log("[NeuralLODController] Initialized. Bias: " + _currentBias +
                  " | INPUT_DIM: " + INPUT_DIM);
    }

    void Update()
    {
        // FAULT-19 fix: periodic warning if extractor never becomes ready
        if (!_extractor.IsReady)
        {
            _notReadyFrames++;
            if (_notReadyFrames % NotReadyWarnInterval == 0)
                Debug.LogWarning("[NeuralLODController] LODFeatureExtractor not ready for " +
                                 _notReadyFrames + " frames.");
            return;
        }

        _notReadyFrames = 0;

        if (Time.frameCount % inferenceInterval != 0) return;

        float cpu = _extractor.CpuFrameTime;
        float gpu = _extractor.GpuFrameTime;

        if (gpu <= 0f || cpu <= 0f)
            return;

        float budget = 16.6f;

        if (cpu < budget)
        {
            _underBudgetFrames++;
            _overBudgetFrames = 0;

            if (_underBudgetFrames >= RecoveryFramesRequired)
            {
                _underBudgetFrames = 0;
                float recovered = Mathf.Min(QualitySettings.lodBias + BiasStep,
                                            _extractor.BiasMax);
                QualitySettings.lodBias = recovered;
                _currentBias            = recovered;
                _lastSwitchFrame        = Time.frameCount;
                return;
            }
        }
        else
        {
            _overBudgetFrames++;
            _underBudgetFrames = 0;

            if (_overBudgetFrames < OverBudgetFramesRequired)
                return;
        }

        float[] features = _extractor.NormalizedFeatures;
        if (features == null || features.Length != INPUT_DIM)
        {
            Debug.LogWarning("[NeuralLODController] Feature array is null or wrong length: " +
                             (features?.Length.ToString() ?? "null") +
                             " (expected " + INPUT_DIM + ").");
            return;
        }

        float predicted = RunInference(features);
        if (float.IsNaN(predicted)) return; // FAULT-22 guard propagated from inference

        _predictedBias = predicted;
        ApplyWithStabilityControls(predicted);
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }

    // inference

    private float RunInference(float[] features)
    {
        using var inputTensor = new Tensor<float>(new TensorShape(1, INPUT_DIM), features);
        _worker.Schedule(inputTensor);

        // FAULT-22 fix: null-check output tensor before cast and readback
        var rawOutput = _worker.PeekOutput("lod_bias_normalized");
        if (rawOutput == null)
        {
            Debug.LogError("[NeuralLODController] Output tensor 'lod_bias_normalized' is null.");
            return float.NaN;
        }

        // LOGIC-05 fix: explicit type check before cast
        if (rawOutput is not Tensor<float> outputTensor)
        {
            Debug.LogError("[NeuralLODController] Output tensor is not Tensor<float>. " +
                           "Actual type: " + rawOutput.GetType().Name);
            return float.NaN;
        }

        // FAULT-22 fix: ReadbackAndClone inside try-catch, dispose clone after read
        float normalized;
        try
        {
            using var cpuTensor = outputTensor.ReadbackAndClone();
            normalized = cpuTensor[0];
        }
        catch (System.Exception ex)
        {
            Debug.LogError("[NeuralLODController] Readback failed: " + ex.Message);
            return float.NaN;
        }

        float denormalized = normalized * (_extractor.BiasMax - _extractor.BiasMin) +
                             _extractor.BiasMin;
        denormalized = Mathf.Clamp(denormalized, _extractor.BiasMin, _extractor.BiasMax);

        if (logPredictions)
            Debug.Log("[NeuralLODController] Raw: " + normalized.ToString("F4") +
                      " Bias: " + denormalized.ToString("F4"));

        return denormalized;
    }

    // stability controls

    private void ApplyWithStabilityControls(float predicted)
    {
        float current = QualitySettings.lodBias;
        _currentBias  = current;

        float delta = predicted - current;

        // hysteresis: ignore small changes
        if (delta > 0f && delta < hysteresisUp)   return;
        if (delta < 0f && -delta < hysteresisDown) return;

        // dwell time
        if (Time.frameCount - _lastSwitchFrame < dwellFrames) return;

        // CONTRA-03 fix: clamp delta BEFORE computing newBias, then clamp newBias to range.
        // Original code computed newBias from clamped delta, which could still violate
        // the hysteresis intent if maxDelta < hysteresis threshold.
        float clampedDelta = Mathf.Clamp(delta, -maxDelta, maxDelta);

        // re-check hysteresis on the clamped delta so a maxDelta-truncated step that
        // falls below hysteresis threshold does not produce a spurious tiny change
        if (clampedDelta > 0f && clampedDelta < hysteresisUp)   return;
        if (clampedDelta < 0f && -clampedDelta < hysteresisDown) return;

        float newBias = Mathf.Clamp(current + clampedDelta,
                                    _extractor.BiasMin,
                                    _extractor.BiasMax);

        if (Mathf.Approximately(newBias, current)) return;

        QualitySettings.lodBias = newBias;
        _currentBias            = newBias;
        _lastSwitchFrame        = Time.frameCount;

        Debug.Log("[NeuralLODController] Bias " + current.ToString("F3") +
                  " -> " + newBias.ToString("F3") +
                  " (predicted: " + predicted.ToString("F3") + ")");
    }
}