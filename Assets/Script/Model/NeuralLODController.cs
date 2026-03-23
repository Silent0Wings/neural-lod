using UnityEngine;
using Unity.InferenceEngine;

/// <summary>
/// NeuralLODController
/// Runs inference every N frames using the trained MLP (lod_mlp_single.onnx)
/// and applies the predicted LOD bias to QualitySettings with stability controls.
///
/// Requires LODFeatureExtractor on the same GameObject.
/// </summary>
public class NeuralLODController : MonoBehaviour
{
    
    // Inspector
    
    [Header("Model")]
    public ModelAsset onnxAsset;

    [Header("Inference Settings")]
    [Tooltip("Run inference every N frames to reduce overhead")]
    public int inferenceInterval = 5;

    [Header("Stability Controls")]
    [Tooltip("Minimum bias increase to trigger an upward switch")]
    public float hysteresisUp   = 0.10f;
    [Tooltip("Minimum bias decrease to trigger a downward switch")]
    public float hysteresisDown = 0.15f;
    [Tooltip("Minimum frames between bias changes")]
    public int   dwellFrames    = 30;
    [Tooltip("Maximum bias change per update")]
    public float maxDelta       = 0.25f;

    [Header("Debug")]
    public bool logPredictions = false;

    
    // Read-only status
    
    [Header("Status (read-only)")]
    [SerializeField] private float _currentBias    = 1.0f;
    [SerializeField] private float _predictedBias  = 1.0f;
    [SerializeField] private float _currentLR      = 0f;

    
    // Private
    
    private Worker             _worker;
    private Model              _model;
    private LODFeatureExtractor _extractor;

    private int   _lastSwitchFrame = 0;
    private const int INPUT_DIM    = 12;

    
    // Lifecycle
    
    void Awake()
    {
        _extractor = GetComponent<LODFeatureExtractor>();

        if (_extractor == null)
        {
            Debug.LogError("[NeuralLODController] LODFeatureExtractor not found on this GameObject.");
            enabled = false;
            return;
        }

        if (onnxAsset == null)
        {
            Debug.LogError("[NeuralLODController] No ONNX model asset assigned.");
            enabled = false;
            return;
        }

        _model  = ModelLoader.Load(onnxAsset);
        _worker = new Worker(_model, BackendType.CPU);

        _currentBias = QualitySettings.lodBias;
        Debug.Log("[NeuralLODController] Initialized. Starting bias: " + _currentBias);
    }

    void Update()
    {
        if (!_extractor.IsReady) return;
        if (Time.frameCount % inferenceInterval != 0) return;

        float predicted = RunInference(_extractor.NormalizedFeatures);
        _predictedBias  = predicted;

        ApplyWithStabilityControls(predicted);
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }


    // Inference

    private float RunInference(float[] features)
    {
        using var inputTensor = new Tensor<float>(new TensorShape(1, INPUT_DIM), features);
        _worker.Schedule(inputTensor);

        var outputTensor = _worker.PeekOutput("lod_bias_normalized") as Tensor<float>;
        using var cpuTensor = outputTensor.ReadbackAndClone();

        float normalized = cpuTensor[0];
        float denormalized = normalized * (_extractor.BiasMax - _extractor.BiasMin) + _extractor.BiasMin;
        denormalized = Mathf.Clamp(denormalized, _extractor.BiasMin, _extractor.BiasMax);

        if (logPredictions)
            Debug.Log($"[NeuralLODController] Raw: {normalized:F4} | Bias: {denormalized:F4}");

        return denormalized;
    }

    // Stability controls

    private void ApplyWithStabilityControls(float predicted)
    {
        float current = _currentBias;
        float delta   = predicted - current;

        // Hysteresis — ignore small changes
        if (delta > 0f && delta < hysteresisUp)   return;
        if (delta < 0f && -delta < hysteresisDown) return;

        // Dwell time — minimum frames between changes
        if (Time.frameCount - _lastSwitchFrame < dwellFrames) return;

        // Max delta per update — clamp step size
        float clampedDelta = Mathf.Clamp(delta, -maxDelta, maxDelta);
        float newBias      = Mathf.Clamp(current + clampedDelta,
                                          _extractor.BiasMin,
                                          _extractor.BiasMax);

        if (Mathf.Approximately(newBias, current)) return;

        QualitySettings.lodBias = newBias;
        _currentBias            = newBias;
        _lastSwitchFrame        = Time.frameCount;

        Debug.Log($"[NeuralLODController] Bias changed: {current:F3} -> {newBias:F3} " +
                  $"(predicted: {predicted:F3})");
    }
}
