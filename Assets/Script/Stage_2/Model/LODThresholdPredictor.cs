using UnityEngine;
using System.IO;
using System.Diagnostics;
// TODO: uncomment when com.unity.ai.inference is installed
// using Unity.InferenceEngine;

// Loads ONNX baker model and runs inference from camera pose.
// Normalizes input using baker_scaler_constants.json from StreamingAssets.
// Writes lastInferenceDurationMs and lastPredictedThreshold for InferenceEvaluationLogger.
// REQUIRES: com.unity.ai.inference package installed.
[RequireComponent(typeof(InferenceEvaluationLogger))]
public class LODThresholdPredictor : MonoBehaviour
{
    [Header("Model")]
    [Tooltip("ONNX model filename inside StreamingAssets")]
    public string modelFileName = "lod_baker_model.onnx";

    [Header("Scaler")]
    [Tooltip("Scaler JSON filename inside StreamingAssets")]
    public string scalerJsonFileName = "baker_scaler_constants.json";

    [Header("Throttling")]
    [Tooltip("Min camera movement in meters before re-predicting")]
    public float spatialDelta = 1.0f;
    [Tooltip("Max seconds between predictions")]
    public float maxInterval = 1.0f;

    [Header("Status (read-only)")]
    public bool IsReady = false;

    // written each inference call, read by InferenceEvaluationLogger
    [HideInInspector] public float   lastInferenceDurationMs = 0f;
    [HideInInspector] public float   lastPredictedThreshold  = 0f;
    [HideInInspector] public float[] lastPrediction;

    // scaler constants, 9 features: pos_x, pos_y, pos_z, sin/cos of rot x/y/z
    private const int FEATURE_COUNT = 9;

    private static readonly string[] ExpectedFeatureNames = {
        "pos_x", "pos_y", "pos_z",
        "sin_pitch", "cos_pitch",
        "sin_yaw",   "cos_yaw",
        "sin_roll",  "cos_roll"
    };

    private float[] _scalerMean  = new float[FEATURE_COUNT];
    private float[] _scalerScale = new float[FEATURE_COUNT];

    // TODO: uncomment when package is available
    // ModelAsset _modelAsset;
    // Worker _worker;

    private Camera  _mainCam;
    private Vector3 _lastPredictPos;
    private float   _lastPredictTime;
    private Stopwatch _stopwatch = new Stopwatch();

    void Start()
    {
        _mainCam         = Camera.main;
        _lastPredictPos  = _mainCam != null ? _mainCam.transform.position : Vector3.zero;
        _lastPredictTime = Time.time;

        LoadScalerConstants();

        if (IsReady)
            LoadModel();
    }

    void Update()
    {
        if (!IsReady || _mainCam == null) return;

        Vector3 pos     = _mainCam.transform.position;
        float   dist    = Vector3.Distance(pos, _lastPredictPos);
        float   elapsed = Time.time - _lastPredictTime;

        if (dist >= spatialDelta || elapsed >= maxInterval)
        {
            RunInference(pos, _mainCam.transform.eulerAngles);
            _lastPredictPos  = pos;
            _lastPredictTime = Time.time;
        }
    }

    void OnDestroy()
    {
        // TODO: uncomment when package is available
        // _worker?.Dispose();
    }

    private void LoadModel()
    {
        string modelPath = System.IO.Path.Combine(Application.streamingAssetsPath, modelFileName);
        if (!File.Exists(modelPath))
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] Model not found at: {modelPath}");
            return;
        }

        // TODO: uncomment when package is available
        // _modelAsset = ModelLoader.Load(modelPath);
        // _worker = new Worker(_modelAsset, BackendType.GPUCompute);
        UnityEngine.Debug.Log("[LODThresholdPredictor] Model loaded.");
    }

    private void RunInference(Vector3 position, Vector3 rotation)
    {
        float[] raw = new float[FEATURE_COUNT]
        {
            position.x,
            position.y,
            position.z,
            Mathf.Sin(rotation.x * Mathf.Deg2Rad),
            Mathf.Cos(rotation.x * Mathf.Deg2Rad),
            Mathf.Sin(rotation.y * Mathf.Deg2Rad),
            Mathf.Cos(rotation.y * Mathf.Deg2Rad),
            Mathf.Sin(rotation.z * Mathf.Deg2Rad),
            Mathf.Cos(rotation.z * Mathf.Deg2Rad)
        };

        // normalize using loaded scaler constants
        float[] features = new float[FEATURE_COUNT];
        for (int i = 0; i < FEATURE_COUNT; i++)
            features[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];

        _stopwatch.Restart();

        // TODO: uncomment when package is available
        // using var inputTensor = new Tensor<float>(new TensorShape(1, FEATURE_COUNT), features);
        // _worker.Schedule(inputTensor);
        // var rawOutput = _worker.PeekOutput();
        // if (rawOutput is not Tensor<float> outputTensor) return;
        // using var cpuTensor = outputTensor.ReadbackAndClone();
        // lastPrediction = cpuTensor.ToReadOnlyArray();

        _stopwatch.Stop();
        lastInferenceDurationMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

        // placeholder until model is deployed
        if (lastPrediction == null)
            lastPrediction = new float[] { 0.6f, 0.3f, 0.1f };

        // write scalar summary for logger, use first threshold as representative value
        lastPredictedThreshold = lastPrediction.Length > 0 ? lastPrediction[0] : 0f;
    }

    private void LoadScalerConstants()
    {
        string path = System.IO.Path.Combine(Application.streamingAssetsPath, scalerJsonFileName);

        if (!File.Exists(path))
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scaler JSON not found at: {path}");
            return;
        }

        string     json = File.ReadAllText(path);
        ScalerData data = JsonUtility.FromJson<ScalerData>(json);

        if (data.mean == null || data.mean.Length != FEATURE_COUNT)
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scaler mean mismatch. " +
                                       $"Expected {FEATURE_COUNT}, got {(data.mean != null ? data.mean.Length : 0)}");
            return;
        }

        if (data.scale == null || data.scale.Length != FEATURE_COUNT)
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scaler scale mismatch. " +
                                       $"Expected {FEATURE_COUNT}, got {(data.scale != null ? data.scale.Length : 0)}");
            return;
        }

        if (data.feature_names == null || data.feature_names.Length != FEATURE_COUNT)
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] feature_names mismatch.");
            return;
        }

        // verify feature order matches training exactly
        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (data.feature_names[i] != ExpectedFeatureNames[i])
            {
                UnityEngine.Debug.LogError($"[LODThresholdPredictor] Feature order mismatch at index {i}. " +
                                           $"Expected '{ExpectedFeatureNames[i]}' got '{data.feature_names[i]}'. Aborting.");
                return;
            }
        }

        // guard against zero scale divide
        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (Mathf.Approximately(data.scale[i], 0f))
            {
                UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scale[{i}] ('{ExpectedFeatureNames[i]}') is zero. Aborting.");
                return;
            }
        }

        _scalerMean  = data.mean;
        _scalerScale = data.scale;

        IsReady = true;
        UnityEngine.Debug.Log("[LODThresholdPredictor] Scaler loaded OK. Predictor is READY.");
    }

    [System.Serializable]
    private class ScalerData
    {
        public string[] feature_names;
        public float[]  mean;
        public float[]  scale;
    }
}
