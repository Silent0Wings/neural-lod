using UnityEngine;
using System.Diagnostics;
using Unity.InferenceEngine;
using Unity.Profiling;

// Loads ONNX baker model and runs inference from camera pose.
// Normalizes input using baker_scaler_constants.json from StreamingAssets.
// Writes lastInferenceDurationMs and lastPredictedThreshold for InferenceEvaluationLogger.
public class LODThresholdPredictor : MonoBehaviour
{
    [Header("Model")]
    public ModelAsset onnxAsset;

    [Header("Scaler")]
    [Tooltip("Scaler JSON filename inside StreamingAssets")]
    public string scalerJsonFileName = "baker_scaler_constants.json";

    [Header("Throttling")]
    [Tooltip("Min camera movement in meters before re-predicting")]
    public float spatialDelta = 1.0f;
    [Tooltip("Max seconds between predictions")]
    public float maxInterval = 1.0f;

    [Header("Output Smoothing")]
    [Tooltip("EMA blend factor per prediction. 0=frozen, 1=no smoothing. 0.15 recommended.")]
    [Range(0.01f, 1.0f)]
    public float emaAlpha = 0.15f;

    [Header("Status (read-only)")]
    public bool IsReady = false;

    // written each inference call, read by InferenceEvaluationLogger
    [HideInInspector] public float lastInferenceDurationMs = 0f;
    [HideInInspector] public float lastPredictedThreshold = 0f;
    [HideInInspector] public float[] lastPrediction;

    private const int FEATURE_COUNT = 13;

    private static readonly string[] ExpectedFeatureNames = {
        "pos_x", "pos_y", "pos_z",
        "sin_pitch", "cos_pitch",
        "sin_yaw",   "cos_yaw",
        "sin_roll",  "cos_roll",
        "triangle_count", "visible_renderer_count",
        "screen_coverage", "draw_call_count"
    };

    private float[] _scalerMean = new float[FEATURE_COUNT];
    private float[] _scalerScale = new float[FEATURE_COUNT];

    ProfilerRecorder _triangleRecorder;
    ProfilerRecorder _rendererRecorder;
    ProfilerRecorder _drawCallRecorder;

    private Model _model;
    private Worker _worker;

    private Camera _mainCam;
    private Vector3 _lastPredictPos;
    private float _lastPredictTime;
    private Stopwatch _stopwatch = new Stopwatch();
    private LODGroup[] _lodGroups;

    void OnEnable()
    {
        _triangleRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _rendererRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Visible Objects Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
    }

    void OnDisable()
    {
        _triangleRecorder.Dispose();
        _rendererRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    void Start()
    {
        _mainCam = Camera.main;
        _lastPredictPos = _mainCam != null ? _mainCam.transform.position : Vector3.zero;
        _lastPredictTime = Time.time;

        _lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);

        LoadScalerConstants();

        if (IsReady)
            LoadModel();
    }

    void Update()
    {
        if (!IsReady || _mainCam == null) return;

        Vector3 pos = _mainCam.transform.position;
        float dist = Vector3.Distance(pos, _lastPredictPos);
        float elapsed = Time.time - _lastPredictTime;

        if (dist >= spatialDelta || elapsed >= maxInterval)
        {
            RunInference(pos, _mainCam.transform.eulerAngles);
            _lastPredictPos = pos;
            _lastPredictTime = Time.time;
        }
    }

    void OnDestroy()
    {
        _worker?.Dispose();
    }

    private void LoadModel()
    {
        if (onnxAsset == null)
        {
            UnityEngine.Debug.LogError("[LODThresholdPredictor] onnxAsset is not assigned.");
            return;
        }

        _model = ModelLoader.Load(onnxAsset);
        _worker = new Worker(_model, BackendType.GPUCompute);
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
            Mathf.Cos(rotation.z * Mathf.Deg2Rad),
            _triangleRecorder.Valid  ? (float)_triangleRecorder.LastValue  : 0f,
            _rendererRecorder.Valid  ? (float)_rendererRecorder.LastValue  : 0f,
            ComputeScreenCoverage(_mainCam),
            _drawCallRecorder.Valid  ? (float)_drawCallRecorder.LastValue  : 0f
        };

        float[] features = new float[FEATURE_COUNT];
        for (int i = 0; i < FEATURE_COUNT; i++)
            features[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];

        _stopwatch.Restart();

        using var inputTensor = new Tensor<float>(new TensorShape(1, FEATURE_COUNT), features);
        _worker.Schedule(inputTensor);
        var rawOutput = _worker.PeekOutput();
        if (rawOutput is not Tensor<float> outputTensor) return;
        using var cpuTensor = outputTensor.ReadbackAndClone();
        float[] rawPrediction = cpuTensor.DownloadToArray();
        _stopwatch.Stop();
        lastInferenceDurationMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

        // EMA smoothing to prevent abrupt threshold jumps at grid cell boundaries
        if (lastPrediction == null || lastPrediction.Length != rawPrediction.Length)
        {
            lastPrediction = (float[])rawPrediction.Clone();
        }
        else
        {
            for (int i = 0; i < rawPrediction.Length; i++)
                lastPrediction[i] = Mathf.Lerp(lastPrediction[i], rawPrediction[i], emaAlpha);
        }

        lastPredictedThreshold = lastPrediction.Length > 0 ? lastPrediction[0] : 0f;
    }

    private float ComputeScreenCoverage(Camera cam)
    {
        if (_lodGroups == null || _lodGroups.Length == 0 || cam == null) return 0f;

        float totalPixels = cam.pixelWidth * cam.pixelHeight;
        if (totalPixels <= 0f) return 0f;

        float coveredPixels = 0f;

        foreach (LODGroup group in _lodGroups)
        {
            if (group == null) continue;

            LOD[] lods = group.GetLODs();
            if (lods.Length == 0) continue;

            Renderer[] renderers = lods[0].renderers;
            if (renderers == null || renderers.Length == 0) continue;

            Renderer firstValid = null;
            for (int r = 0; r < renderers.Length; r++)
            {
                if (renderers[r] != null) { firstValid = renderers[r]; break; }
            }
            if (firstValid == null) continue;

            Bounds b = new Bounds(firstValid.bounds.center, Vector3.zero);
            for (int r = 0; r < renderers.Length; r++)
            {
                if (renderers[r] != null)
                    b.Encapsulate(renderers[r].bounds);
            }

            Vector3 c = b.center;
            Vector3 e = b.extents;
            Vector3[] corners = new Vector3[8]
            {
                cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y,  e.z)),
                cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y,  e.z)),
                cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y,  e.z)),
                cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y,  e.z)),
                cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y, -e.z)),
                cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y, -e.z)),
                cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y, -e.z)),
                cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y, -e.z))
            };

            bool allBehind = true;
            foreach (Vector3 corner in corners)
                if (corner.z > 0f) { allBehind = false; break; }
            if (allBehind) continue;

            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            foreach (Vector3 corner in corners)
            {
                if (corner.z <= 0f) continue;
                if (corner.x < minX) minX = corner.x;
                if (corner.x > maxX) maxX = corner.x;
                if (corner.y < minY) minY = corner.y;
                if (corner.y > maxY) maxY = corner.y;
            }

            minX = Mathf.Max(0f, minX);
            minY = Mathf.Max(0f, minY);
            maxX = Mathf.Min(cam.pixelWidth,  maxX);
            maxY = Mathf.Min(cam.pixelHeight, maxY);

            coveredPixels += Mathf.Max(0f, maxX - minX) * Mathf.Max(0f, maxY - minY);
        }

        return Mathf.Clamp01(coveredPixels / totalPixels);
    }

    private void LoadScalerConstants()
    {
        string path = System.IO.Path.Combine(Application.streamingAssetsPath, scalerJsonFileName);

        if (!System.IO.File.Exists(path))
        {
            UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scaler JSON not found at: {path}");
            return;
        }

        string json = System.IO.File.ReadAllText(path);
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
            UnityEngine.Debug.LogError("[LODThresholdPredictor] feature_names mismatch.");
            return;
        }

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (data.feature_names[i] != ExpectedFeatureNames[i])
            {
                UnityEngine.Debug.LogError($"[LODThresholdPredictor] Feature order mismatch at index {i}. " +
                                           $"Expected '{ExpectedFeatureNames[i]}' got '{data.feature_names[i]}'. Aborting.");
                return;
            }
        }

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (Mathf.Approximately(data.scale[i], 0f))
            {
                UnityEngine.Debug.LogError($"[LODThresholdPredictor] Scale[{i}] ('{ExpectedFeatureNames[i]}') is zero. Aborting.");
                return;
            }
        }

        _scalerMean = data.mean;
        _scalerScale = data.scale;

        IsReady = true;
        UnityEngine.Debug.Log("[LODThresholdPredictor] Scaler loaded OK. Predictor is READY.");
    }

    [System.Serializable]
    private class ScalerData
    {
        public string[] feature_names;
        public float[] mean;
        public float[] scale;
    }
}
