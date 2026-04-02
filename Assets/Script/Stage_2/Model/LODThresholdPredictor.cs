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
    [Tooltip("How often screen coverage is recomputed in seconds. Independent of inference rate. 0.1 = 10hz.")]
    public float screenCoverageInterval = 0.1f;

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

    private float[] _scalerMean  = new float[FEATURE_COUNT];
    private float[] _scalerScale = new float[FEATURE_COUNT];

    ProfilerRecorder _triangleRecorder;
    ProfilerRecorder _rendererRecorder;
    ProfilerRecorder _drawCallRecorder;

    private Model          _model;
    private Worker         _worker;
    private Tensor<float>  _inputTensor; // pre-alloc once in LoadModel

    private Camera   _mainCam;
    private Vector3  _lastPredictPos;
    private float    _lastPredictTime;
    private Stopwatch _stopwatch = new Stopwatch();
    private LODGroup[] _lodGroups;

    // pre-alloc inference scratch buffers so RunInference has zero managed heap allocs
    private readonly float[] _rawBuffer      = new float[FEATURE_COUNT];
    private readonly float[] _featuresBuffer = new float[FEATURE_COUNT];

    // cached LOD renderer arrays to avoid GetLODs() allocation every inference
    private Renderer[][] _cachedRenderers;

    // pre-alloc corners buffer to avoid new Vector3[8] per LODGroup per inference
    private readonly Vector3[] _corners = new Vector3[8];

    // screen coverage is expensive (WorldToScreenPoint x8 per LODGroup).
    // cache it and recompute on its own timer so inference cost is ~0.5ms not ~56ms.
    private float _cachedScreenCoverage  = 0f;
    private float _screenCoverageLastTime = -999f;

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
        _lastPredictPos  = _mainCam != null ? _mainCam.transform.position : Vector3.zero;
        _lastPredictTime = Time.time;

        _lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);
        CacheLODRenderers();

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
        _inputTensor?.Dispose();
        _worker?.Dispose();
    }

    // cache lods[0].renderers once at Start so ComputeScreenCoverage never calls GetLODs()
    private void CacheLODRenderers()
    {
        if (_lodGroups == null || _lodGroups.Length == 0)
        {
            _cachedRenderers = new Renderer[0][];
            return;
        }

        _cachedRenderers = new Renderer[_lodGroups.Length][];

        for (int g = 0; g < _lodGroups.Length; g++)
        {
            LODGroup group = _lodGroups[g];
            if (group == null) continue;

            LOD[] lods = group.GetLODs(); // one-time alloc at startup only
            if (lods.Length == 0) continue;

            _cachedRenderers[g] = lods[0].renderers;
        }
    }

    private void LoadModel()
    {
        if (onnxAsset == null)
        {
            UnityEngine.Debug.LogError("[LODThresholdPredictor] onnxAsset is not assigned.");
            return;
        }

        _model  = ModelLoader.Load(onnxAsset);

        // CPU backend removes the GPUCompute -> CPU readback stall entirely.
        // PeekOutput() on CPU returns a directly readable tensor with no fence.
        _worker = new Worker(_model, BackendType.CPU);

        // pre-alloc the input tensor once; updated in-place each inference via indexer
        _inputTensor = new Tensor<float>(new TensorShape(1, FEATURE_COUNT));

        // pre-alloc lastPrediction so first inference never clones a raw array
        lastPrediction = new float[1];

        UnityEngine.Debug.Log("[LODThresholdPredictor] Model loaded (CPU backend, zero-alloc inference).");
    }

    private void RunInference(Vector3 position, Vector3 rotation)
    {
        // write raw values into pre-alloc buffer
        _rawBuffer[0]  = position.x;
        _rawBuffer[1]  = position.y;
        _rawBuffer[2]  = position.z;
        _rawBuffer[3]  = Mathf.Sin(rotation.x * Mathf.Deg2Rad);
        _rawBuffer[4]  = Mathf.Cos(rotation.x * Mathf.Deg2Rad);
        _rawBuffer[5]  = Mathf.Sin(rotation.y * Mathf.Deg2Rad);
        _rawBuffer[6]  = Mathf.Cos(rotation.y * Mathf.Deg2Rad);
        _rawBuffer[7]  = Mathf.Sin(rotation.z * Mathf.Deg2Rad);
        _rawBuffer[8]  = Mathf.Cos(rotation.z * Mathf.Deg2Rad);
        _rawBuffer[9]  = _triangleRecorder.Valid ? (float)_triangleRecorder.LastValue : 0f;
        _rawBuffer[10] = _rendererRecorder.Valid ? (float)_rendererRecorder.LastValue : 0f;
        // only recompute screen coverage on its own interval, not every inference tick
        if (Time.time - _screenCoverageLastTime >= screenCoverageInterval)
        {
            _cachedScreenCoverage    = ComputeScreenCoverage(_mainCam);
            _screenCoverageLastTime  = Time.time;
        }
        _rawBuffer[11] = _cachedScreenCoverage;
        _rawBuffer[12] = _drawCallRecorder.Valid ? (float)_drawCallRecorder.LastValue : 0f;

        // normalize into pre-alloc features buffer
        for (int i = 0; i < FEATURE_COUNT; i++)
            _featuresBuffer[i] = (_rawBuffer[i] - _scalerMean[i]) / _scalerScale[i];

        // upload into the pre-alloc tensor via CPU indexer -- no new Tensor alloc
        for (int i = 0; i < FEATURE_COUNT; i++)
            _inputTensor[i] = _featuresBuffer[i];

        _stopwatch.Restart();

        _worker.Schedule(_inputTensor);

        // CPU backend: PeekOutput() returns a CPU-resident tensor directly.
        // No ReadbackAndClone(), no DownloadToArray(), no GPU fence stall.
        var outputTensor = _worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null) return;

        // Block until all scheduled ops are done before reading via indexer.
        // CompleteAllPendingOperations is void and does not allocate unlike ReadbackAndClone.
        outputTensor.CompleteAllPendingOperations();

        _stopwatch.Stop();
        lastInferenceDurationMs = (float)_stopwatch.Elapsed.TotalMilliseconds;

        // EMA smoothing -- read output values directly, no managed array alloc
        int outLen = outputTensor.shape.length;
        if (lastPrediction == null || lastPrediction.Length != outLen)
            lastPrediction = new float[outLen]; // only on first call or shape change

        for (int i = 0; i < outLen; i++)
        {
            float predicted = outputTensor[i];
            lastPrediction[i] = Mathf.Lerp(lastPrediction[i], predicted, emaAlpha);
        }

        lastPredictedThreshold = lastPrediction.Length > 0 ? lastPrediction[0] : 0f;
    }

    private float ComputeScreenCoverage(Camera cam)
    {
        if (_cachedRenderers == null || _cachedRenderers.Length == 0 || cam == null) return 0f;

        float totalPixels = cam.pixelWidth * cam.pixelHeight;
        if (totalPixels <= 0f) return 0f;

        float coveredPixels = 0f;
        int   groupCount    = _lodGroups.Length;

        for (int g = 0; g < groupCount; g++)
        {
            if (_lodGroups[g] == null) continue;

            Renderer[] renderers = _cachedRenderers[g]; // cached -- no GetLODs() alloc
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

            // write into pre-alloc corners buffer -- no new Vector3[8] alloc
            _corners[0] = cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y,  e.z));
            _corners[1] = cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y,  e.z));
            _corners[2] = cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y,  e.z));
            _corners[3] = cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y,  e.z));
            _corners[4] = cam.WorldToScreenPoint(c + new Vector3( e.x,  e.y, -e.z));
            _corners[5] = cam.WorldToScreenPoint(c + new Vector3(-e.x,  e.y, -e.z));
            _corners[6] = cam.WorldToScreenPoint(c + new Vector3( e.x, -e.y, -e.z));
            _corners[7] = cam.WorldToScreenPoint(c + new Vector3(-e.x, -e.y, -e.z));

            bool allBehind = true;
            for (int i = 0; i < 8; i++)
                if (_corners[i].z > 0f) { allBehind = false; break; }
            if (allBehind) continue;

            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < 8; i++)
            {
                if (_corners[i].z <= 0f) continue;
                if (_corners[i].x < minX) minX = _corners[i].x;
                if (_corners[i].x > maxX) maxX = _corners[i].x;
                if (_corners[i].y < minY) minY = _corners[i].y;
                if (_corners[i].y > maxY) maxY = _corners[i].y;
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

        string     json = System.IO.File.ReadAllText(path);
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
