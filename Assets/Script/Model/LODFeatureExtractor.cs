using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;

/// <summary>
/// LODFeatureExtractor
/// Collects the 10 runtime features used during training and applies
/// StandardScaler normalization using constants from scaler_constants.json.
///
/// Feature order (must match FEATURE_COLS from training exactly):
///   0  cpu_frame_time_ms
///   1  gpu_frame_time_ms
///   2  triangle_count
///   3  camera_velocity
///   4  camera_angular_velocity
///   5  visible_renderer_count
///   6  draw_call_estimate
///   7  frame_headroom_ms
///   8  screen_coverage
///   9  lod_bias_current
/// </summary>
public class LODFeatureExtractor : MonoBehaviour
{
    
    // Inspector
    
    [Header("References")]
    public Camera targetCamera;

    [Header("Scaler JSON")]
    [Tooltip("Path to scaler_constants.json inside StreamingAssets")]
    public string scalerJsonFileName = "scaler_constants.json";

    [Header("Coverage Sampling")]
    [Tooltip("Sample visible renderers every N frames (performance)")]
    public int coverageSampleInterval = 4;

    
    // Public output — read by NeuralLODController
    
    public float[] NormalizedFeatures { get; private set; } = new float[10];
    public bool    IsReady            { get; private set; } = false;

    
    // Scaler constants loaded from JSON
    
    private float[] _scalerMean  = new float[10];
    private float[] _scalerScale = new float[10];
    private float   _biasMin     = 0.25f;
    private float   _biasMax     = 2.0f;

    
    // Frame timing
    
    private FrameTiming[] _frameTimings = new FrameTiming[1];

    
    // Profiler recorders
    
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    
    // Angular velocity tracking
    
    private Quaternion _lastCamRotation;
    private float      _angularVelocity  = 0f;
    private bool       _skipAngularFrame = true;

    
    // Cached visibility
    
    private Renderer[] _allRenderers;
    private int        _cachedVisibleCount    = 0;
    private float      _cachedScreenCoverage  = 0f;
    private int        _coverageFrameCounter  = 0;

    
    // Lifecycle
    
    void Awake()
    {
        if (targetCamera == null)
            targetCamera = Camera.main;

        LoadScalerConstants();
        _allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
    }

    void OnEnable()
    {
        _trisRecorder     = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
    }

    void OnDisable()
    {
        _trisRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    void Start()
    {
        _lastCamRotation = targetCamera.transform.rotation;
    }

    void Update()
    {
        CollectRawFeatures(out float cpu, out float gpu, out float tris,
                           out float vel, out float angVel,
                           out float visCount, out float drawCalls,
                           out float headroom, out float coverage,
                           out float lodBias);

        UpdateAngularVelocity(ref angVel);
        UpdateCoverageCache(ref visCount, ref coverage);

        float[] raw = new float[10]
        {
            cpu,
            gpu,
            tris,
            vel,
            angVel,
            visCount,
            drawCalls,
            headroom,
            coverage,
            lodBias
        };

        // Apply StandardScaler: (x - mean) / scale
        for (int i = 0; i < 10; i++)
            NormalizedFeatures[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];

        IsReady = true;
    }

    
    // Feature collection
    
    private void CollectRawFeatures(
        out float cpu, out float gpu, out float tris,
        out float vel, out float angVel,
        out float visCount, out float drawCalls,
        out float headroom, out float coverage,
        out float lodBias)
    {
        // Frame timing
        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        cpu = count > 0 ? (float)_frameTimings[0].cpuFrameTime : 0f;
        gpu = count > 0 ? (float)_frameTimings[0].gpuFrameTime : 0f;

        // Triangle and draw call counts
        tris      = _trisRecorder.Valid     ? (float)_trisRecorder.LastValue     : 0f;
        drawCalls = _drawCallRecorder.Valid ? (float)_drawCallRecorder.LastValue : 0f;

        // Camera linear velocity
        vel = targetCamera != null ? targetCamera.velocity.magnitude : 0f;

        // Angular velocity — filled by UpdateAngularVelocity
        angVel = _angularVelocity;

        // Cached visibility
        visCount = _cachedVisibleCount;
        coverage = _cachedScreenCoverage;

        // Frame headroom: budget - cpu (matches training formula)
        headroom = 16.6f - cpu;

        // Current LOD bias
        lodBias = QualitySettings.lodBias;
    }

    
    // Angular velocity
    
    private void UpdateAngularVelocity(ref float angVel)
    {
        if (_skipAngularFrame)
        {
            _lastCamRotation  = targetCamera.transform.rotation;
            _skipAngularFrame = false;
            _angularVelocity  = 0f;
            angVel            = 0f;
            return;
        }

        Quaternion delta = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRotation);
        delta.ToAngleAxis(out float angle, out _);
        _angularVelocity = angle / Time.deltaTime;
        _lastCamRotation = targetCamera.transform.rotation;
        angVel           = _angularVelocity;
    }

    
    // Visibility and screen coverage cache
    
    private void UpdateCoverageCache(ref float visCount, ref float coverage)
    {
        _coverageFrameCounter++;
        if (_coverageFrameCounter < coverageSampleInterval)
        {
            visCount = _cachedVisibleCount;
            coverage = _cachedScreenCoverage;
            return;
        }

        _coverageFrameCounter = 0;

        Plane[] frustum = GeometryUtility.CalculateFrustumPlanes(targetCamera);
        float   screenW = Screen.width;
        float   screenH = Screen.height;
        int     count   = 0;
        float   total   = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (r == null || !r.enabled) continue;
            if (!GeometryUtility.TestPlanesAABB(frustum, r.bounds)) continue;

            Vector3 sp = targetCamera.WorldToScreenPoint(r.bounds.center);
            if (sp.z < 0f) continue;

            count++;
            Vector3 mn = targetCamera.WorldToScreenPoint(r.bounds.min);
            Vector3 mx = targetCamera.WorldToScreenPoint(r.bounds.max);
            float   w  = Mathf.Abs(mx.x - mn.x) / screenW;
            float   h  = Mathf.Abs(mx.y - mn.y) / screenH;
            total += Mathf.Clamp01(w) * Mathf.Clamp01(h);
        }

        _cachedVisibleCount   = count;
        _cachedScreenCoverage = count > 0 ? total / count : 0f;

        visCount = _cachedVisibleCount;
        coverage = _cachedScreenCoverage;
    }

    
    // Scaler JSON loading
    
    private void LoadScalerConstants()
    {
        string path = Application.streamingAssetsPath + "/" + scalerJsonFileName;


        if (!File.Exists(path))
        {
            Debug.LogError($"[LODFeatureExtractor] scaler_constants.json not found at: {path}");
            return;
        }

        string   json = File.ReadAllText(path);
        ScalerData data = JsonUtility.FromJson<ScalerData>(json);

        if (data.mean == null || data.mean.Length != 10)
        {
            Debug.LogError("[LODFeatureExtractor] Invalid scaler JSON — expected 10 mean values.");
            return;
        }

        _scalerMean  = data.mean;
        _scalerScale = data.scale;
        _biasMin     = data.bias_min;
        _biasMax     = data.bias_max;

        Debug.Log("[LODFeatureExtractor] Scaler constants loaded OK.");
    }

    
    // Helper — expose bias range for denormalization in controller
    
    public float BiasMin => _biasMin;
    public float BiasMax => _biasMax;

    
    // JSON schema
    
    [System.Serializable]
    private class ScalerData
    {
        public float[] mean;
        public float[] scale;
        public float   bias_min;
        public float   bias_max;
    }
}
