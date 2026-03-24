using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;

/// <summary>
/// LODFeatureExtractor
/// Collects the 20 runtime features used during training and applies
/// StandardScaler normalization using constants from scaler_constants.json.
///
/// Feature order (must match ALL_FEATURES from training exactly):
///   0   cpu_frame_time_ms
///   1   gpu_frame_time_ms
///   2   triangle_count
///   3   camera_velocity
///   4   camera_angular_velocity
///   5   visible_renderer_count
///   6   draw_call_estimate
///   7   frame_headroom_ms
///   8   screen_coverage
///   9   lod_bias_current
///   10  fps
///   11  previous_bias
///   12  cam_pos_x
///   13  cam_pos_y
///   14  cam_pos_z
///   15  cam_rot_y
///   16  path_progress
///   17  waypoint_index
///   18  move_speed
///   19  rotate_speed
/// </summary>
public class LODFeatureExtractor : MonoBehaviour
{
    private const int FEATURE_COUNT = 20;

    private static readonly string[] ExpectedFeatureNames = {
        "cpu_frame_time_ms", "gpu_frame_time_ms", "triangle_count", "camera_velocity",
        "camera_angular_velocity", "visible_renderer_count", "draw_call_estimate",
        "frame_headroom_ms", "screen_coverage", "lod_bias_current", "fps",
        "previous_bias", "cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_rot_y",
        "path_progress", "waypoint_index", "move_speed", "rotate_speed"
    };

    // Inspector
    [Header("References")]
    public Camera targetCamera;
    public CameraPathAnimator cameraPath;

    [Header("Run Parameters (Inference Defaults)")]
    public float defaultMoveSpeed = 5.0f;
    public float defaultRotateSpeed = 45.0f;

    [Header("Scaler JSON")]
    [Tooltip("Path to scaler_constants.json inside StreamingAssets")]
    public string scalerJsonFileName = "scaler_constants.json";

    [Header("Coverage Sampling")]
    [Tooltip("Sample visible renderers every N frames !!!! must match MetricLogger.coverageSampleInterval")]
    public int coverageSampleInterval = 30;

    // Public output — read by NeuralLODController
    public float[] NormalizedFeatures { get; private set; } = new float[FEATURE_COUNT];
    public float   CpuFrameTime       { get; private set; }
    public float   GpuFrameTime       { get; private set; }
    public bool    IsReady            { get; private set; } = false;

    // Scaler constants loaded from JSON
    private float[] _scalerMean  = new float[FEATURE_COUNT];
    private float[] _scalerScale = new float[FEATURE_COUNT];
    private float   _biasMin     = 0.25f;
    private float   _biasMax     = 2.0f;

    // Internal state
    private FrameTiming[]    _frameTimings = new FrameTiming[1];
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;
    private Renderer[]       _allRenderers;

    private int   _cachedVisibleCount   = 0;
    private float _cachedScreenCoverage = 0f;
    private int   _coverageFrameCounter = 0;

    private float      _lastLodBias      = 1.0f;
    private Quaternion _lastCamRotation;
    private float      _angularVelocity  = 0f;
    private bool       _skipAngularFrame = true;

    private Plane[] _frustumPlanes = new Plane[6]; // pre-alloc — eliminates per-sample GC alloc

    void Awake()
    {
        if (targetCamera == null) targetCamera = Camera.main;
        if (cameraPath == null)   cameraPath   = FindFirstObjectByType<CameraPathAnimator>();

        LoadScalerConstants();
        _allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        _lastLodBias = QualitySettings.lodBias;
    }

    void OnEnable()
    {
        _trisRecorder     = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
        _skipAngularFrame = true;
    }

    void OnDisable()
    {
        _trisRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    void Update()
    {
        if (!IsReady) return;
        if (targetCamera == null) return;

        // 1. Capture Timings
        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);
        
        float cpu = count > 0 ? (float)_frameTimings[0].cpuFrameTime : 0f;
        float gpu = count > 0 ? (float)_frameTimings[0].gpuFrameTime : 0f;
        CpuFrameTime = cpu;
        GpuFrameTime = gpu;

        // 2. Compute Angular Velocity
        UpdateAngularVelocity();

        // 3. Update Visibility Cache
        UpdateCoverageCache();

        // 4. Collect Stats
        float  vel      = targetCamera.velocity.magnitude;
        float  fps      = Time.deltaTime > 0f ? 1f / Time.deltaTime : 60f;
        long   tris     = _trisRecorder.Valid ? _trisRecorder.LastValue : 0;
        long   draws    = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;
        Vector3 pos     = targetCamera.transform.position;
        float   rotY    = targetCamera.transform.eulerAngles.y;
        float   curBias = QualitySettings.lodBias;
        float   pathP   = cameraPath != null ? cameraPath.PathProgress : 0f;
        int     wayIdx  = cameraPath != null ? cameraPath.CurrentIndex : -1;

        // 5. Construct Raw Feature Array (MUST MATCH 20-FEATURE ORDER)
        float[] raw = new float[FEATURE_COUNT]
        {
            cpu,                      // 0
            gpu,                      // 1
            (float)tris,              // 2
            vel,                      // 3
            _angularVelocity,         // 4
            (float)_cachedVisibleCount,// 5
            (float)draws,             // 6
            16.6f - Mathf.Max(cpu, gpu), // 7 (frame_headroom_ms)
            _cachedScreenCoverage,    // 8
            curBias,                  // 9 (lod_bias_current)
            fps,                      // 10
            _lastLodBias,             // 11 (previous_bias)
            pos.x,                    // 12
            pos.y,                    // 13
            pos.z,                    // 14
            rotY,                     // 15
            pathP,                    // 16
            (float)wayIdx,            // 17
            defaultMoveSpeed,         // 18
            defaultRotateSpeed        // 19
        };

        // 6. Normalize
        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            NormalizedFeatures[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];
        }

        // 7. Update State
        _lastLodBias = curBias;
    }

    private void UpdateAngularVelocity()
    {
        if (_skipAngularFrame)
        {
            _lastCamRotation = targetCamera.transform.rotation;
            _skipAngularFrame = false;
            _angularVelocity = 0f;
            return;
        }
        Quaternion delta = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRotation);
        delta.ToAngleAxis(out float angle, out Vector3 _);
        _angularVelocity = angle / Time.deltaTime;
        _lastCamRotation = targetCamera.transform.rotation;
    }

    private void UpdateCoverageCache()
    {
        _coverageFrameCounter++;
        if (_coverageFrameCounter < coverageSampleInterval) return;
        _coverageFrameCounter = 0;

        // non-allocating overload — writes into pre-allocated array
        GeometryUtility.CalculateFrustumPlanes(targetCamera, _frustumPlanes);
        float sw = Screen.width;
        float sh = Screen.height;
        int count = 0;
        float total = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (r == null || !r.enabled) continue;
            if (!GeometryUtility.TestPlanesAABB(_frustumPlanes, r.bounds)) continue;
            Vector3 sp = targetCamera.WorldToScreenPoint(r.bounds.center);
            if (sp.z < 0f) continue;

            count++;
            Vector3 mn = targetCamera.WorldToScreenPoint(r.bounds.min);
            Vector3 mx = targetCamera.WorldToScreenPoint(r.bounds.max);
            float w = Mathf.Abs(mx.x - mn.x) / sw;
            float h = Mathf.Abs(mx.y - mn.y) / sh;
            total += Mathf.Clamp01(w) * Mathf.Clamp01(h);
        }
        _cachedVisibleCount   = count;
        _cachedScreenCoverage = count > 0 ? total / count : 0f;
    }

    private void LoadScalerConstants()
    {
        string path = Application.streamingAssetsPath + "/" + scalerJsonFileName;

        if (!File.Exists(path))
        {
            Debug.LogError($"[LODFeatureExtractor] scaler_constants.json not found at: {path}");
            return;
        }

        string json = File.ReadAllText(path);
        ScalerData data = JsonUtility.FromJson<ScalerData>(json);

        if (data.mean == null || data.mean.Length != FEATURE_COUNT)
        {
            Debug.LogError($"[LODFeatureExtractor] Scaler data mismatch! Expected {FEATURE_COUNT} but found {(data.mean != null ? data.mean.Length : 0)}");
            return;
        }

        if (data.feature_names == null || data.feature_names.Length != FEATURE_COUNT)
        {
            Debug.LogError($"[LODFeatureExtractor] feature_names mismatch! Expected {FEATURE_COUNT} features.");
            return;
        }

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (data.feature_names[i] != ExpectedFeatureNames[i])
            {
                Debug.LogError($"[LODFeatureExtractor] Feature order mismatch at index {i}! " +
                               $"Expected '{ExpectedFeatureNames[i]}' but found '{data.feature_names[i]}'. " +
                               "Aborting to avoid garbage inference.");
                return;
            }
        }

            for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (Mathf.Approximately(data.scale[i], 0f))
            {
                Debug.LogError($"[LODFeatureExtractor] Scale[{i}] ('{ExpectedFeatureNames[i]}') is zero — " +
                            "would cause divide-by-zero in normalization. Aborting.");
                return;
            }
        }

        _scalerMean  = data.mean;
        _scalerScale = data.scale;
        _biasMin     = data.bias_min;
        _biasMax     = data.bias_max;

        IsReady = true;
        Debug.Log("[LODFeatureExtractor] Scaler constants loaded OK. Extractor is READY.");
    }

    public float BiasMin => _biasMin;
    public float BiasMax => _biasMax;

    [System.Serializable]
    private class ScalerData
    {
        public string[] feature_names;
        public float[] mean;
        public float[] scale;
        public float   bias_min;
        public float   bias_max;
    }
}

