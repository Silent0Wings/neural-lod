using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;

/// <summary>
/// LODFeatureExtractor
/// Collects the 12 runtime features used during training and applies
/// StandardScaler normalization using constants from scaler_constants.json.
///
/// Feature order (must match FEATURE_COLS from training exactly):
///   0   cam_rot_y
///   1   screen_coverage
///   2   visible_renderer_count
///   3   cam_pos_y
///   4   triangle_count
///   5   path_progress
///   6   draw_call_estimate
///   7   camera_velocity
///   8   gpu_frame_time_ms
///   9   cam_pos_x
///   10  cam_pos_z
///   11  fps
/// </summary>
public class LODFeatureExtractor : MonoBehaviour
{
    
    // Inspector
    
    [Header("References")]
    public Camera targetCamera;
    public CameraPathAnimator cameraPath;

    [Header("Scaler JSON")]
    [Tooltip("Path to scaler_constants.json inside StreamingAssets")]
    public string scalerJsonFileName = "scaler_constants.json";

    [Header("Coverage Sampling")]
    [Tooltip("Sample visible renderers every N frames (performance)")]
    public int coverageSampleInterval = 4;

    
    // Public output — read by NeuralLODController
    
    public float[] NormalizedFeatures { get; private set; } = new float[12];
    public bool    IsReady            { get; private set; } = false;

    
    // Scaler constants loaded from JSON
    
    private float[] _scalerMean  = new float[12];
    private float[] _scalerScale = new float[12];
    private float   _biasMin     = 0.25f;
    private float   _biasMax     = 2.0f;

    
    // Frame timing
    
    private FrameTiming[] _frameTimings = new FrameTiming[1];

    
    // Profiler recorders
    
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    
    

    
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

        if (cameraPath == null)
            cameraPath = FindFirstObjectByType<CameraPathAnimator>();

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
    }

    void Update()
    {
        CollectRawFeatures(out float camRotY, out float coverage, out float visCount,
                           out float camPosY, out float tris, out float pathProgress,
                           out float drawCalls, out float vel, out float gpu,
                           out float camPosX, out float camPosZ, out float fps);

        UpdateCoverageCache(ref visCount, ref coverage);

        float[] raw = new float[12]
        {
            camRotY,
            coverage,
            visCount,
            camPosY,
            tris,
            pathProgress,
            drawCalls,
            vel,
            gpu,
            camPosX,
            camPosZ,
            fps
        };

        // Apply StandardScaler: (x - mean) / scale
        for (int i = 0; i < 12; i++)
            NormalizedFeatures[i] = (raw[i] - _scalerMean[i]) / _scalerScale[i];

        IsReady = true;
    }

    
    // Feature collection
    
    private void CollectRawFeatures(
        out float camRotY, out float coverage, out float visCount,
        out float camPosY, out float tris, out float pathProgress,
        out float drawCalls, out float vel, out float gpu,
        out float camPosX, out float camPosZ, out float fps)
    {
        // Frame timing
        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        gpu = count > 0 ? (float)_frameTimings[0].gpuFrameTime : 0f;

        // Triangle and draw call counts
        tris      = _trisRecorder.Valid     ? (float)_trisRecorder.LastValue     : 0f;
        drawCalls = _drawCallRecorder.Valid ? (float)_drawCallRecorder.LastValue : 0f;

        // Camera linear velocity
        vel = targetCamera != null ? targetCamera.velocity.magnitude : 0f;

        // Position and rotation
        camPosX = targetCamera != null ? targetCamera.transform.position.x : 0f;
        camPosY = targetCamera != null ? targetCamera.transform.position.y : 0f;
        camPosZ = targetCamera != null ? targetCamera.transform.position.z : 0f;
        camRotY = targetCamera != null ? targetCamera.transform.eulerAngles.y : 0f;

        // FPS
        fps = Time.deltaTime > 0f ? 1f / Time.deltaTime : 60f;

        // Path progress
        pathProgress = cameraPath != null ? cameraPath.PathProgress : 0f;

        // Cached visibility
        visCount = _cachedVisibleCount;
        coverage = _cachedScreenCoverage;
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

        if (data.mean == null || data.mean.Length != 12)
        {
            Debug.LogError("[LODFeatureExtractor] Invalid scaler JSON — expected 12 mean values.");
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
