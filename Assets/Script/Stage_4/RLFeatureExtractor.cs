using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;

// RLFeatureExtractor
// Stage 4 — Collects the 13-feature state vector for the RL agent.
// Exposes both raw (RawFeatures) and normalized (NormalizedFeatures) arrays.
// Scaler loaded from rl_scaler_constants.json in StreamingAssets.
//
// Feature index | Name                    | Source                         | Notes
// 0             | cpu_frame_time          | FrameTimingManager             | ms
// 1             | gpu_frame_time          | FrameTimingManager             | ms — GPU ONLY. Never use max(cpu,gpu). Scene is CPU-bound from scripting overhead.
// 2             | fps                     | 1 / Time.deltaTime
// 3             | visible_renderer_count  | Frustum-based manual count     | URP "Visible Objects Count" profiler string returns zero silently on Unity 6000.3.10f1.
// 4             | triangle_count          | ProfilerRecorder Render
// 5             | draw_call_count         | ProfilerRecorder Render
// 6             | camera_speed            | Transform-delta / deltaTime    | m/s
// 7             | camera_rotation_speed   | Angle-axis delta / deltaTime   | deg/s
// 8             | avg_screen_coverage     | Active LOD renderers only      | ParticleSystemRenderer excluded.
// 9             | previous_bias           | QualitySettings.lodBias prior frame
// 10            | recent_lod_switch_count | Ring buffer over lodSwitchWindow frames
// 11            | floor_dwell_score       | Decaying accumulator near BIAS_MIN
// 12            | ceiling_dwell_score     | Decaying accumulator near BIAS_MAX

public class RLFeatureExtractor : MonoBehaviour
{
    public const int FEATURE_COUNT = 13;

    private static readonly string[] ExpectedFeatureNames =
    {
        "cpu_frame_time",
        "gpu_frame_time",
        "fps",
        "visible_renderer_count",
        "triangle_count",
        "draw_call_count",
        "camera_speed",
        "camera_rotation_speed",
        "avg_screen_coverage",
        "previous_bias",
        "recent_lod_switch_count",
        "floor_dwell_score",
        "ceiling_dwell_score"
    };

    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("References")]
    public Camera targetCamera;

    [Header("Scaler JSON")]
    [Tooltip("Filename inside StreamingAssets for the 13-feature RL scaler.")]
    public string scalerJsonFileName = "rl_scaler_constants.json";

    [Header("Coverage Sampling")]
    [Tooltip("Resample visible renderers every N frames.")]
    [Range(1, 10)]
    public int coverageSampleInterval = 10;

    [Header("LOD Switch Window")]
    [Tooltip("Ring-buffer length (frames) for counting recent LOD switches.")]
    public int lodSwitchWindow = 30;

    [Header("Bias Dwell Memory")]
    [Tooltip("Minimum expected runtime lodBias. Must match training/export guardrails.")]
    public float biasMin = 0.30f;

    [Tooltip("Maximum expected runtime lodBias. Must match training/export guardrails.")]
    public float biasMax = 2.00f;

    [Tooltip("Bias values within this margin of BIAS_MIN contribute to floor dwell.")]
    [Min(0.01f)]
    public float floorMargin = 0.18f;

    [Tooltip("Bias values within this margin of BIAS_MAX contribute to ceiling dwell.")]
    [Min(0.01f)]
    public float ceilingMargin = 0.18f;

    [Tooltip("How quickly dwell accumulators rise while bias stays near a limit.")]
    [Range(0.01f, 1.0f)]
    public float dwellAccumulationRate = 0.30f;

    [Tooltip("How much of the previous dwell score remains each frame.")]
    [Range(0.01f, 0.999f)]
    public float dwellDecay = 0.92f;

    // ── Public Output ──────────────────────────────────────────────────────

    /// <summary>Raw unnormalized features — for CSV logging.</summary>
    public float[] RawFeatures        { get; private set; } = new float[FEATURE_COUNT];

    /// <summary>StandardScaler-normalized features — for ONNX inference input.</summary>
    public float[] NormalizedFeatures { get; private set; } = new float[FEATURE_COUNT];

    public float GpuFrameTime  { get; private set; }
    public float CpuFrameTime  { get; private set; }
    public bool  IsReady       { get; private set; } = false;

    // ── Scaler Constants ───────────────────────────────────────────────────

    private float[] _scalerMean  = new float[FEATURE_COUNT];
    private float[] _scalerScale = new float[FEATURE_COUNT];

    // ── Profiler Recorders ─────────────────────────────────────────────────

    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    private FrameTiming[] _frameTimings = new FrameTiming[1];

    // ── Renderer Cache ─────────────────────────────────────────────────────

    private Renderer[] _allRenderers;
    private Plane[]    _frustumPlanes  = new Plane[6];

    // ── Coverage Cache ─────────────────────────────────────────────────────

    private int   _cachedVisibleCount   = 0;
    private float _cachedScreenCoverage = 0f;
    private int   _coverageFrameCounter = 0;

    // ── Camera Motion ──────────────────────────────────────────────────────

    private Vector3    _lastCamPos;
    private Quaternion _lastCamRot;
    private float      _cameraSpeed    = 0f;
    private float      _cameraRotSpeed = 0f;
    private bool       _skipMotionFrame = true;

    // ── LOD Switch Tracking ────────────────────────────────────────────────

    private float _previousBias      = 1.0f;
    private float _currentBias       = 1.0f;
    private int[] _switchRingBuffer;        // 1 if a switch occurred on that frame
    private int   _ringHead           = 0;
    private int   _recentSwitchCount  = 0;
    private float _floorDwellScore    = 0f;
    private float _ceilingDwellScore  = 0f;

    // ── Lifecycle ──────────────────────────────────────────────────────────

    void Awake()
    {
        if (targetCamera == null)
        {
            targetCamera = Camera.main;
            if (targetCamera == null)
            {
                Debug.LogError("[RLFeatureExtractor] No camera found. " +
                               "Assign targetCamera or tag a camera as MainCamera.");
                enabled = false;
                return;
            }
        }

        coverageSampleInterval = Mathf.Clamp(coverageSampleInterval, 1, 10);
        lodSwitchWindow        = Mathf.Max(1, lodSwitchWindow);
        biasMax                = Mathf.Max(biasMin + 0.01f, biasMax);
        floorMargin            = Mathf.Max(0.01f, floorMargin);
        ceilingMargin          = Mathf.Max(0.01f, ceilingMargin);
        _allRenderers     = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        _previousBias     = QualitySettings.lodBias;
        _currentBias      = QualitySettings.lodBias;
        _coverageFrameCounter = coverageSampleInterval; // prime an immediate first resample
        _switchRingBuffer = new int[lodSwitchWindow];

        LoadScalerConstants();
    }

    void OnEnable()
    {
        _trisRecorder     = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
        _skipMotionFrame  = true;
    }

    void OnDisable()
    {
        _trisRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    void Update()
    {
        if (!IsReady) return;

        // 1. Frame timings — GPU time is the reward signal. CPU time logged for diagnostics only.
        FrameTimingManager.CaptureFrameTimings();
        uint timingCount = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        CpuFrameTime = timingCount > 0 ? (float)_frameTimings[0].cpuFrameTime : 0f;
        GpuFrameTime = timingCount > 0 ? (float)_frameTimings[0].gpuFrameTime : 0f;

        // 2. Triangle guard — profiler needs several frames to warm up.
        long trisRaw = _trisRecorder.Valid ? _trisRecorder.LastValue : 0;
        if (trisRaw <= 0) return;

        // 3. Camera motion (transform-based, no Camera.velocity which is unreliable)
        UpdateCameraMotion();

        // 4. Coverage cache (throttled to avoid every-frame GC from frustum test)
        UpdateCoverageCache();

        // 5. LOD switch detection via ring buffer
        float newBias = QualitySettings.lodBias;
        int   switched = Mathf.Approximately(newBias, _currentBias) ? 0 : 1;

        _recentSwitchCount              -= _switchRingBuffer[_ringHead]; // evict oldest
        _switchRingBuffer[_ringHead]     = switched;
        _recentSwitchCount              += switched;                      // insert newest
        _ringHead                        = (_ringHead + 1) % lodSwitchWindow;

        _previousBias = _currentBias;
        _currentBias  = newBias;
        UpdateBiasDwellScores(_previousBias);

        // 6. Collect remaining stats
        float fps   = Time.deltaTime > 0f ? 1f / Time.deltaTime : 60f;
        long  draws = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;

        // 7. Write raw feature array (must match ExpectedFeatureNames order exactly)
        RawFeatures[0]  = CpuFrameTime;
        RawFeatures[1]  = GpuFrameTime;           // GPU ONLY — reward signal source
        RawFeatures[2]  = fps;
        RawFeatures[3]  = (float)_cachedVisibleCount;  // frustum-based, not URP profiler
        RawFeatures[4]  = (float)trisRaw;
        RawFeatures[5]  = (float)draws;
        RawFeatures[6]  = _cameraSpeed;
        RawFeatures[7]  = _cameraRotSpeed;
        RawFeatures[8]  = _cachedScreenCoverage;  // LOD renderers only, no particles/fog
        RawFeatures[9]  = _previousBias;
        RawFeatures[10] = (float)_recentSwitchCount;
        RawFeatures[11] = _floorDwellScore;
        RawFeatures[12] = _ceilingDwellScore;

        // 8. Normalize for inference
        for (int i = 0; i < FEATURE_COUNT; i++)
            NormalizedFeatures[i] = (RawFeatures[i] - _scalerMean[i]) / _scalerScale[i];
    }

    // ── Episode Reset ──────────────────────────────────────────────────────

    /// <summary>
    /// Call at episode boundary. Restores per-episode counters.
    /// lodBias is reset by RLPolicyController.ResetEpisode().
    /// </summary>
    public void ResetEpisodeState()
    {
        _previousBias      = 1.0f;
        _currentBias       = 1.0f;
        _recentSwitchCount = 0;
        _ringHead          = 0;
        _floorDwellScore   = 0f;
        _ceilingDwellScore = 0f;
        System.Array.Clear(_switchRingBuffer, 0, _switchRingBuffer.Length);
        _skipMotionFrame      = true;
        _coverageFrameCounter = coverageSampleInterval; // force immediate resample next frame
    }

    // ── Private Helpers ────────────────────────────────────────────────────

    private void UpdateCameraMotion()
    {
        if (_skipMotionFrame)
        {
            _lastCamPos      = targetCamera.transform.position;
            _lastCamRot      = targetCamera.transform.rotation;
            _skipMotionFrame = false;
            _cameraSpeed     = 0f;
            _cameraRotSpeed  = 0f;
            return;
        }

        float dt = Time.deltaTime;
        if (dt > 0f)
        {
            _cameraSpeed = Vector3.Distance(targetCamera.transform.position, _lastCamPos) / dt;

            Quaternion delta = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRot);
            delta.ToAngleAxis(out float angle, out Vector3 _);
            _cameraRotSpeed = angle / dt;
        }

        _lastCamPos = targetCamera.transform.position;
        _lastCamRot = targetCamera.transform.rotation;
    }

    private void UpdateCoverageCache()
    {
        _coverageFrameCounter++;
        if (_coverageFrameCounter < coverageSampleInterval) return;
        _coverageFrameCounter = 0;

        GeometryUtility.CalculateFrustumPlanes(targetCamera, _frustumPlanes);
        float sw    = Screen.width;
        float sh    = Screen.height;
        int   count = 0;
        float total = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (r == null || !r.enabled)              continue;
            if (r is ParticleSystemRenderer)          continue; // exclude particles
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

    private void UpdateBiasDwellScores(float bias)
    {
        float floorProximity = Mathf.Clamp01(((biasMin + floorMargin) - bias) / Mathf.Max(floorMargin, 1e-4f));
        float ceilingProximity = Mathf.Clamp01((bias - (biasMax - ceilingMargin)) / Mathf.Max(ceilingMargin, 1e-4f));

        _floorDwellScore = Mathf.Clamp01(_floorDwellScore * dwellDecay + floorProximity * dwellAccumulationRate);
        _ceilingDwellScore = Mathf.Clamp01(_ceilingDwellScore * dwellDecay + ceilingProximity * dwellAccumulationRate);
    }

    private void LoadScalerConstants()
    {
        string path = Application.streamingAssetsPath + "/" + scalerJsonFileName;

        if (!File.Exists(path))
        {
            Debug.LogError($"[RLFeatureExtractor] Scaler not found: {path}. " +
                           "Export rl_scaler_constants.json from the Python training notebook first.");
            return;
        }

        string     json = File.ReadAllText(path);
        ScalerData data = JsonUtility.FromJson<ScalerData>(json);

        if (data.mean == null || data.mean.Length != FEATURE_COUNT)
        {
            Debug.LogError($"[RLFeatureExtractor] mean length mismatch: " +
                           $"expected {FEATURE_COUNT}, got {(data.mean != null ? data.mean.Length : 0)}.");
            return;
        }

        if (data.scale == null || data.scale.Length != FEATURE_COUNT)
        {
            Debug.LogError($"[RLFeatureExtractor] scale length mismatch: " +
                           $"expected {FEATURE_COUNT}, got {(data.scale != null ? data.scale.Length : 0)}.");
            return;
        }

        if (data.feature_names == null || data.feature_names.Length != FEATURE_COUNT)
        {
            Debug.LogError("[RLFeatureExtractor] feature_names length mismatch.");
            return;
        }

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (data.feature_names[i] != ExpectedFeatureNames[i])
            {
                Debug.LogError($"[RLFeatureExtractor] Feature order mismatch at index {i}: " +
                               $"expected '{ExpectedFeatureNames[i]}', got '{data.feature_names[i]}'. " +
                               "Ensure the Python scaler was built with the same feature order. Aborting.");
                return;
            }
        }

        for (int i = 0; i < FEATURE_COUNT; i++)
        {
            if (Mathf.Approximately(data.scale[i], 0f))
            {
                Debug.LogError($"[RLFeatureExtractor] scale[{i}] ('{ExpectedFeatureNames[i]}') is zero — " +
                               "divide-by-zero in normalization. Aborting.");
                return;
            }
        }

        _scalerMean  = data.mean;
        _scalerScale = data.scale;
        IsReady      = true;

        Debug.Log("[RLFeatureExtractor] Scaler loaded OK. Extractor READY.");
    }

    [System.Serializable]
    private class ScalerData
    {
        public string[] feature_names;
        public float[]  mean;
        public float[]  scale;
    }
}
