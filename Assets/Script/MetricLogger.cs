using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;
using System.Collections.Generic;
using System.Text;


public class MetricLogger : MonoBehaviour
{
    [Header("Settings")]
    public Camera targetCamera;
    public int bufferFlushInterval = 120; // rows buffered before disk write

    [Header("Row Cap")]
    public int targetRowCount = 5000;
    public bool loggingComplete = false;

    [Header("Coverage Sampling")]
    [Tooltip("Compute screen coverage every N frames to reduce CPU overhead")]
    public int coverageSampleInterval = 4;

    // ─
    // References — assigned by orchestrator or auto-found
    // ─
    [Header("References")]
    public CameraPathAnimator cameraPath;

    // ─
    // ProfilerRecorders — only for tris + draw calls (FIX #3)
    // ─
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    // ─
    // FrameTimingManager state (FIX #3)
    // ─
    private FrameTiming[] _frameTimings = new FrameTiming[1];

    // ─
    // FIX #2: In-memory row buffer
    // ─
    private List<string> _rowBuffer;
    private StreamWriter _writer;
    private int _frameCount = 0;
    private int _validRows = 0;
    private string _filePath;

    // ─
    // Scene / state
    // ─
    private Renderer[] _allRenderers;
    private int _runIndex = 0;
    private float _lastLodBias = 0f;
    private Quaternion _lastCamRotation;
    private float _angularVelocity = 0f;
    private float _currentMoveSpeed = 0f;
    private float _currentRotateSpeed = 0f;
    private bool _skipAngularFrame = false;

    // ─
    // FIX #8: Cached coverage values
    // ─
    private int _cachedVisibleRenderers = 0;
    private float _cachedScreenCoverage = 0f;
    private int _coverageFrameCounter = 0;

    // ─
    // Init
    // ─
    void Awake()
    {
        // FIX #7: Apply VSync/frame-rate guard
        ApplyTimingGuards();

        if (targetCamera == null)
            targetCamera = Camera.main;

        if (cameraPath == null)
            cameraPath = FindFirstObjectByType<CameraPathAnimator>();
    }

    void OnEnable()
    {
        // FIX #3: Only use ProfilerRecorder for non-timing metrics
        _trisRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
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
        _lastLodBias = QualitySettings.lodBias;
        _allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
    }

    // ─
    // FIX #7: Reusable guard — called in Awake AND ResetLogger
    // ─
    private void ApplyTimingGuards()
    {
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
    }

    // ─
    // Public reset — called by orchestrator between runs
    // ─
    public void ResetLogger(int runIndex, float moveSpeed, float rotateSpeed)
    {
        // FIX #7: Re-apply timing guards every run
        ApplyTimingGuards();

        _currentMoveSpeed = moveSpeed;
        _currentRotateSpeed = rotateSpeed;

        FlushAndCloseFile();

        _runIndex = runIndex;
        _validRows = 0;
        _frameCount = 0;
        _coverageFrameCounter = 0;
        loggingComplete = false;
        _lastLodBias = QualitySettings.lodBias;

        OpenNewFile(moveSpeed, rotateSpeed);
        enabled = true;
        _skipAngularFrame = true;
        _allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        Debug.Log($"[MetricLogger] Reset for run {_runIndex}. File: {_filePath}");
    }

    // ─
    // Open a new CSV file for this run
    // ─
    private void OpenNewFile(float moveSpeed, float rotateSpeed)
    {
        FlushAndCloseFile();

        _filePath = Application.persistentDataPath +
                $"/training_data_bias_{QualitySettings.lodBias:F1}" +
                $"_spd_{moveSpeed:F1}" +
                $"_rot_{rotateSpeed:F1}" +
                $".csv";

        _writer = new StreamWriter(_filePath, append: false);
        _rowBuffer = new List<string>(bufferFlushInterval + 16);

        // FIX #4: Added waypoint_index and path_progress columns
        _writer.WriteLine(
            "cpu_frame_time_ms," +
            "gpu_frame_time_ms," +
            "triangle_count," +
            "camera_velocity," +
            "visible_renderer_count," +
            "lod_bias_current," +
            "fps," +
            "camera_angular_velocity," +
            "draw_call_estimate," +
            "previous_bias," +
            "frame_headroom_ms," +
            "cam_pos_x," +
            "cam_pos_y," +
            "cam_pos_z," +
            "cam_rot_y," +
            "screen_coverage," +
            "move_speed," +
            "rotate_speed," +
            "waypoint_index," +       // FIX #4: discrete node index
            "path_progress," +        // FIX #4: continuous spatial key
            "target_lod_bias"         // oracle label — Phase 2.3
        );

        Debug.Log($"[MetricLogger] Logging to: {_filePath}");
    }

    // ─
    // FIX #2: Flush buffer to disk, then close
    // ─
    private void FlushAndCloseFile()
    {
        if (_writer != null)
        {
            // Write any remaining buffered rows
            if (_rowBuffer != null && _rowBuffer.Count > 0)
            {
                foreach (string row in _rowBuffer)
                    _writer.WriteLine(row);
                _rowBuffer.Clear();
            }

            _writer.Flush();
            _writer.Close();
            _writer.Dispose();
            _writer = null;
        }
    }

    // ─
    // Per-frame capture
    // ─
    void Update()
    {
        if (_writer == null) return;
        if (loggingComplete) return;

        //  FIX #3: Timing via FrameTimingManager 
        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        float cpu = 0f;
        float gpu = 0f;

        if (count > 0)
        {
            cpu = (float)_frameTimings[0].cpuFrameTime;   // already in ms
            gpu = (float)_frameTimings[0].gpuFrameTime;   // already in ms
        }

        long tris = _trisRecorder.Valid ? _trisRecorder.LastValue : 0;

        //  Camera ─
        float velocity = targetCamera != null ? targetCamera.velocity.magnitude : 0f;
        Vector3 camPos = targetCamera.transform.position;
        float camRotY = targetCamera.transform.eulerAngles.y;
        float fps = 1f / Time.deltaTime;

        UpdateAngularVelocity();

        //  FIX #8: Scene coverage cached, updated every N frames 
        _coverageFrameCounter++;
        if (_coverageFrameCounter >= coverageSampleInterval)
        {
            _coverageFrameCounter = 0;
            var (vis, cov) = CountVisibleRenderersAndCoverage();
            _cachedVisibleRenderers = vis;
            _cachedScreenCoverage = cov;
        }

        long drawCalls = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;

        //  LOD state 
        float lodBias = QualitySettings.lodBias;
        float previousBias = _lastLodBias;
        _lastLodBias = lodBias;

        //  Derived 
        float frameHeadroom = 16.6f - gpu;

        //  FIX #4: Spatial key from CameraPathAnimator 
        int waypointIndex = cameraPath != null ? cameraPath.CurrentIndex : -1;
        float pathProgress = cameraPath != null ? cameraPath.PathProgress : -1f;

        //  Validation ─
        bool validRow = cpu > 0f && gpu > 0f && tris > 0 && velocity >= 0f;
        if (!validRow) return;

        //  FIX #2: Buffer row in memory instead of writing to disk 
        string row =
            $"{cpu:F4}," +
            $"{gpu:F4}," +
            $"{tris}," +
            $"{velocity:F4}," +
            $"{_cachedVisibleRenderers}," +
            $"{lodBias:F2}," +
            $"{fps:F2}," +
            $"{_angularVelocity:F4}," +
            $"{drawCalls}," +
            $"{previousBias:F2}," +
            $"{frameHeadroom:F4}," +
            $"{camPos.x:F2}," +
            $"{camPos.y:F2}," +
            $"{camPos.z:F2}," +
            $"{camRotY:F2}," +
            $"{_cachedScreenCoverage:F4}," +
            $"{_currentMoveSpeed:F2}," +
            $"{_currentRotateSpeed:F2}," +
            $"{waypointIndex}," +          // FIX #4
            $"{pathProgress:F4}," +        // FIX #4
            $"?";                          // target_lod_bias — Phase 2.3

        _rowBuffer.Add(row);

        _frameCount++;

        // FIX #2: Batch-write to disk only at flush intervals
        if (_rowBuffer.Count >= bufferFlushInterval)
        {
            foreach (string bufferedRow in _rowBuffer)
                _writer.WriteLine(bufferedRow);
            _rowBuffer.Clear();
            _writer.Flush();
        }

        _validRows++;
        if (_validRows >= targetRowCount)
        {
            FlushAndCloseFile();
            loggingComplete = true;
            enabled = false;
            Debug.Log($"[MetricLogger] Run {_runIndex} complete — {targetRowCount} rows.");
        }
    }

    // ─
    // Count visible renderers + avg screen coverage
    // ─
    private (int count, float coverage) CountVisibleRenderersAndCoverage()
    {
        Plane[] frustum = GeometryUtility.CalculateFrustumPlanes(targetCamera);
        float screenW = Screen.width;
        float screenH = Screen.height;
        int count = 0;
        float totalArea = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (r == null || !r.enabled || !GeometryUtility.TestPlanesAABB(frustum, r.bounds))
                continue;

            Bounds b = r.bounds;
            Vector3 screenPoint = targetCamera.WorldToScreenPoint(b.center);
            if (screenPoint.z < 0) continue;
            count++;
            Vector3 min = targetCamera.WorldToScreenPoint(b.min);
            Vector3 max = targetCamera.WorldToScreenPoint(b.max);
            float w = Mathf.Abs(max.x - min.x) / screenW;
            float h = Mathf.Abs(max.y - min.y) / screenH;
            totalArea += Mathf.Clamp01(w) * Mathf.Clamp01(h);
        }

        float avgCoverage = count > 0 ? totalArea / count : 0f;
        return (count, avgCoverage);
    }

    // ─
    // Angular velocity — degrees/sec from rotation delta
    // ─
    private void UpdateAngularVelocity()
    {
        if (_skipAngularFrame)
        {
            _lastCamRotation = targetCamera.transform.rotation;
            _skipAngularFrame = false;
            _angularVelocity = 0f;
            return;
        }
        Quaternion deltaRot = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRotation);
        deltaRot.ToAngleAxis(out float angle, out Vector3 _);
        _angularVelocity = angle / Time.deltaTime;
        _lastCamRotation = targetCamera.transform.rotation;
    }

    // ─
    // Cleanup
    // ─
    void OnDestroy()
    {
        FlushAndCloseFile();
    }
}
