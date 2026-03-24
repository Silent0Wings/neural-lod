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
    public int bufferFlushInterval = 120;

    [Header("Row Cap")]
    public int targetRowCount = 5000;
    public bool loggingComplete = false;

    [Header("Coverage Sampling")]
    public int coverageSampleInterval = 30; // CHANGED from 4 to 30

    [Header("References")]
    public CameraPathAnimator cameraPath;

    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;
    private ProfilerRecorder _batchesRecorder;
    private ProfilerRecorder _setpassRecorder;

    private FrameTiming[] _frameTimings = new FrameTiming[1];

    private List<string> _rowBuffer;
    private StreamWriter _writer;
    private int _frameCount = 0;
    private int _validRows = 0;
    private string _filePath;

    private Renderer[] _allRenderers;
    private int _runIndex = 0;
    private float _lastLodBias = 0f;
    private Quaternion _lastCamRotation;
    private float _angularVelocity = 0f;
    private float _currentMoveSpeed = 0f;
    private float _currentRotateSpeed = 0f;
    private bool _skipAngularFrame = false;

    private int _cachedVisibleRenderers = 0;
    private float _cachedScreenCoverage = 0f;
    private int _coverageFrameCounter = 0;

    private Plane[] _frustumPlanes = new Plane[6]; // CHANGED pre-allocated to avoid per-call GC alloc

    void Awake()
    {
        ApplyTimingGuards();

        if (targetCamera == null)
            targetCamera = Camera.main;

        if (cameraPath == null)
            cameraPath = FindFirstObjectByType<CameraPathAnimator>();
    }

    void OnEnable()
    {
        _trisRecorder     = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
        _batchesRecorder  = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Batches Count");
        _setpassRecorder  = ProfilerRecorder.StartNew(ProfilerCategory.Render, "SetPass Calls Count");
    }

    void OnDisable()
    {
        _trisRecorder.Dispose();
        _drawCallRecorder.Dispose();
        _batchesRecorder.Dispose();
        _setpassRecorder.Dispose();
    }

    void Start()
    {
        _lastCamRotation = targetCamera.transform.rotation;
        _lastLodBias     = QualitySettings.lodBias;
        _allRenderers    = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
    }

    private void ApplyTimingGuards()
    {
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount  = 0;
    }

    public void ResetLogger(int runIndex, float moveSpeed, float rotateSpeed)
    {
        ApplyTimingGuards();

        _currentMoveSpeed   = moveSpeed;
        _currentRotateSpeed = rotateSpeed;

        FlushAndCloseFile();

        _runIndex            = runIndex;
        _validRows           = 0;
        _frameCount          = 0;
        _coverageFrameCounter = 0;
        loggingComplete      = false;
        _lastLodBias         = QualitySettings.lodBias;

        OpenNewFile(moveSpeed, rotateSpeed);
        _skipAngularFrame = true;
        _allRenderers     = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        Debug.Log($"[MetricLogger] Reset for run {_runIndex}. File: {_filePath}");
    }

    private void OpenNewFile(float moveSpeed, float rotateSpeed)
    {
        FlushAndCloseFile();

        _filePath = Application.persistentDataPath +
                $"/training_data_bias_{QualitySettings.lodBias:F1}" +
                $"_spd_{moveSpeed:F1}" +
                $"_rot_{rotateSpeed:F1}" +
                $".csv";

        _writer   = new StreamWriter(_filePath, append: false);
        _rowBuffer = new List<string>(bufferFlushInterval + 16);

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
            "waypoint_index," +
            "path_progress," +
            "draw_calls," +
            "batches_count," +
            "setpass_calls," +
            "target_lod_bias"
        );

        Debug.Log($"[MetricLogger] Logging to: {_filePath}");
    }

    private void FlushAndCloseFile()
    {
        if (_writer != null)
        {
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

    void Update()
    {
        if (_writer == null) return;
        if (loggingComplete) return;

        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        float cpu = 0f;
        float gpu = 0f;

        if (count > 0)
        {
            cpu = (float)_frameTimings[0].cpuFrameTime;
            gpu = (float)_frameTimings[0].gpuFrameTime;
        }

        long tris = _trisRecorder.Valid ? _trisRecorder.LastValue : 0;

        float velocity = targetCamera != null ? targetCamera.velocity.magnitude : 0f;
        Vector3 camPos = targetCamera.transform.position;
        float camRotY  = targetCamera.transform.eulerAngles.y;
        float fps      = 1f / Time.deltaTime;

        UpdateAngularVelocity();

        _coverageFrameCounter++;
        if (_coverageFrameCounter >= coverageSampleInterval)
        {
            _coverageFrameCounter = 0;
            var (vis, cov) = CountVisibleRenderersAndCoverage();
            _cachedVisibleRenderers = vis;
            _cachedScreenCoverage   = cov;
        }

        long drawCalls    = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;
        long batchesCount = _batchesRecorder.Valid ? _batchesRecorder.LastValue : 0;
        long setpassCalls = _setpassRecorder.Valid ? _setpassRecorder.LastValue : 0;

        float lodBias      = QualitySettings.lodBias;
        float previousBias = _lastLodBias;
        _lastLodBias       = lodBias;

        float frameHeadroom = 16.6f - Mathf.Max(cpu, gpu);

        int   waypointIndex = cameraPath != null ? cameraPath.CurrentIndex  : -1;
        float pathProgress  = cameraPath != null ? cameraPath.PathProgress  : -1f;

        bool validRow = cpu > 0f && gpu > 0f && tris > 0 && velocity >= 0f;
        if (!validRow) return;

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
            $"{waypointIndex}," +
            $"{pathProgress:F4}," +
            $"{drawCalls}," +
            $"{batchesCount}," +
            $"{setpassCalls}," +
            $"?";

        _rowBuffer.Add(row);
        _frameCount++;

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
            enabled         = false;
            Debug.Log($"[MetricLogger] Run {_runIndex} complete — {targetRowCount} rows.");
        }
    }

    private (int count, float coverage) CountVisibleRenderersAndCoverage()
    {
        GeometryUtility.CalculateFrustumPlanes(targetCamera, _frustumPlanes); // CHANGED non-allocating overload uses pre-allocated array
        float screenW   = Screen.width;
        float screenH   = Screen.height;
        int count       = 0;
        float totalArea = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (r == null || !r.enabled || !GeometryUtility.TestPlanesAABB(_frustumPlanes, r.bounds))
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

    private void UpdateAngularVelocity()
    {
        if (_skipAngularFrame)
        {
            _lastCamRotation  = targetCamera.transform.rotation;
            _skipAngularFrame = false;
            _angularVelocity  = 0f;
            return;
        }
        Quaternion deltaRot = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRotation);
        deltaRot.ToAngleAxis(out float angle, out Vector3 _);
        _angularVelocity  = angle / Time.deltaTime;
        _lastCamRotation  = targetCamera.transform.rotation;
    }

    void OnDestroy()
    {
        FlushAndCloseFile();
    }
}