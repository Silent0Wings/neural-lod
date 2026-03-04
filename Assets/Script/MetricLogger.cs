using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;

/// <summary>
/// MetricLogger — Phase 2.2
/// Logs per-frame performance metrics to CSV for ML training data collection.
/// Depends on: CameraPathAnimator (1.2) being active in scene.
/// Attach to any GameObject in the scene.
/// C:/Users/Gica/AppData/LocalLow/DefaultCompany/COMP432_Project/
/// </summary>
public class MetricLogger : MonoBehaviour
{
    [Header("Settings")]
    public Camera targetCamera;
    public int bufferFlushInterval = 120;

    [Header("Row Cap")]
    public int targetRowCount = 5000;
    public bool loggingComplete = false;

    // ─────────────────────────────────────────
    // ProfilerRecorders
    // ─────────────────────────────────────────
    private ProfilerRecorder _gpuRecorder;
    private ProfilerRecorder _cpuRecorder;
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    // ─────────────────────────────────────────
    // Private state
    // ─────────────────────────────────────────
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
    // ─────────────────────────────────────────
    // Init
    // ─────────────────────────────────────────
    void Awake()
    {
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;

        if (targetCamera == null)
            targetCamera = Camera.main;
    }

    void OnEnable()
    {
        _gpuRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "GPU Frame Time");
        _cpuRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Internal, "Main Thread", 15);
        _trisRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
    }

    void OnDisable()
    {
        _gpuRecorder.Dispose();
        _cpuRecorder.Dispose();
        _trisRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    void Start()
    {
        _lastCamRotation = targetCamera.transform.rotation;
        _lastLodBias = QualitySettings.lodBias;
        _allRenderers = FindObjectsByType<Renderer>(FindObjectsSortMode.None);
        //OpenNewFile(0f, 0f);
    }

    // ─────────────────────────────────────────
    // Public reset — called by orchestrator between runs
    // ─────────────────────────────────────────
    public void ResetLogger(int runIndex, float moveSpeed, float rotateSpeed)
    {
        _currentMoveSpeed = moveSpeed;
        _currentRotateSpeed = rotateSpeed;

        CloseFile();

        _runIndex = runIndex;
        _validRows = 0;
        _frameCount = 0;
        loggingComplete = false;
        _lastLodBias = QualitySettings.lodBias;

        OpenNewFile(moveSpeed, rotateSpeed);
        enabled = true;

        Debug.Log($"[MetricLogger] Reset for run {_runIndex}. File: {_filePath}");
    }

    // ─────────────────────────────────────────
    // Open a new CSV file for this run
    // filename: training_data_bias_1.0_run001.csv
    // ─────────────────────────────────────────
    private void OpenNewFile(float moveSpeed, float rotateSpeed)
    {
        CloseFile();
        _filePath = Application.persistentDataPath +
                $"/training_data_bias_{QualitySettings.lodBias:F1}" +
                $"_spd_{moveSpeed:F1}" +
                $"_rot_{rotateSpeed:F1}" +
                $".csv";

        _writer = new StreamWriter(_filePath, append: false);

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
            "target_lod_bias"         // oracle label — Phase 2.3
        );

        Debug.Log($"[MetricLogger] Logging to: {_filePath}");
    }

    private void CloseFile()
    {
        if (_writer != null)
        {
            _writer.Flush();
            _writer.Close();
            _writer.Dispose();
            _writer = null;
        }
    }

    // ─────────────────────────────────────────
    // Per-frame capture
    // ─────────────────────────────────────────
    void Update()
    {
        if (_writer == null) return;
        if (loggingComplete) return;

        // ── Timing ───────────────────────────
        float cpu = _cpuRecorder.Valid ? _cpuRecorder.LastValue / 1_000_000f : 0f;
        float gpu = _gpuRecorder.Valid ? _gpuRecorder.LastValue / 1_000_000f : 0f;
        long tris = _trisRecorder.Valid ? _trisRecorder.LastValue : 0;

        // ── Camera ───────────────────────────
        float velocity = targetCamera != null ? targetCamera.velocity.magnitude : 0f;
        Vector3 camPos = targetCamera.transform.position;
        float camRotY = targetCamera.transform.eulerAngles.y;
        float fps = 1f / Time.deltaTime;

        UpdateAngularVelocity(); // updates _angularVelocity

        // ── Scene ────────────────────────────
        var (visibleRenderers, screenCoverage) = CountVisibleRenderersAndCoverage();
        long drawCalls = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;

        // ── LOD state ────────────────────────
        float lodBias = QualitySettings.lodBias;
        float previousBias = _lastLodBias;   // read BEFORE overwriting
        _lastLodBias = lodBias;         // store for next frame

        // ── Derived ──────────────────────────
        float frameHeadroom = 16.6f - gpu;

        // ── Validation ───────────────────────
        bool validRow = cpu > 0f && gpu > 0f && tris > 0 && velocity >= 0f;
        if (!validRow) return;

        // ── Write row ────────────────────────
        _writer.WriteLine(
            $"{cpu:F4}," +
            $"{gpu:F4}," +
            $"{tris}," +
            $"{velocity:F4}," +
            $"{visibleRenderers}," +
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
            $"{screenCoverage:F4}," +
            $"{_currentMoveSpeed:F2}," +
            $"{_currentRotateSpeed:F2}," +
            $"?"                    // target_lod_bias — oracle label Phase 2.3
        );

        _frameCount++;
        if (_frameCount % bufferFlushInterval == 0)
            _writer.Flush();

        _validRows++;
        if (_validRows >= targetRowCount)
        {
            CloseFile();
            loggingComplete = true;
            enabled = false;
            Debug.Log($"[MetricLogger] Run {_runIndex} complete — {targetRowCount} rows.");
        }

    }

    // ─────────────────────────────────────────
    // Count visible renderers + avg screen coverage
    // ─────────────────────────────────────────
    private (int count, float coverage) CountVisibleRenderersAndCoverage()
    {
        Plane[] frustum = GeometryUtility.CalculateFrustumPlanes(targetCamera);
        float screenW = Screen.width;
        float screenH = Screen.height;
        int count = 0;
        float totalArea = 0f;

        foreach (Renderer r in _allRenderers)
        {
            if (!r.enabled || !GeometryUtility.TestPlanesAABB(frustum, r.bounds))
                continue;

            count++;

            // project bounds to screen → normalized area
            Bounds b = r.bounds;
            Vector3 min = targetCamera.WorldToScreenPoint(b.min);
            Vector3 max = targetCamera.WorldToScreenPoint(b.max);
            float w = Mathf.Abs(max.x - min.x) / screenW;
            float h = Mathf.Abs(max.y - min.y) / screenH;
            totalArea += Mathf.Clamp01(w) * Mathf.Clamp01(h);
        }

        float avgCoverage = count > 0 ? totalArea / count : 0f;
        return (count, avgCoverage);
    }

    // ─────────────────────────────────────────
    // Angular velocity — degrees/sec from rotation delta
    // ─────────────────────────────────────────
    private void UpdateAngularVelocity()
    {
        Quaternion deltaRot = targetCamera.transform.rotation * Quaternion.Inverse(_lastCamRotation);
        deltaRot.ToAngleAxis(out float angle, out Vector3 _);
        _angularVelocity = angle / Time.deltaTime;
        _lastCamRotation = targetCamera.transform.rotation;
    }

    // ─────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────
    void OnDestroy()
    {
        CloseFile();
    }
}