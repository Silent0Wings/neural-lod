using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;
using System.IO;
using System.Collections.Generic;

/// <summary>
/// EvaluationLogger
/// Records per-frame metrics during an evaluation run for comparison
/// between fixed baseline LOD and the NeuralLODController.
///
/// Outputs a CSV with frame time stats, LOD bias, and flip rate.
/// Run once with NeuralLODController enabled, once with it disabled.
/// Compare the two CSVs for Phase 5 evaluation.
/// </summary>
public class EvaluationLogger : MonoBehaviour
{
    // ------------------------------------------------------------------
    // Inspector
    // ------------------------------------------------------------------
    [Header("Settings")]
    [Tooltip("Label for this run — e.g. 'neural' or 'fixed_1.0'")]
    public string runLabel = "neural";

    [Tooltip("How many rows to collect before stopping (0 = unlimited)")]
    public int targetRowCount = 5000;

    [Tooltip("CSV flush interval in rows")]
    public int bufferFlushInterval = 120;

    [Header("References")]
    public Camera targetCamera;
    public CameraPathAnimator cameraPath;

    [Header("Status (read-only)")]
    [SerializeField] private bool  _loggingComplete = false;
    [SerializeField] private int   _validRows       = 0;
    [SerializeField] private float _currentFPS      = 0f;

    // ------------------------------------------------------------------
    // Private
    // ------------------------------------------------------------------
    private FrameTiming[]    _frameTimings = new FrameTiming[1];
    private ProfilerRecorder _trisRecorder;
    private ProfilerRecorder _drawCallRecorder;

    private StreamWriter  _writer;
    private List<string>  _rowBuffer;
    private string        _filePath;

    private float _lastLodBias       = -1f;
    private int   _lodSwitchCount    = 0;
    private float _runStartTime      = 0f;

    // Per-run accumulators for summary stats
    private List<float> _cpuSamples = new List<float>();
    private List<float> _gpuSamples = new List<float>();

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------
    void Awake()
    {
        if (targetCamera == null)
            targetCamera = Camera.main;

        if (cameraPath == null)
            cameraPath = FindFirstObjectByType<CameraPathAnimator>();
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
        OpenFile();
        _lastLodBias  = QualitySettings.lodBias;
        _runStartTime = Time.time;
        Debug.Log($"[EvaluationLogger] Started run '{runLabel}'. Logging to: {_filePath}");
    }

    void Update()
    {
        if (_loggingComplete) return;

        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);
        if (count == 0) return;

        float cpu = (float)_frameTimings[0].cpuFrameTime;
        float gpu = (float)_frameTimings[0].gpuFrameTime;
        if (cpu <= 0f && gpu <= 0f) return;

        float lodBias    = QualitySettings.lodBias;
        float fps        = 1f / Time.deltaTime;
        long  tris       = _trisRecorder.Valid     ? _trisRecorder.LastValue     : 0;
        long  drawCalls  = _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;
        float velocity   = targetCamera.velocity.magnitude;
        int   waypoint   = cameraPath != null ? cameraPath.CurrentIndex : -1;
        float progress   = cameraPath != null ? cameraPath.PathProgress  : -1f;
        float elapsed    = Time.time - _runStartTime;

        // Track LOD switches
        bool switched = !Mathf.Approximately(lodBias, _lastLodBias);
        if (switched) _lodSwitchCount++;
        _lastLodBias = lodBias;

        _currentFPS = fps;
        _cpuSamples.Add(cpu);
        _gpuSamples.Add(gpu);

        string row =
            $"{elapsed:F4}," +
            $"{cpu:F4}," +
            $"{gpu:F4}," +
            $"{fps:F2}," +
            $"{lodBias:F4}," +
            $"{(switched ? 1 : 0)}," +
            $"{tris}," +
            $"{drawCalls}," +
            $"{velocity:F4}," +
            $"{waypoint}," +
            $"{progress:F4}";

        _rowBuffer.Add(row);

        if (_rowBuffer.Count >= bufferFlushInterval)
            Flush();

        _validRows++;

        if (targetRowCount > 0 && _validRows >= targetRowCount)
        {
            FinishRun();
        }
    }

    void OnDestroy()
    {
        if (!_loggingComplete)
            FinishRun();
    }

    // ------------------------------------------------------------------
    // File management
    // ------------------------------------------------------------------
    private void OpenFile()
    {
        _filePath = Application.persistentDataPath +
                    $"/eval_{runLabel}_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";

        _writer    = new StreamWriter(_filePath, append: false);
        _rowBuffer = new List<string>(bufferFlushInterval + 16);

        _writer.WriteLine(
            "elapsed_s,"        +
            "cpu_frame_ms,"     +
            "gpu_frame_ms,"     +
            "fps,"              +
            "lod_bias,"         +
            "bias_switched,"    +
            "triangle_count,"   +
            "draw_calls,"       +
            "camera_velocity,"  +
            "waypoint_index,"   +
            "path_progress"
        );
    }

    private void Flush()
    {
        if (_writer == null || _rowBuffer == null) return;
        foreach (string row in _rowBuffer)
            _writer.WriteLine(row);
        _rowBuffer.Clear();
        _writer.Flush();
    }

    private void FinishRun()
    {
        Flush();

        // Compute summary stats
        float meanCpu = ComputeMean(_cpuSamples);
        float meanGpu = ComputeMean(_gpuSamples);
        float p95Cpu  = ComputePercentile(_cpuSamples, 95f);
        float p99Cpu  = ComputePercentile(_cpuSamples, 99f);
        float p95Gpu  = ComputePercentile(_gpuSamples, 95f);
        float p99Gpu  = ComputePercentile(_gpuSamples, 99f);
        float runTime = Time.time - _runStartTime;
        float flipRate = runTime > 0f ? _lodSwitchCount / runTime : 0f;

        // Write summary block at end of CSV
        _writer.WriteLine("");
        _writer.WriteLine("# SUMMARY");
        _writer.WriteLine($"# Run label:      {runLabel}");
        _writer.WriteLine($"# Total rows:     {_validRows}");
        _writer.WriteLine($"# Run duration:   {runTime:F1}s");
        _writer.WriteLine($"# Mean CPU ms:    {meanCpu:F3}");
        _writer.WriteLine($"# P95  CPU ms:    {p95Cpu:F3}");
        _writer.WriteLine($"# P99  CPU ms:    {p99Cpu:F3}");
        _writer.WriteLine($"# Mean GPU ms:    {meanGpu:F3}");
        _writer.WriteLine($"# P95  GPU ms:    {p95Gpu:F3}");
        _writer.WriteLine($"# P99  GPU ms:    {p99Gpu:F3}");
        _writer.WriteLine($"# LOD switches:   {_lodSwitchCount}");
        _writer.WriteLine($"# Flip rate /s:   {flipRate:F3}");

        _writer.Close();
        _writer = null;

        _loggingComplete = true;
        enabled          = false;

        Debug.Log(
            $"[EvaluationLogger] Run '{runLabel}' complete.\n" +
            $"  Rows:      {_validRows}\n" +
            $"  Mean CPU:  {meanCpu:F3}ms | P95: {p95Cpu:F3}ms | P99: {p99Cpu:F3}ms\n" +
            $"  Mean GPU:  {meanGpu:F3}ms | P95: {p95Gpu:F3}ms | P99: {p99Gpu:F3}ms\n" +
            $"  Flip rate: {flipRate:F3} switches/s\n" +
            $"  File:      {_filePath}"
        );
    }

    // ------------------------------------------------------------------
    // Statistics helpers
    // ------------------------------------------------------------------
    private float ComputeMean(List<float> samples)
    {
        if (samples.Count == 0) return 0f;
        float sum = 0f;
        foreach (float v in samples) sum += v;
        return sum / samples.Count;
    }

    private float ComputePercentile(List<float> samples, float percentile)
    {
        if (samples.Count == 0) return 0f;
        List<float> sorted = new List<float>(samples);
        sorted.Sort();
        float index = (percentile / 100f) * (sorted.Count - 1);
        int   lower = (int)index;
        int   upper = Mathf.Min(lower + 1, sorted.Count - 1);
        float frac  = index - lower;
        return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
    }
}
