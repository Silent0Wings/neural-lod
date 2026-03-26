using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// BaselineMetricsLogger 
/// Records mean FPS and frame time over a run using FrameTimingManager.
/// Logs averaged results to Console and saves to CSV.
/// Attach to any GameObject in the scene.
/// </summary>
public class BaselineMetricsLogger : MonoBehaviour
{
    [Header("Settings")]
    public int   sampleInterval = 60;    // frames between each log entry
    public bool  saveToCSV      = true;

    //  Interval accumulators 
    private int    _sampleCount      = 0;
    private float  _totalCpuTime     = 0f;
    private float  _totalGpuTime     = 0f;
    private float  _minCpuTime       = float.MaxValue;
    private float  _maxCpuTime       = float.MinValue;
    private float  _minGpuTime       = float.MaxValue;
    private float  _maxGpuTime       = float.MinValue;

    //  Run-level accumulators 
    private int    _totalSamples     = 0;
    private float  _runTotalCpu      = 0f;
    private float  _runTotalGpu      = 0f;

    //  FrameTimingManager 
    private FrameTiming[] _frameTimings = new FrameTiming[1];

    private StreamWriter _writer;

    void Start()
    {
        if (saveToCSV)
        {
            string path = Application.persistentDataPath + "/baseline_metrics.csv";
            _writer = new StreamWriter(path, append: false);
            // FIX #6: Separate CPU/GPU columns
            _writer.WriteLine(
                "lod_bias," +
                "avg_fps," +
                "avg_cpu_time_ms," +
                "avg_gpu_time_ms," +
                "min_cpu_time_ms," +
                "max_cpu_time_ms," +
                "min_gpu_time_ms," +
                "max_gpu_time_ms," +
                "sample_count"
            );
            Debug.Log($"[BaselineMetricsLogger] Saving to: {path}");
        }

        Debug.Log($"[BaselineMetricsLogger] Started. LOD Bias: {QualitySettings.lodBias} | Sample interval: {sampleInterval} frames");
    }

    void Update()
    {
        // Use FrameTimingManager for accurate CPU/GPU split
        FrameTimingManager.CaptureFrameTimings();
        uint count = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        if (count == 0) return; // no timing data available yet

        float cpuTime = (float)_frameTimings[0].cpuFrameTime;  // ms
        float gpuTime = (float)_frameTimings[0].gpuFrameTime;  // ms

        if (cpuTime <= 0f && gpuTime <= 0f) return; // skip invalid

        _sampleCount++;
        _totalCpuTime += cpuTime;
        _totalGpuTime += gpuTime;
        _runTotalCpu  += cpuTime;
        _runTotalGpu  += gpuTime;
        _totalSamples++;

        if (cpuTime < _minCpuTime) _minCpuTime = cpuTime;
        if (cpuTime > _maxCpuTime) _maxCpuTime = cpuTime;
        if (gpuTime < _minGpuTime) _minGpuTime = gpuTime;
        if (gpuTime > _maxGpuTime) _maxGpuTime = gpuTime;

        if (_sampleCount >= sampleInterval)
        {
            LogInterval();
            ResetInterval();
        }
    }

    void LogInterval()
    {
        float avgCpu = _totalCpuTime / _sampleCount;
        float avgGpu = _totalGpuTime / _sampleCount;
        float avgFPS = 1000f / Mathf.Max(avgCpu, avgGpu); // bottleneck determines FPS
        float lodBias = QualitySettings.lodBias;

        Debug.Log(
            $"[BaselineMetricsLogger] " +
            $"LOD Bias: {lodBias:F2} | " +
            $"Avg FPS: {avgFPS:F1} | " +
            $"Avg CPU: {avgCpu:F2}ms | " +
            $"Avg GPU: {avgGpu:F2}ms | " +
            $"Min/Max CPU: {_minCpuTime:F2}/{_maxCpuTime:F2}ms | " +
            $"Min/Max GPU: {_minGpuTime:F2}/{_maxGpuTime:F2}ms"
        );

        if (saveToCSV && _writer != null)
        {
            _writer.WriteLine(
                $"{lodBias:F2}," +
                $"{avgFPS:F1}," +
                $"{avgCpu:F2}," +
                $"{avgGpu:F2}," +
                $"{_minCpuTime:F2}," +
                $"{_maxCpuTime:F2}," +
                $"{_minGpuTime:F2}," +
                $"{_maxGpuTime:F2}," +
                $"{_sampleCount}"
            );
            _writer.Flush();
        }
    }

    void ResetInterval()
    {
        _sampleCount  = 0;
        _totalCpuTime = 0f;
        _totalGpuTime = 0f;
        _minCpuTime   = float.MaxValue;
        _maxCpuTime   = float.MinValue;
        _minGpuTime   = float.MaxValue;
        _maxGpuTime   = float.MinValue;
    }

    void OnDestroy()
    {
        if (_totalSamples > 0)
        {
            float overallAvgCpu = _runTotalCpu / _totalSamples;
            float overallAvgGpu = _runTotalGpu / _totalSamples;
            float overallAvgFPS = 1000f / Mathf.Max(overallAvgCpu, overallAvgGpu);

            Debug.Log(
                $"[BaselineMetricsLogger] RUN SUMMARY | " +
                $"LOD Bias: {QualitySettings.lodBias:F2} | " +
                $"Overall Avg FPS: {overallAvgFPS:F1} | " +
                $"Overall Avg CPU: {overallAvgCpu:F2}ms | " +
                $"Overall Avg GPU: {overallAvgGpu:F2}ms | " +
                $"Total Frames: {_totalSamples}"
            );
        }

        if (_writer != null)
        {
            _writer.Close();
        }
    }
}
