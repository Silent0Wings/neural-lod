using System.IO;
using UnityEngine;

/// <summary>
/// BaselineMetricsLogger
/// Records mean FPS and frame time over a run.
/// Logs averaged results to Console and saves to CSV.
/// Attach to any GameObject in the scene.
/// </summary>
/// 
// C:/Users/Gica/AppData/LocalLow/DefaultCompany/COMP432_Project/baseline_metrics.csv

public class BaselineMetricsLogger : MonoBehaviour
{
    [Header("Settings")]
    public int   sampleInterval = 60;    // frames between each log entry
    public bool  saveToCSV      = true;

    
    // Private state
    
    private int    _sampleCount    = 0;
    private float  _totalFrameTime = 0f;
    private float  _minFrameTime   = float.MaxValue;
    private float  _maxFrameTime   = float.MinValue;

    private int    _totalSamples   = 0;
    private float  _runTotalFT     = 0f; // for overall mean at end

    private StreamWriter _writer;

    
    // Init
    
    void Start()
    {
        if (saveToCSV)
        {
            string path = Application.persistentDataPath + "/baseline_metrics.csv";
            _writer = new StreamWriter(path, append: false);
            _writer.WriteLine("lod_bias,avg_fps,avg_frame_time_ms,min_frame_time_ms,max_frame_time_ms,sample_count");
            Debug.Log($"[BaselineMetricsLogger] Saving to: {path}");
        }

        Debug.Log($"[BaselineMetricsLogger] Started. LOD Bias: {QualitySettings.lodBias} | Sample interval: {sampleInterval} frames");
    }

    
    // Per-frame accumulation
    
    void Update()
    {
        float frameTime = Time.deltaTime * 1000f; // convert to ms

        _sampleCount++;
        _totalFrameTime += frameTime;
        _runTotalFT     += frameTime;
        _totalSamples++;

        if (frameTime < _minFrameTime) _minFrameTime = frameTime;
        if (frameTime > _maxFrameTime) _maxFrameTime = frameTime;

        if (_sampleCount >= sampleInterval)
        {
            LogInterval();
            ResetInterval();
        }
    }

    
    // Log averaged interval
    
    void LogInterval()
    {
        float avgFrameTime = _totalFrameTime / _sampleCount;
        float avgFPS       = 1000f / avgFrameTime;
        float lodBias      = QualitySettings.lodBias;

        Debug.Log(
            $"[BaselineMetricsLogger] " +
            $"LOD Bias: {lodBias:F2} | " +
            $"Avg FPS: {avgFPS:F1} | " +
            $"Avg FT: {avgFrameTime:F2}ms | " +
            $"Min FT: {_minFrameTime:F2}ms | " +
            $"Max FT: {_maxFrameTime:F2}ms"
        );

        if (saveToCSV && _writer != null)
        {
            _writer.WriteLine(
                $"{lodBias:F2},{avgFPS:F1},{avgFrameTime:F2},{_minFrameTime:F2},{_maxFrameTime:F2},{_sampleCount}"
            );
            _writer.Flush();
        }
    }

    
    // Reset interval accumulators only
    
    void ResetInterval()
    {
        _sampleCount    = 0;
        _totalFrameTime = 0f;
        _minFrameTime   = float.MaxValue;
        _maxFrameTime   = float.MinValue;
    }

    
    // On stop — log overall run summary
    
    void OnDestroy()
    {
        if (_totalSamples > 0)
        {
            float overallAvgFT  = _runTotalFT / _totalSamples;
            float overallAvgFPS = 1000f / overallAvgFT;

            Debug.Log(
                $"[BaselineMetricsLogger] RUN SUMMARY | " +
                $"LOD Bias: {QualitySettings.lodBias:F2} | " +
                $"Overall Avg FPS: {overallAvgFPS:F1} | " +
                $"Overall Avg FT: {overallAvgFT:F2}ms | " +
                $"Total Frames: {_totalSamples}"
            );
        }

        if (_writer != null)
        {
            _writer.Close();
        }
    }
}
