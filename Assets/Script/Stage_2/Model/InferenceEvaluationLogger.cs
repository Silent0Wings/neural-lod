using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

// Attach to the same GameObject as LODThresholdPredictor and RuntimeLODApplicator
// Records per frame performance metrics during neural LOD inference
// Outputs CSV to Application.persistentDataPath/InferenceEval/
[RequireComponent(typeof(LODThresholdPredictor))]
[RequireComponent(typeof(RuntimeLODApplicator))]
public class InferenceEvaluationLogger : MonoBehaviour
{
    // run label written to every row so multiple runs can be compared in one CSV
    [Header("Run Config")]
    public string runLabel = "neural_baker";
    public bool autoStart = true;

    // how many frames to skip before logging starts, avoids GPU warmup zeros
    [Header("Warmup")]
    public int warmupFrames = 64;

    // max frames to log, 0 means log until manually stopped
    [Header("Capture")]
    public int maxFrames = 0;

    // flush buffer to disk every N rows, avoids data loss on crash
    [Header("IO")]
    public int flushInterval = 120;

    // internal state
    private bool isLogging = false;
    private int frameCount = 0;
    private int warmupCounter = 0;

    private StreamWriter writer;
    private string outputPath;

    private FrameTiming[] frameTimings = new FrameTiming[1];

    // reference to predictor so we can read last predicted threshold and inference time
    private LODThresholdPredictor predictor;

    // stopwatch for inference call duration measurement
    private Stopwatch inferenceWatch = new Stopwatch();

    // last inference duration in ms, set by predictor wrapper
    [HideInInspector]
    public float lastInferenceDurationMs = 0f;

    // last predicted threshold applied, set by predictor
    [HideInInspector]
    public float lastPredictedThreshold = 0f;

    void Awake()
    {
        predictor = GetComponent<LODThresholdPredictor>();
    }

    void Start()
    {
        if (autoStart)
            StartLogging();
    }

    public void StartLogging()
    {
        string folder = System.IO.Path.Combine(Application.persistentDataPath, "InferenceEval");
        Directory.CreateDirectory(folder);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename = $"inference_eval_{runLabel}_{timestamp}.csv";
        outputPath = System.IO.Path.Combine(folder, filename);

        writer = new StreamWriter(outputPath, append: false);

        // write header
        writer.WriteLine(
            "run_label," +
            "frame," +
            "cpu_ms," +
            "gpu_ms," +
            "fps," +
            "inference_duration_ms," +
            "predicted_threshold," +
            "lod_bias_applied"
        );

        FrameTimingManager.CaptureFrameTimings();
        isLogging = true;
        frameCount = 0;
        warmupCounter = 0;

        UnityEngine.Debug.Log($"[InferenceEvaluationLogger] Logging started -> {outputPath}");
    }

    public void StopLogging()
    {
        if (!isLogging) return;
        isLogging = false;

        if (writer != null)
        {
            writer.Flush();
            writer.Close();
            writer = null;
        }

        UnityEngine.Debug.Log($"[InferenceEvaluationLogger] Logging stopped. Frames captured: {frameCount}");
    }

    void LateUpdate()
    {
        if (!isLogging) return;

        // skip warmup frames
        if (warmupCounter < warmupFrames)
        {
            warmupCounter++;
            return;
        }

        // stop if max frames reached
        if (maxFrames > 0 && frameCount >= maxFrames)
        {
            StopLogging();
            return;
        }

        FrameTimingManager.CaptureFrameTimings();
        uint captured = FrameTimingManager.GetLatestTimings(1, frameTimings);

        float cpuMs = 0f;
        float gpuMs = 0f;

        if (captured > 0)
        {
            cpuMs = (float)frameTimings[0].cpuFrameTime;
            gpuMs = (float)frameTimings[0].gpuFrameTime;

            // guard against GPU warmup overflow values
            if (gpuMs > 500f || gpuMs < 0f) gpuMs = 0f;
            if (cpuMs > 500f || cpuMs < 0f) cpuMs = 0f;
        }

        float fps = Time.deltaTime > 0f ? 1f / Time.deltaTime : 0f;

        // current lodBias as applied by RuntimeLODApplicator
        float lodBiasApplied = QualitySettings.lodBias;

        writer.WriteLine(
            $"{runLabel}," +
            $"{frameCount}," +
            $"{cpuMs:F4}," +
            $"{gpuMs:F4}," +
            $"{fps:F4}," +
            $"{lastInferenceDurationMs:F4}," +
            $"{lastPredictedThreshold:F6}," +
            $"{lodBiasApplied:F6}"
        );

        frameCount++;

        // periodic flush
        if (frameCount % flushInterval == 0)
            writer.Flush();
    }

    void OnDestroy()
    {
        StopLogging();
    }
}
