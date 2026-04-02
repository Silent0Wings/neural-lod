using System;
using System.IO;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

// Stage 3 — Evaluation Logger for 4-Value Per-LOD Threshold Prediction.
// Records per-frame performance metrics during neural LOD inference.
// Logs all 4 predicted thresholds (predicted_t0 through predicted_t3).
// Outputs CSV to Application.persistentDataPath/InferenceEval/
// Camera position and rotation logged for oracle comparison.
[RequireComponent(typeof(LODThresholdPredictor4))]
[RequireComponent(typeof(RuntimeLODApplicator4))]
public class InferenceEvaluationLogger4 : MonoBehaviour
{
    [Header("Run Config")]
    public string runLabel = "neural_4thresh";
    public bool autoStart = true;

    [Header("Camera")]
    [Tooltip("Camera to log pose from. Defaults to Camera.main if null.")]
    public Camera targetCamera;

    [Header("Warmup")]
    public int warmupFrames = 64;

    [Header("Capture")]
    public int maxFrames = 0;

    [Header("IO")]
    public int flushInterval = 120;

    [Header("Exit")]
    [Tooltip("Quit the application after maxFrames is reached and logging is saved.")]
    public bool quitOnComplete = true;

    // internal state
    private bool isLogging = false;
    private int frameCount = 0;
    private int warmupCounter = 0;
    private bool _quitting = false;

    private StreamWriter writer;
    private string outputPath;

    private FrameTiming[] frameTimings = new FrameTiming[1];
    private LODThresholdPredictor4 predictor;

    // track last applied thresholds to detect changes
    private float[] _lastLoggedThresholds;

    void Awake()
    {
        predictor = GetComponent<LODThresholdPredictor4>();

        if (targetCamera == null)
            targetCamera = Camera.main;
    }

    void Start()
    {
        if (autoStart)
            StartLogging();
    }

    public void StartLogging()
    {
        string folder = Path.Combine(Application.persistentDataPath, "InferenceEval");
        Directory.CreateDirectory(folder);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename  = $"inference_eval_{runLabel}_{timestamp}.csv";
        outputPath = Path.Combine(folder, filename);

        writer = new StreamWriter(outputPath, append: false);

        // header — logs all 4 predicted thresholds individually
        writer.WriteLine(
            "run_label," +
            "frame," +
            "cpu_ms," +
            "gpu_ms," +
            "fps," +
            "inference_duration_ms," +
            "predicted_t0," +
            "predicted_t1," +
            "predicted_t2," +
            "predicted_t3," +
            "predicted_threshold_mean," +
            "lod_bias_applied," +
            "threshold_changed," +
            "cam_pos_x," +
            "cam_pos_y," +
            "cam_pos_z," +
            "cam_rot_x," +
            "cam_rot_y," +
            "cam_rot_z"
        );

        FrameTimingManager.CaptureFrameTimings();
        isLogging      = true;
        frameCount     = 0;
        warmupCounter  = 0;
        _lastLoggedThresholds = null;

        UnityEngine.Debug.Log($"[InferenceEvaluationLogger4] Logging started -> {outputPath}");
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

        UnityEngine.Debug.Log($"[InferenceEvaluationLogger4] Logging stopped. Frames: {frameCount}. Output: {outputPath}");
    }

    void LateUpdate()
    {
        if (!isLogging) return;

        if (warmupCounter < warmupFrames)
        {
            warmupCounter++;
            return;
        }

        if (maxFrames > 0 && frameCount >= maxFrames)
        {
            StopLogging();
            if (quitOnComplete && !_quitting)
            {
                _quitting = true;
                UnityEngine.Debug.Log($"[InferenceEvaluationLogger4] Capture complete ({frameCount} frames). Exiting application.");
                QuitApplication();
            }
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

            if (gpuMs > 500f || gpuMs < 0f) gpuMs = 0f;
            if (cpuMs > 500f || cpuMs < 0f) cpuMs = 0f;
        }

        float fps            = Time.deltaTime > 0f ? 1f / Time.deltaTime : 0f;
        float lodBiasApplied = QualitySettings.lodBias;

        // camera pose for oracle comparison
        Vector3 pos = targetCamera != null ? targetCamera.transform.position    : Vector3.zero;
        Vector3 rot = targetCamera != null ? targetCamera.transform.eulerAngles : Vector3.zero;

        // normalize euler angles from [0, 360) to [-180, 180)
        rot.x = rot.x > 180f ? rot.x - 360f : rot.x;
        rot.y = rot.y > 180f ? rot.y - 360f : rot.y;
        rot.z = rot.z > 180f ? rot.z - 360f : rot.z;

        // extract all 4 thresholds from predictor
        float t0 = predictor.lastPrediction != null && predictor.lastPrediction.Length > 0 ? predictor.lastPrediction[0] : 0f;
        float t1 = predictor.lastPrediction != null && predictor.lastPrediction.Length > 1 ? predictor.lastPrediction[1] : 0f;
        float t2 = predictor.lastPrediction != null && predictor.lastPrediction.Length > 2 ? predictor.lastPrediction[2] : 0f;
        float t3 = predictor.lastPrediction != null && predictor.lastPrediction.Length > 3 ? predictor.lastPrediction[3] : 0f;

        // detect if any threshold changed since last log
        int thresholdChanged = 0;
        if (_lastLoggedThresholds != null && predictor.lastPrediction != null)
        {
            for (int i = 0; i < Mathf.Min(_lastLoggedThresholds.Length, predictor.lastPrediction.Length); i++)
            {
                if (Mathf.Abs(predictor.lastPrediction[i] - _lastLoggedThresholds[i]) > 0.0001f)
                {
                    thresholdChanged = 1;
                    break;
                }
            }
        }

        // update last logged thresholds
        if (predictor.lastPrediction != null)
        {
            if (_lastLoggedThresholds == null || _lastLoggedThresholds.Length != predictor.lastPrediction.Length)
                _lastLoggedThresholds = new float[predictor.lastPrediction.Length];
            System.Array.Copy(predictor.lastPrediction, _lastLoggedThresholds, predictor.lastPrediction.Length);
        }

        writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2:F4},{3:F4},{4:F4},{5:F4},{6:F6},{7:F6},{8:F6},{9:F6},{10:F6},{11:F6},{12},{13:F4},{14:F4},{15:F4},{16:F4},{17:F4},{18:F4}",
            runLabel,
            frameCount,
            cpuMs,
            gpuMs,
            fps,
            predictor.lastInferenceDurationMs,
            t0,
            t1,
            t2,
            t3,
            predictor.lastPredictedThreshold,
            lodBiasApplied,
            thresholdChanged,
            pos.x,
            pos.y,
            pos.z,
            rot.x,
            rot.y,
            rot.z
        ));

        frameCount++;

        if (frameCount % flushInterval == 0)
            writer.Flush();
    }

    void OnDestroy()
    {
        StopLogging();
    }

    private void QuitApplication()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
}
