using System;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

// Attach to the same GameObject as LODThresholdPredictor and RuntimeLODApplicator
// Records per frame performance metrics during neural LOD inference
// Outputs CSV to Application.persistentDataPath/InferenceEval/
// Camera position and rotation are logged so eval rows can be joined
// against grid_points.csv and labels.csv for oracle comparison.
[RequireComponent(typeof(LODThresholdPredictor))]
[RequireComponent(typeof(RuntimeLODApplicator))]
public class InferenceEvaluationLogger : MonoBehaviour
{
    [Header("Run Config")]
    public string runLabel = "neural_baker";
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
    private LODThresholdPredictor predictor;
    private float _lastLoggedThreshold = -1f;

    // unused public fields kept for Inspector visibility only
    [HideInInspector] public float lastInferenceDurationMs = 0f;
    [HideInInspector] public float lastPredictedThreshold  = 0f;

    void Awake()
    {
        predictor = GetComponent<LODThresholdPredictor>();

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
        string folder = System.IO.Path.Combine(Application.persistentDataPath, "InferenceEval");
        Directory.CreateDirectory(folder);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename  = $"inference_eval_{runLabel}_{timestamp}.csv";
        outputPath = System.IO.Path.Combine(folder, filename);

        writer = new StreamWriter(outputPath, append: false);

        // header matches training data schema for oracle join
        // cam_pos_x/y/z match grid_points.x/y/z (nearest neighbour join)
        // cam_rot_x/y/z match labels.rot_x/y/z (nearest rotation join)
        writer.WriteLine(
            "run_label," +
            "frame," +
            "cpu_ms," +
            "gpu_ms," +
            "fps," +
            "inference_duration_ms," +
            "predicted_threshold," +
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

        UnityEngine.Debug.Log($"[InferenceEvaluationLogger] Logging stopped. Frames: {frameCount}. Output: {outputPath}");
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
                UnityEngine.Debug.Log($"[InferenceEvaluationLogger] Capture complete ({frameCount} frames). Exiting application.");
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

        // camera pose — used to join against grid_points.csv and labels.csv
        Vector3 pos = targetCamera != null ? targetCamera.transform.position    : Vector3.zero;
        Vector3 rot = targetCamera != null ? targetCamera.transform.eulerAngles : Vector3.zero;

        // normalize euler angles from [0, 360) to [-180, 180) to match labels.csv convention
        rot.x = rot.x > 180f ? rot.x - 360f : rot.x;
        rot.y = rot.y > 180f ? rot.y - 360f : rot.y;
        rot.z = rot.z > 180f ? rot.z - 360f : rot.z;

        float currentThreshold = predictor.lastPredictedThreshold;
        int thresholdChanged   = (_lastLoggedThreshold >= 0f &&
                                  Mathf.Abs(currentThreshold - _lastLoggedThreshold) > 0.0001f) ? 1 : 0;
        _lastLoggedThreshold = currentThreshold;

        writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2:F4},{3:F4},{4:F4},{5:F4},{6:F6},{7:F6},{8},{9:F4},{10:F4},{11:F4},{12:F4},{13:F4},{14:F4}",
            runLabel,
            frameCount,
            cpuMs,
            gpuMs,
            fps,
            predictor.lastInferenceDurationMs,
            currentThreshold,
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
