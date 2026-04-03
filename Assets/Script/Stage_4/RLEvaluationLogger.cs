using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

// RLEvaluationLogger
// Stage 4 — Records RL-specific evaluation metrics for analysis and baselines comparison.
//
// Writes two CSV files per run:
//   1. per_step_{runLabel}_{timestamp}.csv — per-frame timeseries
//   2. per_episode_{runLabel}_{timestamp}.csv — episode-level aggregates
//
// RL metrics logged (per the project spec):
//   • Reward convergence  : episode return (cumulative reward proxy) per episode.
//   • Policy stability    : action variance per episode; a good policy settles to low variance.
//   • Generalization      : run on held-out paths with the same logger; compare mean GPU CDF.
//
// Reward proxy computed inline (GPU time only — matches Python training formula):
//   r_t = -alpha * ((gpu_ms - t_target) / t_target)^2
//         - beta  * (1 - screen_coverage)   ← SSIM proxy; document substitution
//         - gamma * (recent_switch_count / N_max)
//
// Baselines to compare against (same CSV format, different runLabel):
//   unity_default    : lodBias=1.0, no controller
//   rule_based       : fps<45 → decrease (RLPolicyController fallback mode)
//   stage2_mlp       : Stage 2 supervised MLP oracle (RuntimeLODApplicator)
//   random_policy    : Uniform random bias in [0.30, 2.00]
//
// SSIM proxy note: screen_coverage delta is used as SSIM approximation.
// Document this substitution in the evaluation report.

[RequireComponent(typeof(RLFeatureExtractor))]
public class RLEvaluationLogger : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("Run Config")]
    [Tooltip("Identifies this run in the CSV. Use e.g. 'neural_rl', 'rule_based', 'unity_default'.")]
    public string runLabel = "neural_rl";
    public bool autoStart = true;

    [Header("Camera")]
    public Camera targetCamera;

    [Header("Reward Weights")]
    [Tooltip("Alpha: weight for frame-time penalty term.")]
    public float alpha = 1.0f;
    [Tooltip("Beta: weight for visual-quality penalty term (SSIM proxy).")]
    public float beta  = 0.5f;
    [Tooltip("Gamma: weight for LOD-switch stability penalty.")]
    public float gamma = 0.2f;
    [Tooltip("GPU frame-time target in ms (VAR_T_TARGET_MS).")]
    public float tTargetMs = 4.5f;
    [Tooltip("N_max for switch-count normalization (matches lodSwitchWindow in RLFeatureExtractor).")]
    public float nMax = 30f;

    [Header("Warmup")]
    public int warmupFrames = 64;

    [Header("Capture")]
    [Tooltip("Stop after this many frames per episode. 0 = follow RLRolloutLogger episode boundary.")]
    public int maxFramesPerEpisode = 0;

    [Header("IO")]
    public int flushInterval = 120;

    [Header("Exit")]
    [Tooltip("Quit the application after the final episode completes.")]
    public bool quitOnComplete = false;

    // ── Private State ──────────────────────────────────────────────────────

    private RLFeatureExtractor _extractor;
    private RLRolloutLogger    _rolloutLogger; // optional; used to detect episode boundaries

    private bool _logging      = false;
    private int  _warmup       = 0;
    private int  _frameInEp    = 0;
    private int  _episodeIndex = 0;
    private bool _quitting     = false;

    // Per-frame CSV
    private StreamWriter _stepWriter;
    private string       _stepPath;

    // Per-episode CSV
    private StreamWriter _epWriter;
    private string       _epPath;

    // Frame timings
    private FrameTiming[] _frameTimings = new FrameTiming[1];

    // Episode accumulators
    private float _episodeReturn   = 0f;
    private float _actionSum       = 0f;
    private float _actionSumSq     = 0f;
    private int   _actionCount     = 0;
    private float _gpuSum          = 0f;
    private int   _gpuCount        = 0;
    private float _prevScreenCov   = -1f;   // for SSIM proxy delta

    // ── Lifecycle ──────────────────────────────────────────────────────────

    void Awake()
    {
        _extractor     = GetComponent<RLFeatureExtractor>();
        _rolloutLogger = GetComponent<RLRolloutLogger>();

        if (targetCamera == null)
            targetCamera = Camera.main;
    }

    void Start()
    {
        if (autoStart)
            StartLogging();
    }

    // ── Logging Control ────────────────────────────────────────────────────

    public void StartLogging()
    {
        string folder    = Path.Combine(Application.persistentDataPath, "EvalRollouts");
        Directory.CreateDirectory(folder);

        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");

        _stepPath = Path.Combine(folder, $"per_step_{runLabel}_{timestamp}.csv");
        _epPath   = Path.Combine(folder, $"per_episode_{runLabel}_{timestamp}.csv");

        _stepWriter = new StreamWriter(_stepPath, append: false);
        _epWriter   = new StreamWriter(_epPath,   append: false);

        // Per-step header
        _stepWriter.WriteLine(
            "run_label," +
            "episode," +
            "frame," +
            "cpu_ms," +
            "gpu_ms," +               // reward signal — GPU ONLY
            "fps," +
            "lod_bias," +
            "action_delta," +
            "reward_step," +          // inline proxy reward
            "cumulative_return," +
            "screen_coverage," +      // SSIM proxy value
            "screen_coverage_delta," +// SSIM proxy delta (coverage change)
            "recent_switch_count"
        );

        // Per-episode header
        _epWriter.WriteLine(
            "run_label," +
            "episode," +
            "total_frames," +
            "episode_return," +
            "mean_gpu_ms," +
            "mean_fps," +
            "mean_lod_bias," +
            "action_mean," +
            "action_variance," +      // policy stability metric
            "switch_count_total"
        );

        FrameTimingManager.CaptureFrameTimings();

        _logging      = true;
        _warmup       = 0;
        _frameInEp    = 0;
        _episodeIndex = 0;
        ResetEpisodeAccumulators();

        Debug.Log($"[RLEvaluationLogger] Logging started.\n  Steps: {_stepPath}\n  Episodes: {_epPath}");
    }

    public void StopLogging()
    {
        if (!_logging) return;
        _logging = false;
        FinaliseEpisode();
        CloseWriters();
        Debug.Log($"[RLEvaluationLogger] Logging stopped. Episodes: {_episodeIndex}.");
    }

    // ── Per-Frame Recording ────────────────────────────────────────────────

    void LateUpdate()
    {
        if (!_logging || !_extractor.IsReady) return;

        if (_warmup < warmupFrames) { _warmup++; return; }

        // Detect episode boundary driven by RLRolloutLogger
        if (_rolloutLogger != null && _rolloutLogger.CollectionComplete)
        {
            StopLogging();
            if (quitOnComplete && !_quitting)
            {
                _quitting = true;
                QuitApplication();
            }
            return;
        }

        if (maxFramesPerEpisode > 0 && _frameInEp >= maxFramesPerEpisode)
        {
            FinaliseEpisode();
            _episodeIndex++;
            _frameInEp = 0;
            ResetEpisodeAccumulators();
        }

        FrameTimingManager.CaptureFrameTimings();
        uint captured = FrameTimingManager.GetLatestTimings(1, _frameTimings);

        float cpuMs = 0f, gpuMs = 0f;
        if (captured > 0)
        {
            cpuMs = (float)_frameTimings[0].cpuFrameTime;
            gpuMs = (float)_frameTimings[0].gpuFrameTime;
            if (gpuMs > 500f || gpuMs < 0f) gpuMs = 0f;
            if (cpuMs > 500f || cpuMs < 0f) cpuMs = 0f;
        }

        float fps        = Time.deltaTime > 0f ? 1f / Time.deltaTime : 0f;
        float lodBias    = QualitySettings.lodBias;
        float actionDelta = _rolloutLogger != null ? _rolloutLogger.LastActionDelta : 0f;

        float[] raw = _extractor.RawFeatures;
        float screenCov      = raw != null ? raw[8] : 0f;
        float recentSwitches = raw != null ? raw[10] : 0f;

        // SSIM proxy delta (screen_coverage change between steps)
        float coverageDelta = _prevScreenCov >= 0f ? screenCov - _prevScreenCov : 0f;
        _prevScreenCov = screenCov;

        // Inline reward proxy — same formula as Python training:
        //   r_t = -alpha * ((gpu_ms - t_target) / t_target)^2
        //         - beta  * (1 - screen_coverage)
        //         - gamma * (recent_switch_count / N_max)
        // WARNING: only valid when gpuMs > 0. Corrupt frames should be filtered
        // in Python (df_samples_clean) before computing episode returns for training.
        float rewardStep = 0f;
        if (gpuMs > 0f)
        {
            float normTime   = (gpuMs - tTargetMs) / tTargetMs;
            float frameTerm  = -alpha * (normTime * normTime);
            float qualTerm   = -beta  * (1f - Mathf.Clamp01(screenCov));
            float stabTerm   = -gamma * Mathf.Clamp01(recentSwitches / nMax);
            rewardStep = frameTerm + qualTerm + stabTerm;
        }

        _episodeReturn += rewardStep;

        // Accumulate episode stats
        if (gpuMs > 0f)
        {
            _gpuSum += gpuMs;
            _gpuCount++;
        }

        _actionSum   += actionDelta;
        _actionSumSq += actionDelta * actionDelta;
        _actionCount++;

        // Write per-step row
        _stepWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F4},{4:F4},{5:F2},{6:F4},{7:F4},{8:F6},{9:F6},{10:F6},{11:F6},{12:F0}",
            runLabel,
            _episodeIndex,
            _frameInEp,
            cpuMs,
            gpuMs,
            fps,
            lodBias,
            actionDelta,
            rewardStep,
            _episodeReturn,
            screenCov,
            coverageDelta,
            recentSwitches
        ));

        _frameInEp++;

        if (_frameInEp % flushInterval == 0)
        {
            _stepWriter.Flush();
            _epWriter.Flush();
        }
    }

    // ── Episode Finalisation ───────────────────────────────────────────────

    private void FinaliseEpisode()
    {
        if (_actionCount == 0) return;

        float meanGpu    = _gpuCount > 0 ? _gpuSum / _gpuCount : 0f;
        float meanFps    = 0f; // fps already aggregated implicitly via frame count
        float meanBias   = 1.0f; // placeholder — logged per-step; Python can aggregate
        float actionMean = _actionSum / _actionCount;

        // Sample variance: E[X^2] - E[X]^2
        float actionVariance = (_actionSumSq / _actionCount) - (actionMean * actionMean);
        actionVariance = Mathf.Max(0f, actionVariance); // numerical safety

        float switchTotal = _extractor.RawFeatures != null ? _extractor.RawFeatures[10] : 0f;

        _epWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3:F6},{4:F4},{5:F2},{6:F4},{7:F6},{8:F6},{9:F0}",
            runLabel,
            _episodeIndex,
            _frameInEp,
            _episodeReturn,
            meanGpu,
            meanFps,
            meanBias,
            actionMean,
            actionVariance,
            switchTotal
        ));

        _epWriter.Flush();

        Debug.Log($"[RLEvaluationLogger] Episode {_episodeIndex} | " +
                  $"return={_episodeReturn:F4} | " +
                  $"action_var={actionVariance:F6} | " +
                  $"mean_gpu={meanGpu:F2}ms | " +
                  $"frames={_frameInEp}");
    }

    private void ResetEpisodeAccumulators()
    {
        _episodeReturn = 0f;
        _actionSum     = 0f;
        _actionSumSq   = 0f;
        _actionCount   = 0;
        _gpuSum        = 0f;
        _gpuCount      = 0;
        _prevScreenCov = -1f;
    }

    // ── IO Helpers ─────────────────────────────────────────────────────────

    private void CloseWriters()
    {
        _stepWriter?.Flush(); _stepWriter?.Close(); _stepWriter = null;
        _epWriter?.Flush();   _epWriter?.Close();   _epWriter   = null;
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
