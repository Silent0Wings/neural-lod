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
// Reward proxy computed inline — IMPROVEMENT-BASED (matches Python training formula):
//   r_t = (gpu_prev - gpu_ms)                      ← positive when GPU time decreases
//         + bonusScale * (gpu_ms <= tTargetMs)      ← bonus for staying under budget
//   Clipped to [-rewardClip, +rewardClip].
//
// Rationale (Stage_4_rl_failure_diagnosis_and_fix.md):
//   The old quadratic reward was always <= 0, causing the agent to learn inaction.
//   The switch penalty (gamma > 0) compounded this: any action was penalized.
//   Improvement-based reward is directional and can be positive.
//
// Baselines to compare against (same CSV format, different runLabel):
//   unity_default    : lodBias=1.0, no controller
//   rule_based       : fps<45 → decrease (RLPolicyController fallback mode)
//   stage2_mlp       : Stage 2 supervised MLP oracle (RuntimeLODApplicator)
//   random_policy    : Uniform random bias in [0.30, 2.00]

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

    [Header("Reward (Improvement-Based)")]
    [Tooltip("GPU frame-time target in ms (VAR_T_TARGET_MS). Used in success bonus.")]
    public float tTargetMs  = 4.5f;
    [Tooltip("Bonus added to reward when gpu_ms <= tTargetMs.")]
    public float bonusScale = 1.0f;
    [Tooltip("Reward clipped to [-rewardClip, +rewardClip] to prevent single-frame dominance.")]
    public float rewardClip = 5.0f;

    [Header("Reward Weights (Legacy — not used in reward computation)")]
    [Tooltip("Alpha was the frame-time penalty weight. Not used in improvement-based reward.")]
    public float alpha = 1.0f;
    [Tooltip("Beta was the visual-quality penalty weight. Not used in improvement-based reward.")]
    public float beta  = 0.5f;
    [Tooltip("Gamma for switch penalty. Must stay 0: switch penalty caused policy collapse.")]
    public float gamma = 0.0f;
    [Tooltip("N_max for switch-count normalization. Kept for baseline comparisons only.")]
    public float nMax  = 30f;

    [Header("Warmup")]
    public int warmupFrames = 64;

    [Header("Capture")]
    [Tooltip("Safety-net episode length in frames. Set equal to RLRolloutLogger.stepsPerEpisode. " +
             "Primary boundary is driven by RLRolloutLogger via NotifyRolloutEpisodeEnd().")]
    public int maxFramesPerEpisode = 1024;

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
    private float _fpsSum          = 0f;
    private int   _fpsCount        = 0;
    private float _biasSum         = 0f;
    private int   _biasCount       = 0;
    private int   _switchTotal     = 0;
    private float _prevScreenCov   = -1f;   // for coverage delta column
    private float _prevGpuMs       = -1f;   // for improvement-based reward (gpu_prev - gpu_t)

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

    /// <summary>
    /// Called by RLRolloutLogger when it starts a new rollout episode.
    /// Finalises the current evaluation episode and starts the next one,
    /// keeping evaluation episode boundaries in sync with rollout boundaries.
    /// </summary>
    public void NotifyRolloutEpisodeEnd()
    {
        if (!_logging) return;
        FinaliseEpisode();
        _episodeIndex++;
        _frameInEp = 0;
        ResetEpisodeAccumulators();
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

        float fps        = Time.deltaTime > 0f ? 1f / Time.deltaTime : 60f; // matches RLFeatureExtractor fallback
        float lodBias    = QualitySettings.lodBias;
        float actionDelta = _rolloutLogger != null ? _rolloutLogger.LastActionDelta : 0f;

        float[] raw = _extractor.RawFeatures;
        float screenCov      = raw != null ? raw[8] : 0f;
        float recentSwitches = raw != null ? raw[10] : 0f;

        // Coverage delta for CSV column
        float coverageDelta = _prevScreenCov >= 0f ? screenCov - _prevScreenCov : 0f;
        _prevScreenCov = screenCov;

        // Improvement-based reward — matches train_rl_policy_stage4.ipynb:
        //   r_t = (gpu_prev - gpu_ms) + bonusScale * (gpu_ms <= tTargetMs)
        //   clipped to [-rewardClip, +rewardClip]
        // Only computed when gpuMs > 0 (corrupt/warmup frames skipped).
        float rewardStep = 0f;
        if (gpuMs > 0f)
        {
            float gpuPrev    = _prevGpuMs > 0f ? _prevGpuMs : gpuMs; // first frame: no improvement
            float improvement = gpuPrev - gpuMs;
            float bonus       = gpuMs <= tTargetMs ? bonusScale : 0f;
            rewardStep        = Mathf.Clamp(improvement + bonus, -rewardClip, rewardClip);
        }
        _prevGpuMs = gpuMs > 0f ? gpuMs : _prevGpuMs;

        _episodeReturn += rewardStep;

        // Accumulate episode stats
        if (gpuMs > 0f)
        {
            _gpuSum += gpuMs;
            _gpuCount++;
        }

        if (fps > 0f)
        {
            _fpsSum += fps;
            _fpsCount++;
        }

        _biasSum  += lodBias;
        _biasCount++;

        if (actionDelta != 0f)
            _switchTotal++;

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

        float meanGpu    = _gpuCount  > 0 ? _gpuSum  / _gpuCount  : 0f;
        float meanFps    = _fpsCount  > 0 ? _fpsSum  / _fpsCount  : 0f;
        float meanBias   = _biasCount > 0 ? _biasSum / _biasCount : 1f;
        float actionMean = _actionSum / _actionCount;

        // Sample variance: E[X^2] - E[X]^2
        float actionVariance = (_actionSumSq / _actionCount) - (actionMean * actionMean);
        actionVariance = Mathf.Max(0f, actionVariance); // numerical safety

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
            _switchTotal
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
        _fpsSum        = 0f;
        _fpsCount      = 0;
        _biasSum       = 0f;
        _biasCount     = 0;
        _switchTotal   = 0;
        _prevScreenCov = -1f;
        _prevGpuMs     = -1f;
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
