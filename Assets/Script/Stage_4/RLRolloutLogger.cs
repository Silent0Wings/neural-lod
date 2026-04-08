using UnityEngine;
using System.IO;
using System.Collections.Generic;
using System.Globalization;

// RLRolloutLogger
// Stage 4 — Logs per-step RL rollout data to CSV for offline REINFORCE training.
//
// One CSV file is produced per episode under Application.persistentDataPath/RLRollouts/.
// The Python training notebook reads these files, applies df_samples_clean filtering
// to remove corrupt GPU readings, then computes per-step improvement-based rewards:
//
//   r_t = (gpu_prev - gpu_frame_time) + BONUS_SCALE * (gpu_frame_time <= 4.5)
//   clipped to [-5.0, +5.0]
//
// t_target = 4.5 ms (VAR_T_TARGET_MS). Use gpu_frame_time ONLY — not cpu_frame_time.
// Switch penalty REMOVED (gamma=0): penalising switches caused policy inaction collapse.
//
// Camera pitch is restricted to {-15, 0, +15} degrees in the scene's CameraPathAnimator.
// Poses like -60 deg pitch produce near-empty frustums and zero reward signal.
//
// Column order matches the 13-feature state vector in RLFeatureExtractor exactly.
// Python scaler must use these same column names to rebuild normalization.

[RequireComponent(typeof(RLFeatureExtractor))]
public class RLRolloutLogger : MonoBehaviour
{
    // ── Inspector ──────────────────────────────────────────────────────────

    [Header("Episode Settings")]
    [Tooltip("Number of valid steps to collect per episode. 0 = unlimited.")]
    public int stepsPerEpisode = 1024;
    [Tooltip("Maximum episodes to collect, then stop. 0 = unlimited.")]
    public int maxEpisodes = 8;

    [Header("IO")]
    [Tooltip("Flush CSV buffer to disk every N rows.")]
    public int bufferFlushInterval = 120;

    [Header("Status (read-only)")]
    [SerializeField] private int  _episodeIndex  = 0;
    [SerializeField] private int  _stepIndex     = 0;
    [SerializeField] private bool _loggingActive = false;

    // ── Cross-component Action Handshake ───────────────────────────────────

    /// <summary>
    /// Written by RLPolicyController each step before LateUpdate runs.
    /// Logged as action_delta in the rollout CSV after guardrails are applied.
    /// </summary>
    [HideInInspector] public float LastActionDelta = 0f;

    /// <summary>
    /// Written by RLPolicyController each decision before guardrails run.
    /// Logged as raw_action_delta in the rollout CSV for diagnostics only.
    /// </summary>
    [HideInInspector] public float LastRawActionDelta = 0f;

    /// <summary>
    /// Written by RLPolicyController.ApplyWithGuardrails() before the action is applied.
    /// Snapshotted bias value BEFORE the action delta is applied.
    /// Logged as lod_bias_before_action in the rollout CSV.
    /// </summary>
    [HideInInspector] public float LastBiasBeforeAction = 1.0f;

    /// <summary>
    /// Written by RLPolicyController each Update before LateUpdate runs.
    /// True only on frames where the controller actually evaluated a decision.
    /// This lets the rollout CSV capture decision points rather than every render frame.
    /// </summary>
    [HideInInspector] public bool ShouldLogDecisionFrame = false;

    /// <summary>True once all episodes have been collected.</summary>
    public bool CollectionComplete { get; private set; } = false;

    // ── Private State ──────────────────────────────────────────────────────

    private RLFeatureExtractor   _extractor;
    private RLPolicyController   _policyController; // optional reset on episode boundary
    private RLEvaluationLogger   _evalLogger;       // optional episode sync notification

    private StreamWriter  _writer;
    private List<string>  _rowBuffer;
    private string        _currentFilePath;
    private bool          _episodeDone = false;

    // ── Lifecycle ──────────────────────────────────────────────────────────

    void Awake()
    {
        _extractor        = GetComponent<RLFeatureExtractor>();
        _policyController = GetComponent<RLPolicyController>();
        _evalLogger       = GetComponent<RLEvaluationLogger>();
    }

    void Start()
    {
        StartEpisode();
    }

    // ── Episode Management ─────────────────────────────────────────────────

    public void StartEpisode()
    {
        FlushAndClose();

        _stepIndex   = 0;
        _episodeDone = false;
        _loggingActive = true;

        string folder = Path.Combine(Application.persistentDataPath, "RLRollouts");
        Directory.CreateDirectory(folder);

        string filename = string.Format("rollout_ep{0:D4}_{1:yyyyMMdd_HHmmss}.csv",
                                        _episodeIndex,
                                        System.DateTime.Now);
        _currentFilePath = Path.Combine(folder, filename);
        _writer          = new StreamWriter(_currentFilePath, append: false);
        _rowBuffer       = new List<string>(bufferFlushInterval + 16);

        // Header matches VAR_STATE_FEATURES exactly plus action columns.
        // Python reward computation uses gpu_frame_time (col index 3).
        _writer.WriteLine(
            "episode," +
            "step," +
            "cpu_frame_time," +
            "gpu_frame_time," +        // GPU ONLY — t_frame for reward formula
            "fps," +
            "visible_renderer_count," +
            "triangle_count," +
            "draw_call_count," +
            "camera_speed," +
            "camera_rotation_speed," +
            "avg_screen_coverage," +   // SSIM proxy used in Python reward
            "previous_bias," +
            "recent_lod_switch_count," +
            "floor_dwell_score," +
            "ceiling_dwell_score," +
            "lod_bias_before_action," +
            "raw_action_delta," +      // pre-guardrail policy/fallback intent
            "action_delta," +          // applied bias delta after guardrails
            "lod_bias_after_action," +
            "selected_target_ms," +
            "target_source," +
            "scene_target_ready," +
            "collection_mode," +
            "gpu_ms_at_target_lock"
        );

        Debug.Log($"[RLRolloutLogger] Episode {_episodeIndex} started → {_currentFilePath}");
    }

    public void EndEpisode()
    {
        if (_episodeDone) return;
        _episodeDone   = true;
        _loggingActive = false;

        FlushAndClose();
        Debug.Log($"[RLRolloutLogger] Episode {_episodeIndex} ended. {_stepIndex} steps logged.");

        _episodeIndex++;

        if (maxEpisodes > 0 && _episodeIndex >= maxEpisodes)
        {
            CollectionComplete = true;
            _evalLogger?.NotifyRolloutEpisodeEnd(); // finalise last eval episode before stopping
            Debug.Log("[RLRolloutLogger] Collection complete. All episodes logged.");
            return;
        }

        // Notify evaluation logger before resetting state
        _evalLogger?.NotifyRolloutEpisodeEnd();

        // Reset environment for next episode
        QualitySettings.lodBias = 1.0f;
        LastRawActionDelta      = 0f;
        LastActionDelta         = 0f;
        _extractor.ResetEpisodeState();
        _policyController?.ResetEpisode(); // sync bias + dwell counter in controller

        StartEpisode();
    }

    // ── Per-Frame Logging ──────────────────────────────────────────────────

    // LateUpdate: runs after RLPolicyController.Update() has written LastActionDelta
    // and applied the new bias, so lod_bias_after_action is already committed.
    void LateUpdate()
    {
        if (!_loggingActive || _episodeDone)   return;
        if (!_extractor.IsReady)               return;
        if (!ShouldLogDecisionFrame)           return;

        float gpuMs = _extractor.GpuFrameTime;
        float cpuMs = _extractor.CpuFrameTime;

        // Discard invalid frames (GPU driver stall, warmup).
        // Python df_samples_clean will additionally filter outliers.
        if (gpuMs <= 0f || cpuMs <= 0f) return;

        float[] raw = _extractor.RawFeatures;
        if (raw == null) return;

        float biasAfter  = QualitySettings.lodBias;
        float biasBefore = LastBiasBeforeAction;  // snapshotted in RLPolicyController.ApplyWithGuardrails()
        float selectedTargetMs = _extractor.SelectedTargetMs;
        string targetSource = _extractor.SelectedTargetSource;
        int sceneTargetReady = RLFeatureExtractor.SceneTargetReady ? 1 : 0;
        string collectionMode = _extractor.CollectionMode;
        float gpuMsAtTargetLock = RLFeatureExtractor.SceneTargetReady ? RLFeatureExtractor.SceneTTargetMs : float.NaN;

        // Write row in invariant culture (period decimal separator)
        string row = string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2:F4},{3:F4},{4:F2},{5:F0},{6:F0},{7:F0},{8:F4},{9:F4},{10:F6},{11:F4},{12:F0},{13:F4},{14:F4},{15:F4},{16:F4},{17:F4},{18:F4},{19:F4},{20},{21},{22},{23:F4}",
            _episodeIndex,          // 0  episode
            _stepIndex,             // 1  step
            cpuMs,                  // 2  cpu_frame_time
            gpuMs,                  // 3  gpu_frame_time  ← reward signal
            raw[2],                 // 4  fps
            raw[3],                 // 5  visible_renderer_count
            raw[4],                 // 6  triangle_count
            raw[5],                 // 7  draw_call_count
            raw[6],                 // 8  camera_speed
            raw[7],                 // 9  camera_rotation_speed
            raw[8],                 // 10 avg_screen_coverage (SSIM proxy)
            raw[9],                 // 11 previous_bias
            raw[10],                // 12 recent_lod_switch_count
            raw[11],                // 13 floor_dwell_score
            raw[12],                // 14 ceiling_dwell_score
            biasBefore,             // 15 lod_bias_before_action
            LastRawActionDelta,     // 16 raw_action_delta
            LastActionDelta,        // 17 action_delta
            biasAfter,              // 18 lod_bias_after_action
            selectedTargetMs,       // 19 selected_target_ms
            targetSource,           // 20 target_source
            sceneTargetReady,       // 21 scene_target_ready
            collectionMode,         // 22 collection_mode
            gpuMsAtTargetLock       // 23 gpu_ms_at_target_lock
        );

        _rowBuffer.Add(row);
        _stepIndex++;

        if (_rowBuffer.Count >= bufferFlushInterval)
            FlushBuffer();

        if (stepsPerEpisode > 0 && _stepIndex >= stepsPerEpisode)
            EndEpisode();
    }

    // ── IO Helpers ─────────────────────────────────────────────────────────

    private void FlushBuffer()
    {
        if (_writer == null || _rowBuffer == null) return;
        foreach (string r in _rowBuffer) _writer.WriteLine(r);
        _rowBuffer.Clear();
        _writer.Flush();
    }

    private void FlushAndClose()
    {
        if (_writer == null) return;
        FlushBuffer();
        _writer.Close();
        _writer.Dispose();
        _writer = null;
    }

    void OnDestroy()
    {
        FlushAndClose();
    }
}
