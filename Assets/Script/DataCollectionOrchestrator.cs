using UnityEngine;
using System.Collections.Generic;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// DataCollectionOrchestrator (Fixed)
/// Automates data collection across all combinations of:
///   bias values × move speeds × rotate speeds

/// 
/// Workflow per run:
///   1. Apply bias, moveSpeed, rotateSpeed
///   2. Reset MetricLogger (new CSV file)
///   3. Reset CameraPathAnimator (camera back to node 0, paused)
///   4. Wait warmupDelay seconds (camera frozen)
///   5. Unpause camera, enable logger
///   6. Wait for MetricLogger.loggingComplete
///   7. Advance to next combination
///   8. When all done — terminate play mode
/// </summary>
public class DataCollectionOrchestrator : MonoBehaviour
{
    // ─
    // Run parameters — set in Inspector
    // ─
    [Header("Run Parameters")]
    public float[] biasValues   = { 0.1f, 0.5f, 1.0f, 2.0f };
    public float[] moveSpeeds   = { 3.0f, 5.0f, 8.0f };
    public float[] rotateSpeeds = { 3.0f, 5.0f };

    [Header("Settings")]
    public float warmupDelay = 1.5f; // seconds to wait after reset before logging starts

    [Header("References")]
    public CameraPathAnimator cpa;
    public MetricLogger       logger;
    public LODBiasController  lodController;
    public RunTerminator      terminator;

    // ─
    // Status — read-only in Inspector
    // ─
    [Header("Status (read-only)")]
    [SerializeField] private int   _currentRun   = 0;
    [SerializeField] private int   _totalRuns    = 0;
    [SerializeField] private float _currentBias  = 0f;
    [SerializeField] private float _currentSpeed = 0f;
    [SerializeField] private float _currentRot   = 0f;
    [SerializeField] private bool  _allDone      = false;

    // ─
    // Internal
    // ─
    private List<(float bias, float speed, float rot)> _runs;
    private bool  _waitingForWarmup = false;
    private float _warmupTimer      = 0f;
    private bool  _runStarted       = false;

    // ─
    // Init
    // ─
    void Awake()
    {
        if (cpa          == null) cpa          = FindFirstObjectByType<CameraPathAnimator>();
        if (logger       == null) logger       = FindFirstObjectByType<MetricLogger>();
        if (lodController == null) lodController = FindFirstObjectByType<LODBiasController>();
        if (terminator   == null) terminator   = FindFirstObjectByType<RunTerminator>();

        // disable RunTerminator self-management
        if (terminator != null)
            terminator.controlledByOrchestrator = true;

        // ensure camera loops
        if (cpa != null)
            cpa.loop = true;

        BuildRunList();
    }

    void Start()
    {
        if (_runs.Count == 0)
        {
            Debug.LogError("[Orchestrator] No runs to execute — check parameter arrays.");
            return;
        }

        Debug.Log($"[Orchestrator] Starting — {_totalRuns} total runs.");
        StartRun(_currentRun);
    }

    // ─
    // Build cartesian product
    // ─
    private void BuildRunList()
    {
        _runs = new List<(float, float, float)>();

        foreach (float bias in biasValues)
            foreach (float speed in moveSpeeds)
                foreach (float rot in rotateSpeeds)
                    _runs.Add((bias, speed, rot));

        _totalRuns = _runs.Count;
        Debug.Log($"[Orchestrator] Built {_totalRuns} runs " +
                  $"({biasValues.Length} bias × {moveSpeeds.Length} speeds × {rotateSpeeds.Length} rotations).");
    }

    // ─
    // Start a specific run by index
    // ─
    private void StartRun(int index)
    {
        var (bias, speed, rot) = _runs[index];

        _currentBias  = bias;
        _currentSpeed = speed;
        _currentRot   = rot;

        // 1. Apply parameters
        lodController.UpdateBias(bias);
        cpa.moveSpeed   = speed;
        cpa.rotateSpeed = rot;

        // 2. Reset logger — opens new CSV
        logger.ResetLogger(index, _currentSpeed, _currentRot);

        // 3. Reset camera path — snaps to node 0
        cpa.ResetPath();

        // FIX #5: Pause camera during warmup so it stays at node 0
        cpa.IsPaused = true;

        // 4. Begin warmup — camera frozen, logger not yet recording
        _waitingForWarmup = true;
        _warmupTimer      = 0f;
        _runStarted       = false;

        Debug.Log($"[Orchestrator] Run {index + 1}/{_totalRuns} — " +
                  $"bias={bias} | speed={speed} | rot={rot}");
    }

    // ─
    // Update — manage warmup + run completion
    // ─
    void Update()
    {
        if (_allDone) return;

        // warmup phase — camera frozen, wait before enabling logger
        if (_waitingForWarmup)
        {
            _warmupTimer += Time.deltaTime;

            if (_warmupTimer >= warmupDelay)
            {
                _waitingForWarmup = false;
                _runStarted       = true;

                // FIX #5: Unpause camera and start logging simultaneously
                cpa.IsPaused      = false;
                logger.enabled    = true;

                Debug.Log($"[Orchestrator] Warmup done — logging started for run {_currentRun + 1}.");
            }
            return;
        }

        // wait for logger to hit row target
        if (_runStarted && logger.loggingComplete)
        {
            _currentRun++;

            if (_currentRun >= _totalRuns)
            {
                _allDone = true;
                Debug.Log($"[Orchestrator] All {_totalRuns} runs complete. Terminating.");
                Terminate();
            }
            else
            {
                StartRun(_currentRun);
            }
        }
    }

    // ─
    // Terminate
    // ─
    private void Terminate()
    {
#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
}
