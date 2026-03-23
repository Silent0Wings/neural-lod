using UnityEngine;
using System.Collections.Generic;
using UnityEngine.UI;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class DataCollectionOrchestrator : MonoBehaviour
{
    [Header("Run Parameters")]
    public float[] biasValues = { 0.1f, 0.5f, 1.0f, 2.0f };
    public float[] moveSpeeds = { 3.0f, 5.0f, 8.0f };
    public float[] rotateSpeeds = { 3.0f, 5.0f };

    [Header("Settings")]
    public float warmupDelay = 1.5f;

    [Header("References")]
    public CameraPathAnimator cpa;
    public MetricLogger logger;
    public LODBiasController lodController;
    public RunTerminator terminator;
    public Slider progressionSlider;



    [Header("Status (read-only)")]
    [Range(0f, 1f)] public float progression = 0f;
    [SerializeField] private int _currentRun = 0;
    [SerializeField] private int _totalRuns = 0;
    [SerializeField] private float _currentBias = 0f;
    [SerializeField] private float _currentSpeed = 0f;
    [SerializeField] private float _currentRot = 0f;
    [SerializeField] private bool _allDone = false;

    private List<(float bias, float speed, float rot)> _runs;
    private bool _waitingForWarmup = false;
    private float _warmupTimer = 0f;
    private bool _runStarted = false;

    void Awake()
    {
        if (cpa == null) cpa = FindFirstObjectByType<CameraPathAnimator>();
        if (logger == null) logger = FindFirstObjectByType<MetricLogger>();
        if (lodController == null) lodController = FindFirstObjectByType<LODBiasController>();
        if (terminator == null) terminator = FindFirstObjectByType<RunTerminator>();

        if (terminator != null)
            terminator.controlledByOrchestrator = true;

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

    private void StartRun(int index)
    {
        var (bias, speed, rot) = _runs[index];

        _currentBias = bias;
        _currentSpeed = speed;
        _currentRot = rot;

        lodController.UpdateBias(bias);
        cpa.moveSpeed = speed;
        cpa.rotateSpeed = rot;

        logger.ResetLogger(index, _currentSpeed, _currentRot);
        logger.enabled = false;

        cpa.ResetPath();
        cpa.IsPaused = true;

        _waitingForWarmup = true;
        _warmupTimer = 0f;
        _runStarted = false;

        Debug.Log($"[Orchestrator] Run {index + 1}/{_totalRuns} — " +
                  $"bias={bias} | speed={speed} | rot={rot}");
    }

    void Update()
    {
        if (_allDone) return;

        if (_waitingForWarmup)
        {
            _warmupTimer += Time.deltaTime;

            if (_warmupTimer >= warmupDelay)
            {
                _waitingForWarmup = false;
                _runStarted = true;
                cpa.IsPaused = false;
                logger.enabled = true;

                Debug.Log($"[Orchestrator] Warmup done — logging started for run {_currentRun + 1}.");
            }
            return;
        }

        if (_runStarted && logger.loggingComplete)
        {
            ForceAdvance();
        }
    }

    private void ForceAdvance()
    {
        _currentRun++;
        if (_totalRuns > 0) progression = (float)_currentRun / _totalRuns;
        if (progressionSlider != null)
            progressionSlider.value = progression;

        if (_currentRun >= _totalRuns)
        {
            _allDone = true;
            Debug.Log($"[Orchestrator] All {_totalRuns} runs complete. Requesting termination.");

            if (terminator != null)
                terminator.Terminate();
            else
                Debug.LogError("[Orchestrator] RunTerminator reference missing.");
        }
        else
        {
            StartRun(_currentRun);
        }
    }
}
