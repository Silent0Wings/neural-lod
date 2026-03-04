using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

/// <summary>
/// RunTerminator
/// Exits play mode (Editor) or quits application (Build).
/// Two termination modes — toggle via Inspector:
///   - PathComplete : stops when CameraPathAnimator finishes
///   - RowCount     : stops when MetricLogger hits target row count
/// Set controlledByOrchestrator = true to disable self-termination
/// when DataCollectionOrchestrator is managing the run loop.
/// </summary>
public class RunTerminator : MonoBehaviour
{
    public enum TerminationMode
    {
        PathComplete,
        RowCount
    }

    [Header("Settings")]
    public TerminationMode mode                   = TerminationMode.RowCount;
    public float           checkInterval          = 0.5f;
    public bool            controlledByOrchestrator = false; // ← set true when orchestrator is active

    [Header("References")]
    public CameraPathAnimator cpa;
    public MetricLogger       logger;

    private float _timer = 0f;

    // ─────────────────────────────────────────
    void Awake()
    {
        if (cpa == null)
            cpa = FindFirstObjectByType<CameraPathAnimator>();

        if (logger == null)
            logger = FindFirstObjectByType<MetricLogger>();

        if (cpa == null)
            Debug.LogWarning("[RunTerminator] No CameraPathAnimator found in scene.");

        if (logger == null)
            Debug.LogWarning("[RunTerminator] No MetricLogger found in scene.");

        if (mode == TerminationMode.RowCount && cpa != null)
            cpa.loop = true;
    }

    // ─────────────────────────────────────────
    void Update()
    {
        // yield control to orchestrator if active
        if (controlledByOrchestrator) return;

        _timer += Time.deltaTime;
        if (_timer < checkInterval) return;
        _timer = 0f;

        switch (mode)
        {
            case TerminationMode.PathComplete:
                if (cpa != null && cpa.completed)
                {
                    Debug.Log("[RunTerminator] Path completed — terminating.");
                    Terminate();
                }
                break;

            case TerminationMode.RowCount:
                if (logger != null && logger.loggingComplete)
                {
                    Debug.Log("[RunTerminator] Row target reached — terminating.");
                    Terminate();
                }
                break;
        }
    }

    // ─────────────────────────────────────────
    public void Terminate()
    {
#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }
}
