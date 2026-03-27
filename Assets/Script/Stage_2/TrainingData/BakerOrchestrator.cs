// File: BakerOrchestrator.cs
// Master controller that runs the full Scene-Specific Neural LOD Baker pipeline:
//   1. collect_lods   -> LODCollector.ScanScene()
//   2. generate_grid  -> BoundingBoxGridGenerator.GenerateGrid()
//   3. sample          -> RotationSampler.StartSampling()
//   4. label           -> ThresholdLabeler.LabelSession()
//   5. export          -> DatasetExporter.Export()
// Attach to a GameObject with all component references assigned.
// Press Play, then call StartBaking() or toggle autoStart.
using UnityEngine;
using UnityEngine.SceneManagement;
using System.Collections;
using UnityEngine.UI;

[RequireComponent(typeof(LODCollector))]
[RequireComponent(typeof(BoundingBoxGridGenerator))]
[RequireComponent(typeof(RotationSampler))]
[RequireComponent(typeof(LODMetricsRecorder))] // profiler recorders managed by OnEnable/OnDisable in LODMetricsRecorder
[RequireComponent(typeof(ThresholdLabeler))]
[RequireComponent(typeof(DatasetExporter))]
public class BakerOrchestrator : MonoBehaviour
{
    [Header("Auto Start")]
    public bool autoStart = false;

    [Header("Profiling Camera")]
    public Camera profilingCamera;

    [Header("Progression Slider")]
    public Slider progressionSlider;


    // component refs
    LODCollector collector;
    BoundingBoxGridGenerator gridGen;
    RotationSampler sampler;
    LODMetricsRecorder recorder;
    ThresholdLabeler labeler;
    DatasetExporter exporter;

    // session
    ProfilingSession session;

    // re-entry guard
    private bool _isRunning = false;

    void Awake()
    {
        collector = GetComponent<LODCollector>();
        gridGen = GetComponent<BoundingBoxGridGenerator>();
        sampler = GetComponent<RotationSampler>();
        recorder = GetComponent<LODMetricsRecorder>();
        labeler = GetComponent<ThresholdLabeler>();
        exporter = GetComponent<DatasetExporter>();

        if (profilingCamera == null)
            profilingCamera = Camera.main;
    }

    void Start()
    {
        Application.runInBackground = true;
        if (autoStart)
            StartBaking();
    }

    public void StartBaking()
    {
        if (_isRunning)
        {
            Debug.LogWarning("[BakerOrchestrator] Pipeline already running. Ignoring duplicate call.");
            return;
        }
        StartCoroutine(RunPipeline());
    }

    IEnumerator RunPipeline()
    {
        _isRunning = true;
        Debug.Log("[BakerOrchestrator] === Pipeline Start ===");

        // step 1: collect LODs
        collector.ScanScene();

        // step 2: generate grid
        gridGen.GenerateGrid(collector.lodGroups);

        // step 3: configure and run sampling
        session = new ProfilingSession();
        session.sessionId = System.Guid.NewGuid().ToString("N").Substring(0, 8);
        session.sceneName = SceneManager.GetActiveScene().name;
        session.objects.AddRange(collector.collectedObjects);
        session.grid.AddRange(gridGen.gridPoints);

        sampler.gridPoints = gridGen.gridPoints;
        sampler.lodGroups = collector.lodGroups;
        sampler.metricsRecorder = recorder;
        sampler.profilingCamera = profilingCamera;
        recorder.lodGroups = collector.lodGroups;

        Debug.Log("[BakerOrchestrator] Starting sampling (this may take a long time)...");
        sampler.StartSampling();

        // wait for sampling to finish
        while (sampler.isRunning)
        {
            // log progress periodically
            yield return new WaitForSeconds(10f);
            if (sampler.isRunning)
            {
                if (progressionSlider != null)
                    progressionSlider.value = sampler.progress;
                Debug.Log($"[BakerOrchestrator] Sampling progress: {sampler.progress * 100f:F1}% " +
                          $"({(int)(sampler.progress * gridGen.gridPoints.Count * sampler.pitchAngles.Length * sampler.yawAngles.Length * sampler.rollAngles.Length)}" +
                          $"/{gridGen.gridPoints.Count * sampler.pitchAngles.Length * sampler.yawAngles.Length * sampler.rollAngles.Length} positions)");
            }
        }

        session.samples.AddRange(sampler.allRecords);
        Debug.Log($"[BakerOrchestrator] Sampling complete: {session.samples.Count} records.");

        // step 4: label
        labeler.LabelSession(session);

        // step 5: export
        exporter.Export(session);

        _isRunning = false;
        Debug.Log("[BakerOrchestrator] === Pipeline Complete ===");

        // EXIT after files are saved
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #else
        Application.Quit();
        #endif
    }
}
