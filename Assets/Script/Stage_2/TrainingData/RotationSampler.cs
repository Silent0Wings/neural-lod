// File: RotationSampler.cs
// Orchestrates the full profiling capture:
//   For each GridPoint:
//     For each rotation (discrete Euler steps):
//       For each LOD level:
//         Delegate to LODMetricsRecorder to capture a SampleRecord.
// Runs as a coroutine so Unity can render frames between captures.
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class RotationSampler : MonoBehaviour
{
    [Header("Rotation Coverage")]
    public float[] pitchAngles = { -60f, 0f, 60f };       // 3
    public float[] yawAngles = { 0f, 90f, 180f, 270f };  // 4
    public float[] rollAngles = { -30f, 0f, 30f };         // 3

    [Header("LOD Levels to Sample")]
    [Tooltip("Max LOD index to sweep. -1 = auto-detect from first LODGroup.")]
    public int maxLodLevel = -1;

    [Header("References")]
    public Camera profilingCamera;

    // set by the orchestrator before starting
    [HideInInspector] public List<GridPoint> gridPoints;
    [HideInInspector] public LODGroup[] lodGroups;
    [HideInInspector] public LODMetricsRecorder metricsRecorder;

    // results
    [HideInInspector] public List<SampleRecord> allRecords = new List<SampleRecord>();
    [HideInInspector] public bool isRunning = false;
    [HideInInspector] public float progress = 0f; // 0..1

    /// starts the full sampling sweep
    public void StartSampling()
    {
        StartCoroutine(RunSampling());
    }

    IEnumerator RunSampling()
    {
        isRunning = true;
        allRecords.Clear();

        // detect max LOD level if auto
        int lodCount = DetectLodLevelCount();
        if (maxLodLevel >= 0)
            lodCount = Mathf.Min(lodCount, maxLodLevel + 1);

        // build rotation list
        List<Vector3> rotations = BuildRotationList();

        int totalSteps = gridPoints.Count * rotations.Count * lodCount;
        int currentStep = 0;

        Debug.Log($"[RotationSampler] Starting: {gridPoints.Count} points, " +
                  $"{rotations.Count} rotations, {lodCount} LOD levels = {totalSteps} total captures.");

        foreach (GridPoint point in gridPoints)
        {
            foreach (Vector3 rot in rotations)
            {
                for (int lod = 0; lod < lodCount; lod++)
                {
                    yield return StartCoroutine(
                        metricsRecorder.CaptureMetrics(point, rot, lod, profilingCamera)
                    );

                    allRecords.Add(metricsRecorder.lastRecord);
                    currentStep++;
                    progress = (float)currentStep / totalSteps;
                }
            }
        }

        isRunning = false;
        Debug.Log($"[RotationSampler] Complete. {allRecords.Count} records captured.");
    }

    List<Vector3> BuildRotationList()
    {
        List<Vector3> rotations = new List<Vector3>();

        foreach (float pitch in pitchAngles)
        {
            foreach (float yaw in yawAngles)
            {
                foreach (float roll in rollAngles)
                {
                    rotations.Add(new Vector3(pitch, yaw, roll));
                }
            }
        }

        return rotations;
    }

    int DetectLodLevelCount()
    {
        if (lodGroups == null || lodGroups.Length == 0) return 1;

        int maxFound = 0;
        foreach (LODGroup group in lodGroups)
        {
            int count = group.GetLODs().Length;
            if (count > maxFound) maxFound = count;
        }

        return Mathf.Max(maxFound, 1);
    }
}
