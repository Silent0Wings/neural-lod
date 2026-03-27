// File: RotationSampler.cs
// Orchestrates the full profiling capture:
//   For each GridPoint:
//     For each rotation:
//       WarmUpPosition once (camera move + GPU stabilize)
//       CaptureAllLods (mini-warmup between each LOD level)
// Runs as a coroutine so Unity can render frames between captures.
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class RotationSampler : MonoBehaviour
{
    [Header("Rotation Coverage")]
    // player realistic angles only
    // pitch: slight down to slight up, no sky facing extremes
    // yaw: 8 cardinal and diagonal directions
    // roll: fixed at 0 players do not roll
    public float[] pitchAngles = { -15f, 0f, 15f };
    public float[] yawAngles   = { 0f, 45f, 90f, 135f, 180f, 225f, 270f, 315f };
    public float[] rollAngles  = { 0f };

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
    [HideInInspector] public float progress = 0f;

    public void StartSampling()
    {
        StartCoroutine(RunSampling());
    }

    IEnumerator RunSampling()
    {
        isRunning = true;
        allRecords.Clear();

        int lodCount = DetectLodLevelCount();
        if (maxLodLevel >= 0)
            lodCount = Mathf.Min(lodCount, maxLodLevel + 1);

        List<Vector3> rotations = BuildRotationList();

        // total = points x rotations only, lod sweep is inside a single coroutine call per position
        int totalPositions = gridPoints.Count * rotations.Count;
        int currentPosition = 0;

        Debug.Log($"[RotationSampler] Starting: {gridPoints.Count} points, " +
                  $"{rotations.Count} rotations, {lodCount} LOD levels. " +
                  $"Total positions: {totalPositions}");

        foreach (GridPoint point in gridPoints)
        {
            foreach (Vector3 rot in rotations)
            {
                // warmup once for this (point, rotation)
                yield return StartCoroutine(
                    metricsRecorder.WarmUpPosition(point, rot, profilingCamera)
                );

                // capture all LOD levels with mini-warmup between each
                yield return StartCoroutine(
                    metricsRecorder.CaptureAllLods(point, rot, lodCount, profilingCamera)
                );

                allRecords.AddRange(metricsRecorder.lodRecords);

                currentPosition++;
                progress = (float)currentPosition / totalPositions;
            }
        }

        isRunning = false;
        Debug.Log($"[RotationSampler] Complete. {allRecords.Count} records captured.");
    }

    List<Vector3> BuildRotationList()
    {
        List<Vector3> rotations = new List<Vector3>();

        foreach (float pitch in pitchAngles)
            foreach (float yaw in yawAngles)
                foreach (float roll in rollAngles)
                    rotations.Add(new Vector3(pitch, yaw, roll));

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
