// File: LODMetricsRecorder.cs
// For a given (gridPoint, rotation, lodLevel):
//   1) Forces all LODGroups to the specified level.
//   2) Renders warm-up frames to stabilize GPU.
//   3) Captures CPU/GPU frame times over a sampling window.
//   4) Returns a SampleRecord with aggregated metrics.
// Requires: Project Settings > Player > Enable Frame Timing Stats = ON.
using UnityEngine;
using System.Collections;

public class LODMetricsRecorder : MonoBehaviour
{
    [Header("Sampling Configuration")]
    public int warmUpFrames = 10;
    public int sampleFrames = 30;

    // references set by the orchestrator
    [HideInInspector] public LODGroup[] lodGroups;

    // latest result after CaptureMetrics completes
    [HideInInspector] public SampleRecord lastRecord;

    FrameTiming[] timingBuffer = new FrameTiming[1];

    /// coroutine that captures metrics for one configuration
    public IEnumerator CaptureMetrics(GridPoint point, Vector3 eulerAngles, int lodLevel, Camera cam)
    {
        // position and orient camera
        cam.transform.position = point.coordinates;
        cam.transform.rotation = Quaternion.Euler(eulerAngles);

        // force all LODGroups to the specified level
        foreach (LODGroup group in lodGroups)
        {
            group.ForceLOD(lodLevel);
        }

        // warm-up: let GPU stabilize
        for (int i = 0; i < warmUpFrames; i++)
        {
            yield return null;
        }

        // sampling window
        float totalGpu = 0f;
        float totalCpu = 0f;
        int validSamples = 0;

        for (int i = 0; i < sampleFrames; i++)
        {
            yield return null; // wait one frame

            FrameTimingManager.CaptureFrameTimings();
            uint count = FrameTimingManager.GetLatestTimings(1, timingBuffer);

            if (count > 0)
            {
                float gpu = (float)timingBuffer[0].gpuFrameTime;
                float cpu = (float)timingBuffer[0].cpuMainThreadFrameTime;

                // skip zero readings from warmup lag
                if (gpu > 0.001f)
                {
                    totalGpu += gpu;
                    totalCpu += cpu;
                    validSamples++;
                }
            }
        }

        // reset forced LOD
        foreach (LODGroup group in lodGroups)
        {
            group.ForceLOD(-1);
        }

        // build record
        lastRecord = new SampleRecord();
        lastRecord.pointId = point.pointId;
        lastRecord.rotationAngles = eulerAngles;
        lastRecord.lodLevel = lodLevel;

        if (validSamples > 0)
        {
            lastRecord.meanGpuTimeMs = totalGpu / validSamples;
            lastRecord.meanCpuTimeMs = totalCpu / validSamples;
            float avgFrameTimeMs = Mathf.Max(lastRecord.meanGpuTimeMs, lastRecord.meanCpuTimeMs);
            lastRecord.meanFps = avgFrameTimeMs > 0.001f ? 1000f / avgFrameTimeMs : 0f;
        }
        else
        {
            lastRecord.meanGpuTimeMs = -1f;
            lastRecord.meanCpuTimeMs = -1f;
            lastRecord.meanFps = -1f;
            Debug.LogWarning($"[MetricsRecorder] No valid timing samples for point {point.pointId}, " +
                             $"rot {eulerAngles}, LOD {lodLevel}.");
        }
    }
}
