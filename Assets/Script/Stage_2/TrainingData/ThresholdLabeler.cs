// File: ThresholdLabeler.cs
// Groups SampleRecords by (pointId, rotation) viewpoint.
// For each viewpoint, determines the optimal LOD threshold distribution:
//   - Finds the highest LOD index that meets the target frame time budget.
//   - Maps that LOD index back to threshold values from the LODObjectInfo baseline.
// Produces LabelledSample entries attached to the ProfilingSession.
using UnityEngine;
using System.Collections.Generic;

public class ThresholdLabeler : MonoBehaviour
{
    [Header("Labeling Configuration")]
    [Tooltip("Target frame time in ms (e.g. 16.67 for 60 FPS).")]
    public float targetFrameTimeMs = 16.67f;

    /// runs the labeling pass over a completed profiling session
    public void LabelSession(ProfilingSession session)
    {
        session.labels.Clear();

        // group records by (pointId, rotation) key
        Dictionary<string, List<SampleRecord>> groups = new Dictionary<string, List<SampleRecord>>();

        foreach (SampleRecord r in session.samples)
        {
            string key = $"{r.pointId}_{r.rotationAngles.x}_{r.rotationAngles.y}_{r.rotationAngles.z}";
            if (!groups.ContainsKey(key))
                groups[key] = new List<SampleRecord>();
            groups[key].Add(r);
        }

        // baseline thresholds from first object for scaling
        // in a full implementation each object would get its own label
        List<float> baselineThresholds = new List<float>();
        if (session.objects.Count > 0)
            baselineThresholds = new List<float>(session.objects[0].thresholds);

        foreach (var kvp in groups)
        {
            List<SampleRecord> records = kvp.Value;

            // sort by LOD level ascending (LOD 0 = highest detail)
            records.Sort((a, b) => a.lodLevel.CompareTo(b.lodLevel));

            // find highest LOD index that meets frame time budget
            // frame time = max(gpu, cpu) to capture true bottleneck
            int bestLod = 0; // default to highest detail
            for (int i = records.Count - 1; i >= 0; i--)
            {
                float frameTime = Mathf.Max(records[i].meanGpuTimeMs, records[i].meanCpuTimeMs);
                if (frameTime > 0 && frameTime <= targetFrameTimeMs)
                {
                    bestLod = records[i].lodLevel;
                    break;
                }
            }

            // if nothing meets budget, use the LOD with lowest frame time
            if (bestLod == 0 && records.Count > 0)
            {
                float lowestTime = float.MaxValue;
                foreach (SampleRecord r in records)
                {
                    float ft = Mathf.Max(r.meanGpuTimeMs, r.meanCpuTimeMs);
                    if (ft > 0 && ft < lowestTime)
                    {
                        lowestTime = ft;
                        bestLod = r.lodLevel;
                    }
                }
            }

            // compute optimal thresholds by scaling baseline
            // scale factor: ratio of bestLod to max LOD index
            List<float> optimalThresholds = ComputeOptimalThresholds(baselineThresholds, bestLod, records.Count);

            LabelledSample label = new LabelledSample();
            label.pointId = records[0].pointId;
            label.rotationAngles = records[0].rotationAngles;
            label.optimalThresholds = optimalThresholds;

            session.labels.Add(label);
        }

        Debug.Log($"[ThresholdLabeler] Labeled {session.labels.Count} viewpoints.");
    }

    List<float> ComputeOptimalThresholds(List<float> baseline, int bestLod, int totalLods)
    {
        List<float> result = new List<float>();

        if (baseline.Count == 0)
        {
            // no baseline data, return a single scalar
            float scalar = totalLods > 1 ? (float)bestLod / (totalLods - 1) : 0f;
            result.Add(scalar);
            return result;
        }

        // shift thresholds based on how aggressive the optimal LOD is
        // higher bestLod = more aggressive = raise thresholds to trigger earlier transitions
        float shiftFactor = totalLods > 1 ? (float)bestLod / (totalLods - 1) : 0f;

        for (int i = 0; i < baseline.Count; i++)
        {
            // relax threshold toward 1.0 as lod gets more aggressive
            float shifted = Mathf.Lerp(baseline[i], 1.0f, shiftFactor * 0.5f);
            // clamp to valid range and ensure monotonic decrease
            shifted = Mathf.Clamp01(shifted);
            result.Add(shifted);
        }

        // enforce monotonic decreasing
        for (int i = 1; i < result.Count; i++)
        {
            if (result[i] >= result[i - 1])
                result[i] = result[i - 1] * 0.9f;
        }

        return result;
    }
}
