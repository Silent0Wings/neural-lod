// File: ThresholdLabeler.cs
// Groups SampleRecords by (pointId, rotation) viewpoint.
// For each viewpoint, determines the optimal LOD threshold distribution:
//   - Finds the highest LOD index that meets the target frame time budget.
//   - Maps that LOD index back to threshold values from the LODObjectInfo baseline.
// Produces LabelledSample entries attached to the ProfilingSession.
//
// FIXES APPLIED:
//   FIX-3a: targetFrameTimeMs default changed from 16.67 to 4.5.
//            At 16.67ms (60fps budget) every LOD level in the scene met budget
//            (scene runs at 150-190fps), causing bestLod=0 for 82% of viewpoints
//            and collapsing labels to a single value.
//            4.5ms targets ~222fps and forces meaningful differentiation across
//            LOD levels given the scene's actual GPU time range of 3.4-5.3ms.
//            Tune this value in the Inspector to match your scene's GPU range.
//
//   FIX-3b: Oracle now uses GPU time only instead of Mathf.Max(gpu, cpu).
//            The scene is CPU-bound (scripting/logging overhead), so CPU time
//            is near-constant across LOD levels and carries no LOD quality signal.
//            Using max(gpu,cpu) caused the CPU bottleneck to dominate and made
//            every LOD level appear equally slow, preventing label differentiation.

using UnityEngine;
using System.Collections.Generic;

public class ThresholdLabeler : MonoBehaviour
{
    [Header("Labeling Configuration")]
    [Tooltip("GPU frame time budget in ms. Set tighter than your scene's LOD0 GPU time " +
             "to force label spread. Example: if GPU ranges 3.4-5.3ms, use 4.5ms.")]
    // FIX-3a: was 16.67f — too loose, caused 82% label collapse to bestLod=0
    public float targetFrameTimeMs = 4.5f;

    /// runs the labeling pass over a completed profiling session
    public void LabelSession(ProfilingSession session)
    {
        session.labels.Clear();

        // group records by (pointId, rotation) key
        Dictionary<string, List<SampleRecord>> groups = new Dictionary<string, List<SampleRecord>>();

        foreach (SampleRecord r in session.samples)
        {
            // use fixed-precision formatting to avoid float string key fragility
            string key = $"{r.pointId}_{r.rotationAngles.x:F4}_{r.rotationAngles.y:F4}_{r.rotationAngles.z:F4}";
            if (!groups.ContainsKey(key))
                groups[key] = new List<SampleRecord>();
            groups[key].Add(r);
        }

        // baseline thresholds from first object for scaling
        // in a full implementation each object would get its own label
        List<float> baselineThresholds = new List<float>();
        if (session.objects.Count > 0)
            baselineThresholds = new List<float>(session.objects[0].thresholds);

        int labeledCount   = 0;
        int collapsedCount = 0;

        foreach (var kvp in groups)
        {
            List<SampleRecord> records = kvp.Value;

            // sort by LOD level ascending (LOD 0 = highest detail)
            records.Sort((a, b) => a.lodLevel.CompareTo(b.lodLevel));

            // FIX-3b: use GPU time only — CPU time is near-constant in this scene
            // (CPU bottleneck is scripting overhead, not LOD mesh complexity)
            // Original: float frameTime = Mathf.Max(records[i].meanGpuTimeMs, records[i].meanCpuTimeMs)
            // Fixed:    float frameTime = records[i].meanGpuTimeMs

            // find highest LOD index whose GPU time meets the frame time budget
            int bestLod    = 0; // default to highest detail
            bool foundValid = false;
            for (int i = records.Count - 1; i >= 0; i--)
            {
                float frameTime = records[i].meanGpuTimeMs; // FIX-3b: GPU only
                if (frameTime > 0f && frameTime <= targetFrameTimeMs)
                {
                    bestLod    = records[i].lodLevel;
                    foundValid = true;
                    break;
                }
            }

            // if nothing meets budget, pick LOD with lowest GPU time
            if (!foundValid && records.Count > 0)
            {
                float lowestTime = float.MaxValue;
                foreach (SampleRecord r in records)
                {
                    float ft = r.meanGpuTimeMs; // FIX-3b: GPU only
                    if (ft > 0f && ft < lowestTime)
                    {
                        lowestTime = ft;
                        bestLod    = r.lodLevel;
                    }
                }
                collapsedCount++;
            }

            // compute optimal thresholds by scaling baseline
            List<float> optimalThresholds = ComputeOptimalThresholds(baselineThresholds, bestLod, records.Count);

            LabelledSample label = new LabelledSample();
            label.pointId           = records[0].pointId;
            label.rotationAngles    = records[0].rotationAngles;
            label.optimalThresholds = optimalThresholds;

            session.labels.Add(label);
            labeledCount++;
        }

        Debug.Log($"[ThresholdLabeler] Labeled {labeledCount} viewpoints. " +
                  $"Budget fallback (no LOD met target): {collapsedCount} viewpoints. " +
                  $"Target GPU budget: {targetFrameTimeMs}ms.");
    }

    List<float> ComputeOptimalThresholds(List<float> baseline, int bestLod, int totalLods)
    {
        List<float> result = new List<float>();

        if (baseline.Count == 0)
        {
            // no baseline data — return a single normalised scalar
            float scalar = totalLods > 1 ? (float)bestLod / (totalLods - 1) : 0f;
            result.Add(scalar);
            return result;
        }

        // shift thresholds based on how aggressive the optimal LOD is
        // higher bestLod = more aggressive = raise thresholds to trigger earlier transitions
        float shiftFactor = totalLods > 1 ? (float)bestLod / (totalLods - 1) : 0f;

        for (int i = 0; i < baseline.Count; i++)
        {
            // relax threshold toward 1.0 as LOD gets more aggressive
            float shifted = Mathf.Lerp(baseline[i], 1.0f, shiftFactor * 0.5f);
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
