// File: RuntimeLODApplicator.cs
// Applies predicted threshold vectors to all LODGroup components in the scene.
// Uses LODGroup.SetLODs() to replace screenRelativeTransitionHeight values.
// Enforces valid ranges (0..1) and monotonically decreasing order.
using UnityEngine;
using System.Collections.Generic;

public class RuntimeLODApplicator : MonoBehaviour
{
    [Header("References")]
    public LODThresholdPredictor predictor;

    [Header("Configuration")]
    [Tooltip("Minimum difference between adjacent thresholds.")]
    public float minThresholdGap = 0.01f;

    LODGroup[] lodGroups;
    float[] lastAppliedThresholds;

    void Start()
    {
        lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);
        Debug.Log($"[RuntimeLODApplicator] Found {lodGroups.Length} LODGroups to manage.");
    }

    void LateUpdate()
    {
        if (predictor == null || predictor.lastPrediction == null) return;
        if (lodGroups == null || lodGroups.Length == 0) return;

        float[] predicted = predictor.lastPrediction;

        // skip if unchanged
        if (lastAppliedThresholds != null && ArraysEqual(predicted, lastAppliedThresholds))
            return;

        // sanitize: clamp and enforce monotonic
        float[] sanitized = SanitizeThresholds(predicted);

        // apply to all LODGroups
        foreach (LODGroup group in lodGroups)
        {
            ApplyToGroup(group, sanitized);
        }

        lastAppliedThresholds = (float[])sanitized.Clone();
    }

    void ApplyToGroup(LODGroup group, float[] thresholds)
    {
        LOD[] lods = group.GetLODs();
        if (lods.Length == 0) return;

        int count = Mathf.Min(lods.Length, thresholds.Length);
        for (int i = 0; i < count; i++)
        {
            lods[i].screenRelativeTransitionHeight = thresholds[i];
        }

        // for any remaining LOD levels beyond prediction length, set to near zero
        for (int i = count; i < lods.Length; i++)
        {
            lods[i].screenRelativeTransitionHeight = 0.001f;
        }

        group.SetLODs(lods);
        group.RecalculateBounds();
    }

    float[] SanitizeThresholds(float[] raw)
    {
        float[] result = new float[raw.Length];

        // clamp to valid range
        for (int i = 0; i < raw.Length; i++)
        {
            result[i] = Mathf.Clamp01(raw[i]);
        }

        // enforce monotonically decreasing
        for (int i = 1; i < result.Length; i++)
        {
            if (result[i] >= result[i - 1])
            {
                result[i] = result[i - 1] - minThresholdGap;
                if (result[i] < 0f) result[i] = 0f;
            }
        }

        return result;
    }

    bool ArraysEqual(float[] a, float[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (Mathf.Abs(a[i] - b[i]) > 0.0001f) return false;
        }
        return true;
    }
}
