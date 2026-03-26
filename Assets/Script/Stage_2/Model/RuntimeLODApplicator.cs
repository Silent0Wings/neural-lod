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
        if (predictor == null)
            predictor = GetComponent<LODThresholdPredictor>();

        if (predictor == null)
            UnityEngine.Debug.LogError("[RuntimeLODApplicator] No LODThresholdPredictor found.");

        lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);
        UnityEngine.Debug.Log($"[RuntimeLODApplicator] Found {lodGroups.Length} LODGroups.");
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

        // final guard before SetLODs
        bool valid = true;
        for (int i = 1; i < lods.Length; i++)
        {
            if (lods[i].screenRelativeTransitionHeight >= lods[i - 1].screenRelativeTransitionHeight)
            {
                valid = false;
                break;
            }
        }
        if (!valid) return;

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

        // enforce strictly decreasing from index 0 downward
        // start from highest LOD and work down
        for (int i = 1; i < result.Length; i++)
        {
            float maxAllowed = result[i - 1] - minThresholdGap;
            if (result[i] >= result[i - 1])
                result[i] = Mathf.Max(0f, maxAllowed);
        }

        // second pass — ensure no value went negative and broke order
        for (int i = result.Length - 1; i >= 1; i--)
        {
            if (result[i] < 0f) result[i] = 0f;
            if (result[i - 1] <= result[i])
                result[i - 1] = result[i] + minThresholdGap;
            result[i - 1] = Mathf.Clamp01(result[i - 1]);
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
