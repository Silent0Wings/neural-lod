// File: RuntimeLODApplicator.cs
// Applies predicted threshold vectors to all LODGroup components in the scene.
// Uses LODGroup.SetLODs() to replace screenRelativeTransitionHeight values.
// Enforces valid ranges (0..1) and monotonically decreasing order.
using UnityEngine;

public class RuntimeLODApplicator : MonoBehaviour
{
    [Header("References")]
    public LODThresholdPredictor predictor;

    [Header("Configuration")]
    [Tooltip("Minimum difference between adjacent thresholds.")]
    public float minThresholdGap = 0.01f;

    LODGroup[] _lodGroups;

    // cached LOD arrays so ApplyToGroup never calls GetLODs() at runtime
    LOD[][] _cachedLODs;

    // pre-alloc sanitize and last-applied buffers to avoid per-frame heap allocs
    float[] _sanitizedBuffer;
    float[] _lastAppliedBuffer;

    void Start()
    {
        if (predictor == null)
            predictor = GetComponent<LODThresholdPredictor>();

        if (predictor == null)
            UnityEngine.Debug.LogError("[RuntimeLODApplicator] No LODThresholdPredictor found.");

        _lodGroups = Object.FindObjectsByType<LODGroup>(FindObjectsSortMode.None);
        UnityEngine.Debug.Log($"[RuntimeLODApplicator] Found {_lodGroups.Length} LODGroups.");

        CacheLODs();

        // disable stack trace extraction for logs in this session to prevent
        // StackTraceUtility.ExtractStackTrace allocations from any remaining log calls
        Application.SetStackTraceLogType(LogType.Warning, StackTraceLogType.None);
        Application.SetStackTraceLogType(LogType.Log,     StackTraceLogType.None);
    }

    // cache GetLODs() results once at startup so LateUpdate has zero GetLODs() calls
    void CacheLODs()
    {
        if (_lodGroups == null) { _cachedLODs = new LOD[0][]; return; }

        _cachedLODs = new LOD[_lodGroups.Length][];
        for (int g = 0; g < _lodGroups.Length; g++)
        {
            LODGroup group = _lodGroups[g];
            if (group != null)
                _cachedLODs[g] = group.GetLODs(); // one-time alloc at startup only
        }
    }

    void LateUpdate()
    {
        if (predictor == null || predictor.lastPrediction == null) return;
        if (_lodGroups == null || _lodGroups.Length == 0) return;

        float[] predicted = predictor.lastPrediction;

        // init pre-alloc buffers on first run or if prediction length changed
        if (_sanitizedBuffer == null || _sanitizedBuffer.Length != predicted.Length)
        {
            _sanitizedBuffer    = new float[predicted.Length];
            _lastAppliedBuffer  = new float[predicted.Length];
        }

        // skip if prediction unchanged
        if (ArraysEqual(predicted, _lastAppliedBuffer))
            return;

        // sanitize into pre-alloc buffer
        SanitizeThresholds(predicted, _sanitizedBuffer);

        // apply to all LODGroups
        int groupCount = _lodGroups.Length;
        for (int g = 0; g < groupCount; g++)
        {
            if (_lodGroups[g] != null && _cachedLODs[g] != null)
                ApplyToGroup(_lodGroups[g], _cachedLODs[g], _sanitizedBuffer);
        }

        // copy sanitized into pre-alloc last-applied buffer -- no Clone() alloc
        System.Array.Copy(_sanitizedBuffer, _lastAppliedBuffer, _sanitizedBuffer.Length);
    }

    void ApplyToGroup(LODGroup group, LOD[] lods, float[] thresholds)
    {
        if (lods.Length == 0) return;

        int count = Mathf.Min(lods.Length, thresholds.Length);
        for (int i = 0; i < count; i++)
            lods[i].screenRelativeTransitionHeight = thresholds[i];

        // remaining LOD levels beyond prediction length go near zero
        for (int i = count; i < lods.Length; i++)
            lods[i].screenRelativeTransitionHeight = 0.001f;

        // final monotonic guard -- silent skip, no LogWarning in hot path
        for (int i = 1; i < lods.Length; i++)
        {
            if (lods[i].screenRelativeTransitionHeight >= lods[i - 1].screenRelativeTransitionHeight)
                return;
        }

        group.SetLODs(lods);
        group.RecalculateBounds();
    }

    // writes result into pre-alloc output buffer -- no heap alloc
    void SanitizeThresholds(float[] raw, float[] output)
    {
        for (int i = 0; i < raw.Length; i++)
            output[i] = Mathf.Clamp01(raw[i]);

        // enforce strictly decreasing from index 0 downward
        for (int i = 1; i < output.Length; i++)
        {
            float maxAllowed = output[i - 1] - minThresholdGap;
            if (output[i] >= output[i - 1])
                output[i] = Mathf.Max(0f, maxAllowed);
        }

        // second pass to fix any negative values that broke order
        for (int i = output.Length - 1; i >= 1; i--)
        {
            if (output[i] < 0f) output[i] = 0f;
            if (output[i - 1] <= output[i])
                output[i - 1] = output[i] + minThresholdGap;
            output[i - 1] = Mathf.Clamp01(output[i - 1]);
        }
    }

    bool ArraysEqual(float[] a, float[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (Mathf.Abs(a[i] - b[i]) > 0.02f) return false;
        }
        return true;
    }
}
