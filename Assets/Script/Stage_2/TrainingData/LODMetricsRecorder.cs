// File: LODMetricsRecorder.cs
// Captures performance metrics for all LOD levels at a given (gridPoint, rotation).
// Warmup runs ONCE per (point, rotation) after camera moves.
// A short mini-warmup runs between LOD level switches.
// Requires: Project Settings > Player > Enable Frame Timing Stats = ON.
//
// FIXES APPLIED:
//   FIX-1: Removed _rendererRecorder ("Visible Objects Count" invalid in Unity 6 URP).
//           CountVisibleRenderersForLod uses GeometryUtility.CalculateFrustumPlanes +
//           TestPlanesAABB + WorldToScreenPoint scoped to the CURRENTLY FORCED LOD
//           level renderers only. Previous attempt cached all renderers from all LOD
//           levels causing constant count of 14752 regardless of forced LOD.
//
//   FIX-2: ComputeScreenCoverage uses current forced LOD renderers, not always LOD0.

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.Profiling;

public class LODMetricsRecorder : MonoBehaviour
{
    [Header("Sampling Configuration")]
    public int warmUpFrames    = 15;
    public int lodSwitchWarmup = 3;
    public int sampleFrames    = 15;

    // references set by the orchestrator
    [HideInInspector] public LODGroup[] lodGroups;

    // results for all LOD levels after CaptureAllLods completes
    [HideInInspector] public List<SampleRecord> lodRecords = new List<SampleRecord>();

    FrameTiming[] timingBuffer = new FrameTiming[1];

    ProfilerRecorder _triangleRecorder;
    ProfilerRecorder _drawCallRecorder;

    // FIX-1: frustum planes array, reused every frame
    private Plane[] _frustumPlanes = new Plane[6];

    void OnEnable()
    {
        _triangleRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Triangles Count");
        _drawCallRecorder = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count");
    }

    void OnDisable()
    {
        _triangleRecorder.Dispose();
        _drawCallRecorder.Dispose();
    }

    // moves camera to (point, rotation) and waits warmUpFrames for GPU to stabilize
    // call this once before CaptureAllLods for each (point, rotation)
    public IEnumerator WarmUpPosition(GridPoint point, Vector3 eulerAngles, Camera cam)
    {
        cam.transform.position = point.coordinates;
        cam.transform.rotation = Quaternion.Euler(eulerAngles);

        foreach (LODGroup group in lodGroups)
            group.ForceLOD(-1);

        for (int i = 0; i < warmUpFrames; i++)
            yield return null;
    }

    // sweeps all LOD levels at the current camera position
    // WarmUpPosition must be called first for this (point, rotation)
    public IEnumerator CaptureAllLods(GridPoint point, Vector3 eulerAngles, int lodCount, Camera cam)
    {
        lodRecords.Clear();

        for (int lod = 0; lod < lodCount; lod++)
        {
            foreach (LODGroup group in lodGroups)
                group.ForceLOD(lod);

            // mini-warmup after LOD switch
            for (int i = 0; i < lodSwitchWarmup; i++)
                yield return null;

            float totalGpu      = 0f;
            float totalCpu      = 0f;
            int   validSamples  = 0;

            long totalTriangles    = 0;
            long totalRenderers    = 0;
            long totalDrawCalls    = 0;
            int  complexitySamples = 0;

            for (int i = 0; i < sampleFrames; i++)
            {
                yield return null;

                FrameTimingManager.CaptureFrameTimings();
                uint count = FrameTimingManager.GetLatestTimings(1, timingBuffer);

                if (count > 0)
                {
                    float gpu = (float)timingBuffer[0].gpuFrameTime;
                    float cpu = (float)timingBuffer[0].cpuMainThreadFrameTime;

                    if (gpu > 0.001f)
                    {
                        totalGpu += gpu;
                        totalCpu += cpu;
                        validSamples++;
                    }
                }

                totalTriangles += _triangleRecorder.Valid ? _triangleRecorder.LastValue : 0;
                totalDrawCalls += _drawCallRecorder.Valid ? _drawCallRecorder.LastValue : 0;

                // FIX-1: frustum check scoped to current LOD level renderers only
                totalRenderers += CountVisibleRenderersForLod(cam, lod);
                complexitySamples++;
            }

            foreach (LODGroup group in lodGroups)
                group.ForceLOD(-1);

            SampleRecord record = new SampleRecord();
            record.pointId        = point.pointId;
            record.rotationAngles = eulerAngles;
            record.lodLevel       = lod;

            if (validSamples > 0)
            {
                record.meanGpuTimeMs = totalGpu / validSamples;
                record.meanCpuTimeMs = totalCpu / validSamples;
                float avgFrameTimeMs = Mathf.Max(record.meanGpuTimeMs, record.meanCpuTimeMs);
                record.meanFps = avgFrameTimeMs > 0.001f ? 1000f / avgFrameTimeMs : 0f;
            }
            else
            {
                record.meanGpuTimeMs = -1f;
                record.meanCpuTimeMs = -1f;
                record.meanFps       = -1f;
                Debug.LogWarning($"[MetricsRecorder] No valid samples: point {point.pointId} " +
                                 $"rot {eulerAngles} LOD {lod}");
            }

            if (complexitySamples > 0)
            {
                record.triangleCount        = (int)(totalTriangles / complexitySamples);
                record.visibleRendererCount = (int)(totalRenderers  / complexitySamples);
                record.drawCallCount        = (int)(totalDrawCalls  / complexitySamples);
            }

            // FIX-2: pass current lod so coverage uses active LOD renderers
            record.screenCoverage = ComputeScreenCoverage(cam, lod);
            lodRecords.Add(record);
        }
    }

    // FIX-1: frustum-based visible renderer count scoped to the currently forced LOD.
    // For each LODGroup, retrieves renderers at lods[currentLod] only, then applies:
    //   1. GeometryUtility.CalculateFrustumPlanes  — build 6 frustum planes from camera
    //   2. GeometryUtility.TestPlanesAABB          — AABB intersection test per renderer
    //   3. cam.WorldToScreenPoint depth check      — reject behind-camera renderers
    // This matches the proven stage 1 MetricLogger method exactly.
    private int CountVisibleRenderersForLod(Camera cam, int currentLod)
{
    if (lodGroups == null || cam == null) return 0;

    GeometryUtility.CalculateFrustumPlanes(cam, _frustumPlanes);
    int count = 0;

    foreach (LODGroup group in lodGroups)
    {
        if (group == null) continue;

        LOD[] lods = group.GetLODs();
        if (lods.Length == 0) continue;

        int safeIdx = Mathf.Clamp(currentLod, 0, lods.Length - 1);
        Renderer[] renderers = lods[safeIdx].renderers;
        if (renderers == null) continue;

        foreach (Renderer r in renderers)
        {
            if (r == null || !r.enabled) continue;
            if (r is ParticleSystemRenderer) continue;
            if (r is LineRenderer) continue;
            if (r is TrailRenderer) continue;
            if (!GeometryUtility.TestPlanesAABB(_frustumPlanes, r.bounds)) continue;

            Vector3 screenPoint = cam.WorldToScreenPoint(r.bounds.center);
            if (screenPoint.z < 0f) continue;

            count++;
        }
    }

    return count;
}


    // FIX-2: screen coverage computed from active LOD renderers, not always LOD0.
    float ComputeScreenCoverage(Camera cam, int currentLod)
    {
        if (lodGroups == null || cam == null) return 0f;

        float screenW = cam.pixelWidth;
        float screenH = cam.pixelHeight;
        if (screenW <= 0f || screenH <= 0f) return 0f;

        int count      = 0;
        float totalArea = 0f;

        GeometryUtility.CalculateFrustumPlanes(cam, _frustumPlanes);

        foreach (LODGroup group in lodGroups)
        {
            if (group == null) continue;

            LOD[] lods = group.GetLODs();
            if (lods.Length == 0) continue;

            int safeIdx = Mathf.Clamp(currentLod, 0, lods.Length - 1);
            Renderer[] renderers = lods[safeIdx].renderers;
            if (renderers == null || renderers.Length == 0) continue;

            foreach (Renderer r in renderers)
            {
                if (r == null || !r.enabled) continue;
                if (r is ParticleSystemRenderer) continue;
                if (r is LineRenderer) continue;
                if (r is TrailRenderer) continue;
                if (!GeometryUtility.TestPlanesAABB(_frustumPlanes, r.bounds)) continue;

                Bounds b = r.bounds;
                Vector3 screenPoint = cam.WorldToScreenPoint(b.center);
                if (screenPoint.z < 0f) continue;

                Vector3 min = cam.WorldToScreenPoint(b.min);
                Vector3 max = cam.WorldToScreenPoint(b.max);

                float w = Mathf.Abs(max.x - min.x) / screenW;
                float h = Mathf.Abs(max.y - min.y) / screenH;

                totalArea += Mathf.Clamp01(w) * Mathf.Clamp01(h);
                count++;
            }
        }

        // average coverage per visible renderer — matches stage 1 MetricLogger exactly
        return count > 0 ? totalArea / count : 0f;
    }
}