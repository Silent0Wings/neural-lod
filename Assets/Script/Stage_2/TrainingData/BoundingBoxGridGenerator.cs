// File: BoundingBoxGridGenerator.cs
// Computes a global AABB over all LODGroup objects (or a manual override)
// and subdivides it into a 3D grid of GridPoint instances.
// Layer 0 uses raycast to snap to ground, upper layers stack above it.
// Bounding box is recomputed after snapping to match actual point positions.
using UnityEngine;
using System.Collections.Generic;

public class BoundingBoxGridGenerator : MonoBehaviour
{
    [Header("Grid Resolution")]
    public int resolutionX = 20;
    public int resolutionY = 20;
    public int resolutionZ = 20;

    [Header("Optional Manual Bounds (leave zero to auto-compute)")]
    public Bounds manualBounds;

    [Header("Height Settings")]
    [Tooltip("Height above detected ground surface for layer 0.")]
    public float heightAboveGround = 1.5f;

    [Tooltip("Max raycast distance when searching for ground.")]
    public float raycastRange = 200f;

    [Header("Gizmos")]
    public bool drawGizmos = true;
    public float gizmoSphereSize = 0.2f;
    public float gizmoBoundsSize = 0.5f;

    [HideInInspector] public Bounds computedBounds;
    [HideInInspector] public List<GridPoint> gridPoints = new List<GridPoint>();

    private Dictionary<(int, int), float> groundHeights = new Dictionary<(int, int), float>();

    public void GenerateGrid(LODGroup[] lodGroups)
    {
        gridPoints.Clear();
        groundHeights.Clear();

        // initial bounds for grid cell calculation
        Bounds initialBounds;
        if (manualBounds.size.sqrMagnitude > 0.001f)
        {
            initialBounds = manualBounds;
        }
        else
        {
            initialBounds = ComputeAggregateBounds(lodGroups);
        }

        Vector3 min = initialBounds.min;
        Vector3 size = initialBounds.size;

        float cellX = size.x / resolutionX;
        float cellY = size.y / Mathf.Max(resolutionY, 1);
        float cellZ = size.z / resolutionZ;

        float layerSpacing = size.y / Mathf.Max(resolutionY, 1);

        int id = 0;
        int snappedCount = 0;

        // loop order: ix -> iz -> iy so layer 0 is always processed first per column
        for (int ix = 0; ix < resolutionX; ix++)
        {
            for (int iz = 0; iz < resolutionZ; iz++)
            {
                for (int iy = 0; iy < resolutionY; iy++)
                {
                    Vector3 cellMin = new Vector3(
                        min.x + ix * cellX,
                        min.y + iy * cellY,
                        min.z + iz * cellZ
                    );
                    Vector3 cellMax = new Vector3(
                        cellMin.x + cellX,
                        cellMin.y + cellY,
                        cellMin.z + cellZ
                    );

                    Vector3 center = (cellMin + cellMax) * 0.5f;

                    if (iy == 0)
                    {
                        // layer 0: raycast down to find ground
                        Vector3 rayOrigin = center + Vector3.up * raycastRange * 0.5f;
                        if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hit, raycastRange))
                        {
                            center.y = hit.point.y + heightAboveGround;
                            snappedCount++;
                        }
                        groundHeights[(ix, iz)] = center.y;
                    }
                    else
                    {
                        // upper layers: stack above ground height
                        center.y = groundHeights[(ix, iz)] + iy * layerSpacing;
                    }

                    GridPoint gp = new GridPoint();
                    gp.pointId = id;
                    gp.coordinates = center;
                    gp.volumeBoundsMin = cellMin;
                    gp.volumeBoundsMax = cellMax;

                    gridPoints.Add(gp);
                    id++;
                }
            }
        }

        // recompute bounds to match actual grid point positions after snapping
        if (gridPoints.Count > 0)
        {
            Bounds actualBounds = new Bounds(gridPoints[0].coordinates, Vector3.zero);
            foreach (var gp in gridPoints)
            {
                actualBounds.Encapsulate(gp.coordinates);
            }
            actualBounds.Expand(1f);
            computedBounds = actualBounds;
        }
        else
        {
            computedBounds = initialBounds;
        }

        Debug.Log($"[GridGenerator] Generated {gridPoints.Count} grid points " +
                  $"({resolutionX}x{resolutionY}x{resolutionZ}) " +
                  $"in bounds {computedBounds}. " +
                  $"Raycast snapped={snappedCount}/{resolutionX * resolutionZ}. " +
                  $"Layer spacing={layerSpacing:F2}m.");
    }

    Bounds ComputeAggregateBounds(LODGroup[] groups)
    {
        if (groups == null || groups.Length == 0)
        {
            Debug.LogWarning("[GridGenerator] No LODGroups provided, using default bounds.");
            return new Bounds(Vector3.zero, Vector3.one * 100f);
        }

        Bounds b = new Bounds(groups[0].transform.position, Vector3.zero);

        foreach (LODGroup group in groups)
        {
            LOD[] lods = group.GetLODs();
            foreach (LOD lod in lods)
            {
                if (lod.renderers == null) continue;
                foreach (Renderer r in lod.renderers)
                {
                    if (r == null) continue;
                    b.Encapsulate(r.bounds);
                }
            }
        }

        b.Expand(1f);
        return b;
    }

    private void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        if (gridPoints == null) return;

        // draw grid points
        Gizmos.color = Color.yellow;
        foreach (var gp in gridPoints)
        {
            Gizmos.DrawSphere(gp.coordinates, gizmoSphereSize);
        }

        // draw bounding box corners
        Gizmos.color = Color.yellow;
        Vector3 min = computedBounds.min;
        Vector3 max = computedBounds.max;

        Gizmos.DrawSphere(new Vector3(min.x, min.y, min.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(min.x, min.y, max.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(min.x, max.y, min.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(min.x, max.y, max.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(max.x, min.y, min.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(max.x, min.y, max.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(max.x, max.y, min.z), gizmoBoundsSize);
        Gizmos.DrawSphere(new Vector3(max.x, max.y, max.z), gizmoBoundsSize);

        // draw bounding box wireframe
        Gizmos.color = new Color(1f, 1f, 0f, 0.3f);
        Gizmos.DrawWireCube(computedBounds.center, computedBounds.size);
    }
}