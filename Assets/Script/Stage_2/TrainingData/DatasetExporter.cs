// File: DatasetExporter.cs
// Writes ProfilingSession data to disk as CSV files.
// Outputs: lod_objects.csv, grid_points.csv, sample_records.csv, labels.csv
// Default output folder: Application.persistentDataPath/BakerExport/
using UnityEngine;
using System.IO;
using System.Text;

public class DatasetExporter : MonoBehaviour
{
    [Header("Export Settings")]
    public string subfolder = "BakerExport";

    string outputDir;

    /// exports the full session to CSV files
    public void Export(ProfilingSession session)
    {
        outputDir = Path.Combine(Application.persistentDataPath, subfolder);
        Directory.CreateDirectory(outputDir);

        ExportLodObjects(session);
        ExportGridPoints(session);
        ExportSampleRecords(session);
        ExportLabels(session);

        Debug.Log($"[DatasetExporter] Exported to: {outputDir}");
    }

    void ExportLodObjects(ProfilingSession session)
    {
        string path = Path.Combine(outputDir, "lod_objects.csv");
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("reference_id,renderer_count,thresholds");

        foreach (LODObjectInfo obj in session.objects)
        {
            string threshStr = string.Join(";", obj.thresholds);
            sb.AppendLine($"{obj.referenceId},{obj.rendererCount},{threshStr}");
        }

        File.WriteAllText(path, sb.ToString());
    }

    void ExportGridPoints(ProfilingSession session)
    {
        string path = Path.Combine(outputDir, "grid_points.csv");
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("point_id,x,y,z,vol_min_x,vol_min_y,vol_min_z,vol_max_x,vol_max_y,vol_max_z");

        foreach (GridPoint gp in session.grid)
        {
            sb.AppendLine($"{gp.pointId}," +
                          $"{gp.coordinates.x},{gp.coordinates.y},{gp.coordinates.z}," +
                          $"{gp.volumeBoundsMin.x},{gp.volumeBoundsMin.y},{gp.volumeBoundsMin.z}," +
                          $"{gp.volumeBoundsMax.x},{gp.volumeBoundsMax.y},{gp.volumeBoundsMax.z}");
        }

        File.WriteAllText(path, sb.ToString());
    }

    void ExportSampleRecords(ProfilingSession session)
    {
        string path = Path.Combine(outputDir, "sample_records.csv");

        // stream to avoid large string allocations
        using (StreamWriter writer = new StreamWriter(path, false))
        {
            writer.WriteLine("point_id,rot_x,rot_y,rot_z,lod_level,mean_gpu_ms,mean_cpu_ms,mean_fps");

            foreach (SampleRecord r in session.samples)
            {
                writer.WriteLine($"{r.pointId}," +
                                 $"{r.rotationAngles.x},{r.rotationAngles.y},{r.rotationAngles.z}," +
                                 $"{r.lodLevel}," +
                                 $"{r.meanGpuTimeMs},{r.meanCpuTimeMs},{r.meanFps}");
            }
        }
    }

    void ExportLabels(ProfilingSession session)
    {
        if (session.labels == null || session.labels.Count == 0) return;

        string path = Path.Combine(outputDir, "labels.csv");
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("point_id,rot_x,rot_y,rot_z,optimal_thresholds");

        foreach (LabelledSample label in session.labels)
        {
            string threshStr = string.Join(";", label.optimalThresholds);
            sb.AppendLine($"{label.pointId}," +
                          $"{label.rotationAngles.x},{label.rotationAngles.y},{label.rotationAngles.z}," +
                          $"{threshStr}");
        }

        File.WriteAllText(path, sb.ToString());
    }
}
