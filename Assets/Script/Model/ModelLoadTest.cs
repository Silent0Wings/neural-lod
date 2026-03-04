using UnityEngine;
using Unity.InferenceEngine;

public class ModelLoadTest : MonoBehaviour
{
    public ModelAsset onnxAsset;

    void Start()
    {
        var model = ModelLoader.Load(onnxAsset);
        var worker = new Worker(model, BackendType.CPU);

        Debug.Log($"[ModelLoadTest] Model loaded OK");
        Debug.Log($"[ModelLoadTest] Input  name: {model.inputs[0].name}");
        Debug.Log($"[ModelLoadTest] Output name: {model.outputs[0].name}");

        worker.Dispose();
    }
}