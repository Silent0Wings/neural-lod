using UnityEngine;
using System.Collections.Generic;

[System.Serializable]
public class Node
{
    public string id;
    public Vector3 position;
    public Quaternion rotation;
    public Node(string id, Vector3 position, Quaternion rotation)
    {
        this.id = id;
        this.position = position;
        this.rotation = rotation;
    }
}

[System.Serializable]
public class Path
{
    public List<Node> nodes = new List<Node>();

    public int Count => nodes.Count;

    public void AddNode(Node node)
    {
        nodes.Add(node);
    }

    public Node GetNode(int index)
    {
        return nodes[index % nodes.Count];
    }

    public Node GetNextNode(int currentIndex)
    {
        return GetNode(currentIndex + 1);
    }
}

public class CameraPathAnimator : MonoBehaviour
{
    [Header("Path Settings")]
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 5.0f;
    public bool loop = true;
    public Transform path_transform;
    public bool completed = false;

    [Header("Debug")]
    public bool drawGizmos = true;

    [Header("Node Timeout")]
    public float nodeTimeout = 10.0f;
    private float _nodeTimer = 0f;

    private Path _path = new Path();
    private int _currentIndex = 0;
    private bool _isRunning = false;

    public bool IsPaused { get; set; } = false;

    public float PathProgress
    {
        get
        {
            if (_path.Count == 0) return 0f;
            Node target = _path.GetNode(_currentIndex);
            int prevIndex = (_currentIndex - 1 + _path.Count) % _path.Count;
            Node prev = _path.GetNode(prevIndex);
            float segmentLength = Vector3.Distance(prev.position, target.position);
            if (segmentLength < 0.001f) return _currentIndex;
            float distToTarget = Vector3.Distance(transform.position, target.position);
            float fraction = 1f - Mathf.Clamp01(distToTarget / segmentLength);
            return _currentIndex + fraction;
        }
    }

    public int CurrentIndex => _currentIndex;
    public int NodeCount => _path.Count;

    void Awake()
    {
        BuildPath();
    }

    void Start()
    {
        if (_path.Count == 0)
        {
            Debug.LogWarning("[CameraPathAnimator] No nodes found under 'Path' child.");
            return;
        }

        transform.position = _path.GetNode(0).position;
        transform.rotation = _path.GetNode(0).rotation;
        _currentIndex = 0;
        _isRunning = true;

        if (_path.Count > 1)
        {
            float dist = Vector3.Distance(_path.GetNode(0).position, _path.GetNode(1).position);
            nodeTimeout = (dist / moveSpeed) * 1.5f;
        }

        Debug.Log($"[CameraPathAnimator] Started. Total nodes: {_path.Count}");
    }

    void BuildPath()
    {
        Transform pathRoot;
        if (path_transform != null)
        {
            pathRoot = path_transform;
        }
        else
        {
            pathRoot = transform.Find("Paths");
        }

        if (pathRoot == null)
        {
            Debug.LogError("[CameraPathAnimator] No child named 'Path' found on this GameObject.");
            return;
        }

        for (int i = 0; i < pathRoot.childCount; i++)
        {
            Transform child = pathRoot.GetChild(i);
            _path.AddNode(new Node(child.name, child.position, child.rotation));
        }

        Debug.Log($"[CameraPathAnimator] Built path with {_path.Count} nodes from '{pathRoot.name}'.");
    }

    void Update()
    {
        if (!_isRunning || _path.Count == 0) return;
        if (IsPaused) return;

        _nodeTimer += Time.deltaTime;
        Node target = _path.GetNode(_currentIndex);

        bool arrived  = Vector3.Distance(transform.position, target.position) < 0.05f;
        bool timedOut = _nodeTimer >= nodeTimeout;

        if (arrived || timedOut)
        {
            if (timedOut)
                Debug.LogWarning($"[CameraPathAnimator] Node {target.id} timed out.");
            else
                Debug.Log($"[CameraPathAnimator] Reached node: {target.id}");

            AdvanceNode();
            _nodeTimer = 0f;
            return;
        }

        transform.position = Vector3.MoveTowards(
            transform.position,
            target.position,
            moveSpeed * Time.deltaTime
        );

        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            target.rotation,
            rotateSpeed * Time.deltaTime
        );
    }

    void AdvanceNode()
    {
        _currentIndex++;

        if (_currentIndex >= _path.Count)
        {
            if (loop) { _currentIndex = 0; }
            else { _isRunning = false; completed = true; return; }
        }

        Node next = _path.GetNode(_currentIndex);
        float dist = Vector3.Distance(transform.position, next.position);
        _nodeTimer = 0f;
        nodeTimeout = (dist / moveSpeed) * 1.5f;
    }

    void OnDrawGizmos()
    {
        if (!drawGizmos || _path == null || _path.Count == 0) return;

        for (int i = 0; i < _path.Count; i++)
        {
            Node current = _path.GetNode(i);
            Node next = _path.GetNextNode(i);

            Gizmos.color = Color.cyan;
            Gizmos.DrawSphere(current.position, 0.2f);
            Gizmos.DrawLine(current.position, next.position);

            Vector3 forward = current.rotation * Vector3.forward;
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(current.position, forward * 1.5f);

            Vector3 tip = current.position + forward * 1.5f;
            Vector3 right = current.rotation * Vector3.right;
            Gizmos.DrawLine(tip, tip - forward * 0.3f + right * 0.15f);
            Gizmos.DrawLine(tip, tip - forward * 0.3f - right * 0.15f);
        }
    }

    public void ResetPath()
    {
        if (_path.Count == 0)
        {
            Debug.LogWarning("[CameraPathAnimator] Cannot reset — path is empty.");
            return;
        }

        _currentIndex = 0;
        completed = false;
        _isRunning = true;

        transform.position = _path.GetNode(0).position;
        transform.rotation = _path.GetNode(0).rotation;
        _nodeTimer = 0f;

        if (_path.Count > 1)
        {
            float dist = Vector3.Distance(_path.GetNode(0).position, _path.GetNode(1).position);
            nodeTimeout = (dist / moveSpeed) * 1.5f;
        }

        Debug.Log("[CameraPathAnimator] Path reset to node 0.");
    }
}
