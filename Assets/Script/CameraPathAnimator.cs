using UnityEngine;
using System.Collections.Generic;


// Node
// Stores a single waypoint: position, rotation, ID

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


// Path
// Owns and controls an ordered list of Nodes

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
        return nodes[index % nodes.Count]; // wraps for looping
    }

    public Node GetNextNode(int currentIndex)
    {
        return GetNode(currentIndex + 1);
    }
}


// CameraPathAnimator
// Moves the camera along the Path deterministically
// Attach to the Camera GameObject

public class CameraPathAnimator : MonoBehaviour
{
    [Header("Path Settings")]
    public float moveSpeed = 5.0f;
    public float rotateSpeed = 5.0f;
    public bool loop = true;
    public Transform path_transform;
    public bool completed=false;
    [Header("Debug")]
    public bool drawGizmos = true;

    // Internal path instance
    private Path _path = new Path();
    private int _currentIndex = 0;
    private bool _isRunning = false;

    
    // Setup — read waypoints from child transforms
    // Hierarchy: this → Path → [node_00, node_01, ...]


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

        // Snap camera to first node deterministically
        transform.position = _path.GetNode(0).position;
        transform.rotation = _path.GetNode(0).rotation;
        _currentIndex = 0;
        _isRunning = true;

        Debug.Log($"[CameraPathAnimator] Started. Total nodes: {_path.Count}");
    }

    
    // Build path from children of "Path" child
    // Order = sibling index (top to bottom in Hierarchy)
    
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

    
    // Update — move toward current target node
    
    void Update()
    {
        if (!_isRunning || _path.Count == 0) return;

        Node target = _path.GetNode(_currentIndex);

        // Move
        transform.position = Vector3.MoveTowards(
            transform.position,
            target.position,
            moveSpeed * Time.deltaTime
        );

        // Rotate
        transform.rotation = Quaternion.Slerp(
            transform.rotation,
            target.rotation,
            rotateSpeed * Time.deltaTime
        );

        // Reached target node?
        if (Vector3.Distance(transform.position, target.position) < 0.05f)
        {
            Debug.Log($"[CameraPathAnimator] Reached node: {target.id}");
            AdvanceNode();
        }
    }

    
    // Advance to next node or stop/loop
    
    void AdvanceNode()
    {
        _currentIndex++;

        if (_currentIndex >= _path.Count)
        {
            if (loop)
            {
                _currentIndex = 0;
                Debug.Log("[CameraPathAnimator] Looping path.");
            }
            else
            {
                _isRunning = false;
                completed = true;  
                Debug.Log("[CameraPathAnimator] Path complete.");
            }
        }
    }

    
    // Gizmos — visualize path in Scene view
    
    void OnDrawGizmos()
    {
        if (!drawGizmos || _path == null || _path.Count == 0) return;

        for (int i = 0; i < _path.Count; i++)
        {
            Node current = _path.GetNode(i);
            Node next = _path.GetNextNode(i);

            // Path line + node sphere
            Gizmos.color = Color.cyan;
            Gizmos.DrawSphere(current.position, 0.2f);
            Gizmos.DrawLine(current.position, next.position);

            // Forward vector arrow
            Vector3 forward = current.rotation * Vector3.forward;
            Gizmos.color = Color.blue;
            Gizmos.DrawRay(current.position, forward * 1.5f);

            // Arrowhead
            Vector3 tip = current.position + forward * 1.5f;
            Vector3 right = current.rotation * Vector3.right;
            Gizmos.DrawLine(tip, tip - forward * 0.3f + right * 0.15f);
            Gizmos.DrawLine(tip, tip - forward * 0.3f - right * 0.15f);
        }
    }

	// Reset — restart path from node 0

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

		Debug.Log("[CameraPathAnimator] Path reset to node 0.");
	}
}
