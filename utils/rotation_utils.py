import torch

def generate_random_labels(b, k, N):
    """
    Generate random labels for each batch.

    Args:
    - b: Batch size
    - k: Number of elements in each batch
    - N: Maximum value for each element

    Returns:
    - random_labels: Tensor of shape (b, k) containing random labels
    """
    random_labels = torch.cat([torch.randperm(N)[:k].unsqueeze(0) for _ in range(b)], dim=0)
    return random_labels

def compute_class_means(features, labels, k):
    """
    Compute the mean features for each class based on labels.

    Args:
    - features: Input tensor of shape (B, N, C)
    - labels: Tensor of shape (B, N) containing class labels
    - k: Number of classes

    Returns:
    - class_means: Tensor of shape (B, k, C) containing the mean features for each class
    """
    device = features.device
    B, N, C = features.size()

    # Initialize tensors for sum and count
    class_sums = torch.zeros((B, k, C), device=device)
    class_counts = torch.zeros((B, k), device=device)

    # Accumulate sum and count for each class
    class_sums.scatter_add_(1, labels.unsqueeze(-1).expand(B, N, C), features)
    class_counts.scatter_add_(1, labels, torch.ones((B, N), device=device))

    # Avoid division by zero by replacing count=0 with count=1
    class_counts = torch.where(class_counts == 0, torch.tensor(1, device=device), class_counts)

    # Compute mean features for each class
    class_means = class_sums / class_counts.unsqueeze(-1)

    return class_means

def gather_points(points, indices):
    """
    Gather points from a point cloud using indices.

    Parameters:
        points (torch.Tensor): Input point cloud tensor of shape (batch_size, num_points, c).
        indices (torch.Tensor): Index tensor of shape (batch_size, num_samples).

    Returns:
        gathered_points (torch.Tensor): Gathered points tensor of shape (batch_size, num_samples, c).
    """
    batch_size, num_samples = indices.size()
    _, num_points, c = points.size()

    # Reshape indices to be used with torch.gather
    indices = indices.view(batch_size, num_samples, 1).expand(batch_size, num_samples, c)

    # Gather points using indices
    gathered_points = torch.gather(points, 1, indices)

    return gathered_points

def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) for point clouds.

    Parameters:
        points (torch.Tensor): Input point cloud tensor of shape (batch_size, num_points, 3).
        num_samples (int): Number of points to be sampled.

    Returns:
        sampled_points (torch.Tensor): Sampled points tensor of shape (batch_size, num_samples, 3).
        sampled_indices (torch.Tensor): Indices of the sampled points in the original point cloud.
    """
    batch_size, num_points, _ = points.size()

    # Initialize sampled points and distances
    sampled_points = torch.zeros(batch_size, num_samples, 3, device=points.device)
    sampled_indices = torch.zeros(batch_size, num_samples, dtype=torch.long, device=points.device)
    distances = torch.ones(batch_size, num_points, device=points.device) * 1e10

    # Randomly select the first point in each batch
    rand_indices = torch.randint(0, num_points, (batch_size,), device=points.device)
    sampled_points[:, 0, :] = points[torch.arange(batch_size), rand_indices, :]
    sampled_indices[:, 0] = rand_indices
    
    for i in range(1, num_samples):
        # Compute distances from the last sampled point to all other points
        dist_to_last_point = torch.norm(points - sampled_points[:, i - 1, :].view(batch_size, 1, 3), dim=2)
        
        # Update distances if a shorter distance is found
        distances = torch.min(distances, dist_to_last_point)
        
        # Find the index of the farthest point
        farthest_point_indices = torch.argmax(distances, dim=1)
        
        # Update sampled points and indices
        sampled_points[:, i, :] = points[torch.arange(batch_size), farthest_point_indices, :]
        sampled_indices[:, i] = farthest_point_indices

    return sampled_points, sampled_indices


def normalize_vector(v, dim=1, return_mag =False):
    v_mag = torch.sqrt(v.pow(2).sum(dim=dim, keepdim=True))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.expand_as(v)
    v = v/v_mag
    return v

def cross_product(u, v):
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
    out = torch.cat((i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)),1)#batch*3
    return out

def Ortho6d2Mat(x_raw, y_raw):
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)#batch*3
    x = cross_product(y,z)#batch*3

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

