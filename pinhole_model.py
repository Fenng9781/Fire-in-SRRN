import torch

# Constants for Planck's Law
h = torch.tensor(6.626e-34)  # Planck's constant in J·s
c = torch.tensor(3e8)  # Speed of light in m/s
kB = torch.tensor(1.381e-23)  # Boltzmann's constant in J/K

# 1. Intrinsic matrix
def intrinsic_matrix(fx=50, fy=50, cx=50, cy=50, s=0):
    return torch.tensor([
        [fx, s, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)

# 2. Extrinsic matrix
def extrinsic_matrix(camera_pos, orientation_point, up_vector=torch.tensor([0, 0, 1], dtype=torch.float32)):

    forward = orientation_point - camera_pos
    forward = forward / torch.norm(forward)

    side = torch.cross(up_vector, forward)
    side = side / torch.norm(side)

    up = torch.cross(forward, side)
    R = torch.stack([side, up, forward])

    T = -torch.mv(R, camera_pos)
    extrinsic = torch.column_stack([R, T])
    return extrinsic


# 4. World to Camera
def W2C(extrinsic, world_coords):
    R = extrinsic[:, :3]
    T = extrinsic[:, 3]
    camera_coords = torch.mv(R.T, (world_coords - T))
    return camera_coords


# 5. Camera to World
def C2W(extrinsic, camera_coords):
    ones = torch.ones((camera_coords.shape[0], 1), device=camera_coords.device)
    camera_coords = torch.cat([camera_coords, ones], dim=-1)  # Convert to homogeneous

    extrinsic = torch.vstack([extrinsic, torch.tensor([0.0, 0.0, 0.0, 1.0])])
    inverse_extrinsic = torch.linalg.inv(extrinsic)
    inverse_extrinsic_device = inverse_extrinsic.to(camera_coords.device)
    world_coords = torch.mm(inverse_extrinsic_device, camera_coords.T).T
    return world_coords[:, :3]

# 6. Camera to Pixel
def C2P(intrinsic, camera_coords):
    camera_coords = torch.cat([camera_coords, torch.tensor([1.0])])  # Convert to homogeneous
    pixel_coords = torch.mv(intrinsic, camera_coords)
    return (pixel_coords[:-1] / pixel_coords[-1]).int()

# 7. Pixel to Camera
def P2C(intrinsic, pixel_coords):
    if len(pixel_coords.shape) == 1:
        pixel_coords = pixel_coords.unsqueeze(0)
    homogeneous_pixel = torch.cat([pixel_coords, torch.ones(pixel_coords.shape[0], 1, device=pixel_coords.device)], dim=-1)
    intrinsic_device = intrinsic.to(pixel_coords.device)
    camera_coords = torch.mm(torch.linalg.inv(intrinsic_device), homogeneous_pixel.T).T
    return camera_coords

# 8. TCS to WCS
def tcs_to_wcs(points_tcs, scale_factor=1.0, translation=torch.tensor([20.0, 20.0, 20.0])):
    return points_tcs * scale_factor + translation

# 9. WCS to TCS
def W2T(sample_point, translation=torch.tensor([20.0, 20.0, 20.0])):
    return sample_point - translation.to(sample_point.device)



