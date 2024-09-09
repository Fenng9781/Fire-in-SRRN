import matplotlib.pyplot as plt
import math
import sys
import os
from datetime import datetime
import numpy as np
from pinhole_model import *

#Generate camera positions
def generate_camera(start_pos, field_center_pos, start=0, end=180, interval=15):
    radius = torch.sqrt(torch.sum((start_pos - field_center_pos) ** 2))
    camera_pos = []
    initial_angle = torch.atan2(start_pos[1] - field_center_pos[1], start_pos[0] - field_center_pos[0])
    initial_angle = torch.rad2deg(initial_angle)
    for angle in range(start, end, interval):
        relative_angle = initial_angle + angle
        x = field_center_pos[0] + radius * torch.cos(torch.deg2rad(relative_angle))
        y = field_center_pos[1] + radius * torch.sin(torch.deg2rad(relative_angle))
        z = start_pos[2]
        camera_pos.append(torch.tensor([x, y, z], dtype=torch.float32))
    return camera_pos


# 10. 3D Temperature Field
def get_3D_TF(x, y, z, max_temperature=2000, wavelength=694 * 1e-9):

    # tripl peak
    # K1 = ((x - 3) ** 2 + (y) ** 2 + (z-3) ** 2) / (6 ** 2)
    # K2 = ((x + 6) ** 2 + (y - 4) ** 2 + (z+3) ** 2) / (6 ** 2)
    # K3 = ((x + 6) ** 2 + (y + 4) ** 2 + z ** 2) / (6 ** 2)
    # K1 = torch.where(K1 <= 1, torch.exp(-K1 / 30), torch.tensor(0.0))
    # K2 = torch.where(K2 <= 1, torch.exp(-K2 / 30), torch.tensor(0.0))
    # K3 = torch.where(K3 <= 1, torch.exp(-K3 / 30), torch.tensor(0.0))
    # K_temp = torch.where(K1 >= K2, K1, K2)
    # K = torch.where(K_temp >= K3, K_temp, K3)

    # double peak
    # K1 = ((x - 4) ** 2 + (y + 4) ** 2 + (z) ** 2) / (9 ** 2)
    # K2 = ((x + 4) ** 2 + (y - 4) ** 2 + (z) ** 2) / (9 ** 2)
    # K1 = torch.where(K1 <= 1, torch.exp(-K1/60), torch.tensor(0.0))
    # K2 = torch.where(K2 <= 1, torch.exp(-K2/60), torch.tensor(0.0))
    # K = torch.where(K1 >= K2, K1, K2)

    # single peak
    K = (x**2 + y**2 + z**2) / (2 ** 2)

    K = torch.where(K <= 1, torch.exp(-K/60), torch.tensor(0.0,device=x.device))
    T = max_temperature * K
    # planck's blackbody radiation law
    # numerator = 2 * h * c ** 2
    # denominator = (wavelength ** 5) * (torch.exp(h * c / (wavelength * kB * T)) - 1)
    # emis = numerator / denominator
    return T

# def get_3D_emissivity(x, y, z, max_Emissivity=0.5):
#
#     # tripl peak
#     K1 = ((x-3)**2 + (y)**2 + (z-3)**2) / (6 ** 2)
#     K2 = ((x+6)**2 + (y-4)**2 + (z+3)**2) / (6 ** 2)
#     K3 = ((x+6) ** 2 + (y+4) ** 2 + z ** 2) / (6 ** 2)
#     K1 = torch.where(K1 <= 1, torch.exp(-K1/60), torch.tensor(0.0))
#     K2 = torch.where(K2 <= 1, torch.exp(-K2/60), torch.tensor(0.0))
#     K3 = torch.where(K3 <= 1, torch.exp(-K3/60), torch.tensor(0.0))
#     K_temp = torch.where(K1 >= K2, K1, K2)
#     K=torch.where(K_temp >= K3, K_temp, K3)
#
#     # double peak
#     # K1 = ((x - 4) ** 2 + (y + 4) ** 2 + (z) ** 2) / (9 ** 2)
#     # K2 = ((x + 4) ** 2 + (y - 4) ** 2 + (z) ** 2) / (9 ** 2)
#     # K1 = torch.where(K1 <= 1, torch.exp(-K1/60), torch.tensor(0.0))
#     # K2 = torch.where(K2 <= 1, torch.exp(-K2/60), torch.tensor(0.0))
#     # K = torch.where(K1 >= K2, K1, K2)
#
#     # single peak
#     # K = (x**2 + y**2 + z**2) / (7 ** 2)
#     # K = torch.where(K <= 1, torch.exp(-K/60), torch.tensor(0.0))
#
#     emissivity = max_Emissivity * K
#     return emissivity



def generate_rays(width, height, intrinsic, extrinsic, camera_pose):
    # Define the device (GPU if CUDA is available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create a meshgrid for pixel coordinates
    ii, jj = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
    pixel_h = torch.stack((ii.flatten(), jj.flatten()), dim=-1).to(dtype=torch.float32, device=device)
    # Back-project to camera coordinates
    cam_coords = P2C(intrinsic, pixel_h)
    # Convert camera coordinates to world coordinates
    world_coords = C2W(extrinsic, cam_coords)
    # Compute ray directions for all pixels
    camera_pose_device = camera_pose.to(world_coords.device)
    ray_directions = world_coords[:, :3] - camera_pose_device
    ray_directions /= torch.norm(ray_directions, dim=-1, keepdim=True)
    # Replicate camera pose for all rays
    camera_poses = camera_pose.repeat(ray_directions.shape[0], 1)
    return list(zip(camera_poses, ray_directions))

def ray_casting(rays, width, height, sampling_distance=0.5, max_distance=100.0, batch_size=180):
    # Define the device (GPU if CUDA is available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_rays = len(rays)
    image = torch.zeros(num_rays).to(device)
    image_dis = torch.zeros(num_rays).to(device)
    for batch_start in range(0, num_rays, batch_size):
        # Determine the size of the current batch
        current_batch_size = min(batch_size, num_rays - batch_start)

        # Extract the batch of rays and move to GPU
        batched_rays = rays[batch_start:batch_start + current_batch_size]

        # Unpack ray origins and directions from the batched rays and move to GPU
        ray_origins = torch.stack([ray[0] for ray in batched_rays]).to(device)
        ray_directions = torch.stack([ray[1] for ray in batched_rays]).to(device)

        # Create a tensor of sampling distances for the current batch of rays and move to GPU
        ts = torch.arange(5, max_distance, sampling_distance).unsqueeze(0).to(device)
        ts = ts.repeat(current_batch_size, 1)

        # Compute sample points for all rays and sampling distances
        sample_points = ray_origins.unsqueeze(1) + ts.unsqueeze(-1) * ray_directions.unsqueeze(1)

        # Reshape sample_points for processing
        sample_points = sample_points.view(-1, 3)

        # Compute the temperature field for all sample points in the current batch
        translated_points = W2T(sample_points)
        x, y, z = translated_points.unbind(-1)

        # 考虑发射率
        temper_values = get_3D_TF(x, y, z)
        # emiss_values = get_3D_emissivity(x, y, z)
        emiss_values = 1
        values = temper_values * emiss_values

        # Reshape the values tensor to shape (current_batch_size, num_samples_per_ray) and move to GPU
        num_samples_per_ray = ts.size(1)
        values = values.reshape(current_batch_size, num_samples_per_ray).to(device)

        # Compute the average value for each ray in the current batch and flatten
        batched_image = torch.trapz(values, dx=sampling_distance, dim=-1).flatten()

        # Update the main image tensor with the results of the current batch
        image[batch_start:batch_start + current_batch_size] = batched_image

    # Reshape the flattened image to its original 2D shape
    image = image.reshape(height, width)
    return image

def save_data(projection, camera_position, flame_angle,intrinsic, extrinsic,current_time):
    # Create a directory named by the current time
    if not os.path.exists('./dataset/'+current_time):
        os.makedirs('./dataset/'+current_time)

    projection_filename = os.path.join('./dataset/'+current_time, f"flame_{flame_angle}.csv")
    # Save projection as .csv
    #projection_filename = os.path.join(current_time, "projection.csv")
    np.savetxt(projection_filename, projection.cpu().numpy(), delimiter=",")

    # Save camera details in a .txt file
    details_filename = os.path.join('./dataset/'+current_time, "camera_details.txt")
    with open(details_filename, 'a') as file:
        file.write("\nCamera Position:\n")
        file.write(str(camera_position.tolist()) + "\n")
        file.write("Intrinsic Parameters:\n")
        file.write(str(intrinsic.tolist()) + "\n")
        file.write("Extrinsic Parameters:\n")
        file.write(str(extrinsic.tolist()) + "\n")
        file.write("Projection Filename:\n")
        file.write(projection_filename + "\n")
    print(f"Saved data in {current_time}/")

# 11. Uniform Sampling in Sphere
def show_uniform_sampl(num_samples):
    radii = 20 * (torch.rand(num_samples) ** (1.0 / 3.0))  # Using power function to compute cube root
    theta = torch.rand(num_samples) * 2 * math.pi
    phi = torch.acos(2 * torch.rand(num_samples) - 1)
    x = radii * torch.sin(phi) * torch.cos(theta)
    y = radii * torch.sin(phi) * torch.sin(theta)
    z = radii * torch.cos(phi)
    return x, y, z

def visualize_ray(ax, origin, direction, length=50.0):
    """
    Draw a ray in the 3D plot.
    Parameters:
        ax : The 3D axis where the ray should be drawn.
        origin : The starting point of the ray.
        direction : The direction of the ray (should be normalized).
        length : Length for the ray visualization.
    """
    # Compute the endpoint of the ray
    direction_device = direction.to(origin.device)
    end = origin + direction_device * length
    ax.scatter(origin[0], origin[1], origin[2], c='red', marker='o',  s=100, label='Generated Camera Positions')
    # Plot the ray as a line
    ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color='orange')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # camera location
    start=0
    end=360
    interval=120
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    orientation_point = torch.tensor([20, 20, 20])
    camera_positions = generate_camera(torch.tensor([0, 0, 20]), orientation_point, start, end, interval)

    ###################################
    """
    visualize of camera and temperature field
    """
    # Convert spherical coordinates to cartesian coordinates
    x, y, z = show_uniform_sampl(5000)
    # Calculate K for each sampled point
    temperature_field = get_3D_TF(x,y,z)
    tcs = torch.stack([x, y, z], dim=1)
    wcs = tcs_to_wcs(tcs)
    x,y,z =torch.t(wcs)
    # Visualize the 3D field
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, c=temperature_field, marker='o', s=0.4, alpha=0.5, cmap='viridis')
    wcs_o = tcs_to_wcs(torch.tensor([0.,0.,0.]))
    # Position the camera in the WCS
    # generate all the camera position
    xs, ys, zs = zip(*camera_positions)
    ax.scatter(xs, ys, zs, c='red', marker='o',  s=100, label='Generated Camera Positions')
    # Draw the TCS in the WCS
    ax.quiver(wcs_o[0], wcs_o[1], wcs_o[2], 20, 0, 0, color='green', label='TCS X-axis', arrow_length_ratio=0.1)
    ax.quiver(wcs_o[0], wcs_o[1], wcs_o[2], 0, 20, 0, color='blue', label='TCS Y-axis', arrow_length_ratio=0.1)
    ax.quiver(wcs_o[0], wcs_o[1], wcs_o[2], 0, 0, 20, color='red', label='TCS Z-axis', arrow_length_ratio=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization in World Coordinate System')
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Value of K')
    ax.legend()
    plt.show()

    ############################################################
    # parameters of camera intrinsic
    fx,fy=256,256
    cx,cy=256,256
    s=0
    intrinsic = intrinsic_matrix(fx,fy,cx,cy,s)

    # Test the function
    width, height = cx*2, cy*2  # Simplified resolution for demonstration
    print("原点坐标：",tcs_to_wcs(torch.tensor([0.,0.,0.])))

    i = 0
    for camera_position in camera_positions:
        extrinsic = extrinsic_matrix(camera_position, tcs_to_wcs(torch.tensor([0., 0., 0.])))
        rays = generate_rays(width, height, intrinsic, extrinsic, camera_position)
        pixel = torch.tensor([cx, cy])
        #visualization and ray casting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x, y, z, c=temperature_field, marker='o', s=2, alpha=0.8, cmap='viridis')

        # After plotting camera positions and before showing the plot
        num_rays_to_visualize = 10
        for ray_origin, ray_direction in rays[:num_rays_to_visualize]:
            visualize_ray(ax, ray_origin, ray_direction)
        plt.show()

        # Perform ray casting on the reduced rays
        ray_casted_image = ray_casting(rays, width, height, sampling_distance=0.5, max_distance=80.0)

        # Display the resulting image
        plt.imshow(ray_casted_image.cpu(), cmap='viridis')
        plt.colorbar(label='Value of K')
        plt.title('2D Projection of the 3D Temperature Field (Ray Casting)')
        plt.show()

        # Save the data
        save_data(ray_casted_image, camera_position,i*interval, intrinsic, extrinsic,current_time)
        i += 1
    sys.exit()