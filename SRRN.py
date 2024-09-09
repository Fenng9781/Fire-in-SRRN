import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from Generate_dataset import generate_rays,W2T
from datetime import datetime
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


class SRRN(nn.Module):
    def __init__(self, input_dim=33, hidden_dim=256, num_layers=5, skip_in=[3]):
        super(SRRN, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_in = skip_in

        # Initial layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())

        # Middle layers (first six layers)
        for i in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_dim + (input_dim if i+1 in skip_in else 0), hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # Additional hidden layer for emissivity (without activation function)
        self.emissivity_layer = nn.Linear(hidden_dim, hidden_dim)

        # Continuing the rest of the layers after emissivity
        self.layers.append(nn.Linear(hidden_dim, 128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(128, 64))
        self.layers.append(nn.ReLU())

        # Output layer for temperature
        self.temperature_layer = nn.Linear(64, 1)

    def forward(self, x):
        inputs = x
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 5:  # Right before the additional layers after emissivity layer
                break
            if i in self.skip_in:
                x = torch.cat([x, inputs], dim=-1)
            x = layer(x)

        # Emissivity output
        x = self.emissivity_layer(x)
        x = torch.relu(x)
        # Continue through the rest of the network for temperature
        for layer in self.layers[i:]:
            x = layer(x)
        temperature = torch.relu(self.temperature_layer(x))
        return temperature


def get_dataset(camera_config):
    with open(camera_config,'r') as file:
        lines = file.readlines()
    # Initialize empty lists to store extracted data
    camera_positions = []
    intrinsics = []
    extrinsics = []
    projection_filenames = []

    i=0
    while i < len(lines):
        line = lines[i].strip()
        if line == "Camera Position:":
            camera_positions.append(eval(lines[i + 1].strip()))
            i += 2
        elif line == "Intrinsic Parameters:":
            intrinsics.append(eval(lines[i + 1].strip()))
            i += 2
        elif line == "Extrinsic Parameters:":
            extrinsics.append(eval(lines[i + 1].strip()))
            i += 2
        elif line == "Projection Filename:":
            projection_filenames.append(lines[i + 1].strip())
            i += 2
        else:
            i += 1

    return camera_positions, intrinsics, extrinsics, projection_filenames

def model_casting_batch_rays(rays, sampling_distance=0.5, max_distance=80.0):
    """
    Casts a batch of rays through the scene and computes the model's predicted values at sampled points along each ray.

    Parameters:
    - rays: a batch of rays, each defined by an origin and a direction.
    - sampling_distance: the distance between sampled points along each ray.
    - max_distance: the maximum distance to sample along each ray.

    Returns:
    - A tensor of the model's predicted values at the sampled points for each ray in the batch.
    """

    # Define the device (GPU if CUDA is available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Unpack ray origins and directions and move to GPU
    ray_origins = torch.stack([ray[0] for ray in rays]).to(device)
    ray_directions = torch.stack([ray[1] for ray in rays]).to(device)

    # Create a tensor of sampling distances and move to GPU
    ts = torch.arange(5, max_distance, sampling_distance).to(device)

    # Compute sample points for each ray in the batch and sampling distances
    sample_points = ray_origins[:, None, :] + ts[None, :, None] * ray_directions[:, None, :]

    # Compute the encoded points for model input
    sample_points = sample_points.reshape(-1, 3)  # Flatten the batch and sequence dimensions
    sample_points_t = W2T(sample_points)
    sample_points_pe = positional_encoding(sample_points_t)

    # return values.sum(dim=-1)  # Sum the values along each ray in the batch
    temperature = model(sample_points_pe)

    # Apply emissivity to the temperature values
    combined_values = temperature

    # Reshape for integration along each ray
    combined_values = combined_values.view(ray_origins.shape[0], -1)

    # Integrate values along each ray (you can replace this with a continuous integration method if needed)
    integrated_values = torch.trapz(combined_values,dx=sampling_distance)

    return integrated_values

def create_ray_dataset(camera_config):
    camera_positions, intrinsics, extrinsics, projection_filenames = get_dataset(camera_config)
    dataset = []
    for i in range(len(camera_positions)):
        intrinsic = torch.tensor(intrinsics[i], dtype=torch.float32)
        extrinsic = torch.tensor(extrinsics[i], dtype=torch.float32)
        camera_position = torch.tensor(camera_positions[i], dtype=torch.float32)
        width, height = intrinsic[0, 2].int(), intrinsic[1, 2].int()
        width = width*2
        height = height*2
        rays = generate_rays(width, height, intrinsic, extrinsic, camera_position)

        projection = pd.read_csv(projection_filenames[i]).values.flatten()  # Flatten the projection

        for ray, pixel_value in zip(rays, projection):
            data = {
                "ray": ray,
                "pixel_value": pixel_value
            }
            dataset.append(data)

    return dataset


def visualize_rays(dataset, num_rays=100):
    # Create a new matplotlib figure and axes.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Randomly select `num_rays` rays to plot.
    # We do this to avoid overcrowding the plot.
    selected_rays = np.random.choice(dataset, num_rays, replace=False)

    for data in selected_rays:
        ray = data['ray']
        ray_origin = ray[0].cpu().numpy()  # Move tensor to CPU before converting to numpy
        ray_direction = ray[1].cpu().numpy()  # Move tensor to CPU before converting to numpy

        # Calculate the end point of the ray for visualization.
        # We scale the direction by an arbitrary amount to make it visible in the plot.
        ray_end = ray_origin + 50 * ray_direction

        # Create an array representing the ray.
        ray_points = np.array([ray_origin, ray_end])

        # Plot the ray.
        ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2])

    # Set labels and title.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualized Rays')

    # Show the plot.
    plt.show()

def positional_encoding(x, L=5):
    """
    x: input tensor of shape (B, D) where B is the batch size and D is the dimension of input coordinates (e.g., 3 for 3D coordinates)
    L: number of frequency levels
    Returns: encoded tensor of shape (B, D * 2 * L)
    """
    x_encoded = [x]
    B, D = x.shape
    for l in range(L):
        for d in range(D):  # loop over each dimension
            values = x[:, d]
            x_encoded.append(torch.sin(2**l * torch.pi * values).unsqueeze(-1))
            x_encoded.append(torch.cos(2**l * torch.pi * values).unsqueeze(-1))
    return torch.cat(x_encoded, dim=-1)

if __name__ == "__main__":

    dataset = create_ray_dataset("./dataset/20240909_153438/camera_details.txt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#################
    num_epochs = 50
    batch_size = 3000

    # Initializes the TensorBoard writer
    writer = SummaryWriter('runs/camerra_pos')

    # 2. Set up the NeRF Model
    model = SRRN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4,weight_decay=0.95)

    loss_fn = torch.nn.MSELoss()

    start_time = datetime.now()
    print("num of rays:",len(dataset))
    print("start time:",start_time)
    start_time_model = start_time.strftime('%m_%d_%H_%M')
    model_save_dir = "./model_saved/" + start_time_model+"/"

    # Create a directory named by the current time
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # Training Loop
    for epoch in range(num_epochs):
        # Shuffle the dataset at the beginning of each epoch
        random.shuffle(dataset)

        epoch_loss = 0.0  # Initialize epoch loss
        # Process rays in batches
        for batch_start in range(0, len(dataset), batch_size):
            # Zero gradients
            optimizer.zero_grad()

            # Create lists to store rays and target values for the current batch
            batched_rays, target_pixel_values = [], []
            for data in dataset[batch_start:batch_start + batch_size]:
                batched_rays.append(data["ray"])
                target_pixel_values.append(data["pixel_value"])

            # Convert lists to tensors
            target_pixel_values = torch.tensor(target_pixel_values, dtype=torch.float32).to(device)

            # Render pixel values using the model for the entire batch
            rendered_pixel_values = []

            rendered_pixel_values = model_casting_batch_rays(batched_rays)

            # Calculate loss for the batch
            loss = loss_fn(rendered_pixel_values, target_pixel_values)

            # print(loss)
            epoch_loss += loss.item()  # Accumulate batch loss
            # Log loss to TensorBoard after each batch
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataset) + batch_start // batch_size)
            # Backpropagate
            loss.backward()

            # Optionally log gradients of model parameters
            for name, param in model.named_parameters():
                # print(name, param.grad)
                writer.add_histogram(f'gradients/{name}', param.grad, epoch * len(dataset) + batch_start // batch_size)

            optimizer.step()
            current_time = datetime.now()
            print(f"Epoch {epoch + 1}/{num_epochs} - Batch {batch_start // batch_size + 1} - Loss: {loss.item()} - time: {current_time-start_time}")
            start_time = datetime.now()
        # Log average loss for the epoch to TensorBoard
        writer.add_scalar('Average Loss/epoch', epoch_loss / (len(dataset) / batch_size), epoch)

        if (epoch+1) % 1 == 0 and epoch != 0:
            # Save the model parameters
            current = current_time.strftime('%H_%M_%S')
            model_save_path = model_save_dir + current + ".pth"
            optimizer_save_path = model_save_dir + current + "_optimizer.pth"
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), optimizer_save_path)
            print(f"Model parameters saved to {model_save_path}")
    # Close the TensorBoard writer at the end of training
    writer.close()
    current_time = datetime.now()
    print("Training complete!")
    current = current_time.strftime('%H_%M_%S')
    model_save_path = model_save_dir + current + ".pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model parameters saved to {model_save_path}")

    # model = MLP().to(device)  # or however you define your model
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()  # Set the model to evaluation mode
    sys.exit()
