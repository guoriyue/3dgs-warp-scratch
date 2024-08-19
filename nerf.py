import math
import imageio
import os
import re
from datetime import datetime

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ChatGPT generated
def create_gif(image_folder, gif_path, duration=5):
    def sort_key(filename):
        number = re.search(r"(\d+)", filename)
        return int(number.group(1)) if number else 0

    filenames = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')],
                       key=sort_key)
    images = [imageio.imread(filename) for filename in filenames]
    imageio.mimsave(gif_path, images, duration=duration)

def load_data():
    data = np.load(f"./tiny_nerf_data.npz")

    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0

    # Cameras for the training images 
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]

    # Validation images: 
    images_val = data["images_val"] / 255.0

    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]

    # Test cameras for novel-view video rendering: 
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]

    # Camera focal length
    focal = data["focal"]  # float
    
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal

def transform(c2w, x_c):
    B, H, W, _ = x_c.shape
    x_c_homogeneous = torch.cat([x_c, torch.ones(B, H, W, 1, device=x_c.device)], dim=-1)

    # batched matmul
    x_w_homogeneous_reshaped = x_c_homogeneous.view(B, -1, 4)  # [100, 40000, 4]
    x_w_homogeneous_reshaped = x_w_homogeneous_reshaped.permute(0, 2, 1)
    x_w_homogeneous_reshaped = c2w.bmm(x_w_homogeneous_reshaped)
    x_w_homogeneous = x_w_homogeneous_reshaped.permute(0, 2, 1).view(B, H, W, 4)
    x_w = x_w_homogeneous[:, :, :, :3]
    return x_w

def intrinsic_matrix(fx, fy, ox, oy):
    K = torch.tensor([[fx,  0, ox],
                      [ 0, fy, oy],
                      [ 0,  0,  1]])
    return K

def pixel_to_camera(K, uv, s):
    B, H, W, C = uv.shape
    uv_reshaped = uv.view(B, -1, 3).permute(0, 2, 1)
    uv_homogeneous_reshaped = torch.cat([uv_reshaped[:, 1:], torch.ones((B, 1, H*W), device=uv.device)], dim=1)
    K_inv = torch.inverse(K)
    uv_homogeneous_reshaped = torch.stack((uv_homogeneous_reshaped[:, 1], uv_homogeneous_reshaped[:, 0], uv_homogeneous_reshaped[:, 2]), dim=1)
    x_c_homogeneous_reshaped = K_inv.bmm(uv_homogeneous_reshaped)
    x_c_homogeneous = x_c_homogeneous_reshaped.permute(0, 2, 1).view(B, H, W, 3)
    x_c = x_c_homogeneous * s
    
    return x_c

def pixel_to_ray(K, c2w, uv):
    # find x_c
    B, H, W, C = uv.shape # C = (image_idx, y, x)
    x_c = pixel_to_camera(K, uv, torch.ones((B, H, W, 1), device=uv.device))
    
    w2c = torch.inverse(c2w)
    R = w2c[:, :3, :3]
    R_inv = torch.inverse(R)
    T = w2c[:, :3, 3]
    # ray origins
    r_o = -torch.bmm(R_inv, T.unsqueeze(-1)).squeeze(-1)
    
    # ray directions
    x_w = transform(c2w, x_c)
    r_o = r_o.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1)
    r_d = (x_w - r_o) / torch.norm((x_w - r_o), dim=-1, keepdim=True)
    
    return r_o, r_d

def sample_along_rays(r_o, r_d, perturb=True, near=2.0, far=6.0, n_samples=64):
    t = torch.linspace(near, far, n_samples, device=r_o.device)
    if perturb:
        t = t + torch.rand_like(t) * (far - near) / n_samples
    x = r_o + r_d * t.unsqueeze(-1).unsqueeze(-1)
    return x

class RaysData:
    def __init__(self, images, K, c2w, device='cuda'):
        self.images = images
        self.K = K
        self.c2w = c2w
        self.device = device
        
        self.height = images.shape[1]
        self.width = images.shape[2]
        
        # create UV grid
        self.uv = torch.stack(torch.meshgrid(torch.arange(self.images.shape[0]), torch.arange(self.height), torch.arange(self.width)), dim=-1).to(device).float()
        # add 0.5 offset to each pixel
        self.uv[..., 1] += 0.5
        self.uv[..., 2] += 0.5
        self.uv_flattened = self.uv.reshape(-1, 3)
        
        self.r_o, self.r_d = pixel_to_ray(K, c2w, self.uv)
        self.pixels = images.reshape(-1, 3)
        self.r_o_flattened = self.r_o.reshape(-1, 3)
        self.r_d_flattened = self.r_d.reshape(-1, 3)
        
    def sample_rays(self, batch_size):
        # sample rays
        idx = torch.randint(0, self.pixels.shape[0], (batch_size,), device=self.pixels.device)
        return self.r_o_flattened[idx], self.r_d_flattened[idx], self.pixels[idx]
    
    # used for validation
    def sample_rays_single_image(self, image_index=None):
        if image_index is None:
            image_index = torch.randint(0, self.c2w.shape[0], (1,), device=self.device).item()
        start_idx = image_index * self.height * self.width
        end_idx = start_idx + self.height * self.width

        r_o_single = self.r_o_flattened[start_idx:end_idx]
        r_d_single = self.r_d_flattened[start_idx:end_idx]
        pixels_single = self.pixels[start_idx:end_idx]
        
        return r_o_single, r_d_single, pixels_single
        
def volrend(sigmas, rgbs, step_size):
    # received help from ChatGPT here to figure out cumsum
    B, N, _ = sigmas.shape
    # transmittance of first ray is 1
    T_i = torch.cat([torch.ones((B, 1, 1), device=rgbs.device), torch.exp(-step_size * torch.cumsum(sigmas, dim=1)[:, :-1])], dim=1)
    alpha = 1 - torch.exp(-sigmas * step_size)
    weights = alpha * T_i
    
    # accumulated_transmittance = torch.prod(1 - alpha, dim=1, keepdim=True)
    
    rendered_colors = torch.sum(weights * rgbs, dim=1)# + accumulated_transmittance.squeeze(1) * torch.ones((B, 3), device=rgbs.device)
    return rendered_colors

def volrend_depth(sigmas, step_size):
    depths = torch.linspace(2.0, 6.0, sigmas.shape[1], device=sigmas.device)
    T_i = torch.exp(-step_size * torch.cumsum(sigmas, dim=1))
    T_i = torch.cat([torch.ones((sigmas.shape[0], 1, 1), device=sigmas.device), T_i[:, :-1]], dim=1)
    alpha = 1 - torch.exp(-sigmas * step_size)
    weights = alpha * T_i
    rendered_depths = torch.sum(weights * depths.unsqueeze(0).unsqueeze(-1), dim=1)

    return rendered_depths

def positional_encoding(x, L):
    freqs = 2.0 ** torch.arange(L).float().to(x.device)
    x_input = x.unsqueeze(-1) * freqs * 2 * torch.pi
    encoding = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
    encoding = torch.cat([x, encoding.reshape(*x.shape[:-1], -1)], dim=-1) # add to original input    
    return encoding

def psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        # Initial input linear layers for xyz
        self.fc1_block1 = nn.Linear(2 * 3 * 10 + 3, 256)
        self.fc2_block1 = nn.Linear(256, 256)
        self.fc3_block1 = nn.Linear(256, 256)
        self.fc4_block1 = nn.Linear(256, 256)
        
        # Linear layers for ray direction
        self.fc1_d = nn.Linear(2 * 3 * 4 + 3, 256)
        
        # Linear layers after concatenation
        self.fc1_block2 = nn.Linear(256 + 2 * 3 * 10 + 3, 256)
        self.fc2_block2 = nn.Linear(256, 256)
        self.fc3_block2 = nn.Linear(256, 256)
        self.fc4_block2 = nn.Linear(256, 256)
        
        # Output layers
        self.linear_density = nn.Linear(256, 1)
        
        # Linear layers for RGB
        self.fc1_block3 = nn.Linear(256, 256)
        self.fc2_block3 = nn.Linear(256 + 2 * 3 * 4 + 3, 128)
        
        self.linear_rgb = nn.Linear(128, 3)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, r_d):
        # Positional encoding
        x_encoded = positional_encoding(x, L=10) #[64, 10000, 63]
        r_d_encoded = positional_encoding(r_d, L=4) #[10000, 27]
        
        # Process x through initial layers
        x = self.relu(self.fc1_block1(x_encoded))
        x = self.relu(self.fc2_block1(x))
        x = self.relu(self.fc3_block1(x))
        x = self.relu(self.fc4_block1(x))
        
        # concat x again
        x = torch.cat([x, x_encoded], dim=-1)
        
        x = self.relu(self.fc1_block2(x))
        x = self.relu(self.fc2_block2(x))
        x = self.relu(self.fc3_block2(x))
        x = self.fc4_block2(x)
        
        # output density
        density = self.relu(self.linear_density(x))
        
        # Process ray direction
        x = self.fc1_block3(x)
        
        x = torch.cat([x, r_d_encoded], dim=-1)
        # Process after concatenation
        x = self.relu(self.fc2_block3(x))
        rgb = self.linear_rgb(x)
        rgb = self.sigmoid(rgb)
        
        return rgb, density

def render_images(model, test_dataset):
    # testing
    model.eval()
    with torch.no_grad():
        for i in range(test_dataset.c2w.shape[0]):
            rays_o, rays_d, _ = test_dataset.sample_rays_single_image(i)
            points = sample_along_rays(rays_o, rays_d)
            points = points.permute(1, 0, 2)
            rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
            rgb, sigmas = model(points, rays_d)
            comp_rgb = volrend(sigmas, rgb, step_size=(6.0 - 2.0) / 64)
            # save image
            image = comp_rgb.reshape(200, 200, 3).cpu().numpy()
            plt.imsave(f"final_render/render_{i}.jpg", image)
            
    create_gif('final_render', 'final_render/training.gif')
    
def render_depth(model, test_dataset):
    # testing
    model.eval()
    with torch.no_grad():
        for i in range(test_dataset.c2w.shape[0]):
            rays_o, rays_d, _ = test_dataset.sample_rays_single_image(i)
            points = sample_along_rays(rays_o, rays_d)
            points = points.permute(1, 0, 2)
            rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
            rgb, sigmas = model(points, rays_d)
            comp_depth = volrend_depth(sigmas, step_size=(6.0 - 2.0) / 64)
            # save image
            image = comp_depth.reshape(200, 200).cpu().numpy()
            
            # normalize from 0 to 1
            depth_min = image.min()
            depth_max = image.max()
            image = (image - depth_min) / (depth_max - depth_min)
            plt.imsave(f"depth/render_{i}.jpg", image, cmap='gray')
            
    create_gif('depth', 'depth/training.gif')

def train_model(model, train_dataset, val_dataset, test_dataset, optimizer, criterion, iters=3000, batch_size=10000, device='cuda'):
    
    psnr_scores = []
    
    model.train()
    for i in tqdm(range(iters)):
        rays_o, rays_d, pixels = train_dataset.sample_rays(batch_size)
        points = sample_along_rays(rays_o, rays_d, perturb=True)
        points = points.permute(1, 0, 2)
        rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
        
        optimizer.zero_grad()
        rgb, sigmas = model(points, rays_d)
        comp_rgb = volrend(sigmas, rgb, step_size=(6.0 - 2.0) / 64)
        
        loss = criterion(comp_rgb, pixels)
        loss.backward()
        optimizer.step()
        
        # training PSNR
        print(f"Training PSNR: {psnr(comp_rgb.detach().cpu().numpy(), pixels.cpu().numpy())}")
        
        # validation
        if (i + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                rays_o, rays_d, pixels = val_dataset.sample_rays_single_image()
                points = sample_along_rays(rays_o, rays_d)
                points = points.permute(1, 0, 2)
                rays_d = rays_d.unsqueeze(1).repeat(1, points.shape[1], 1)
                rgb, sigmas = model(points, rays_d)
                comp_rgb = volrend(sigmas, rgb, step_size=(6.0 - 2.0) / 64)
                loss = criterion(comp_rgb, pixels)
                # print(f"Validation loss: {loss}")
                curr_psnr = psnr(comp_rgb.cpu().numpy(), pixels.cpu().numpy())
                print(f"Validation PSNR: {curr_psnr:.2f} dB")
                psnr_scores.append(curr_psnr)
                # save image
                image = comp_rgb.reshape(200, 200, 3).cpu().numpy()
                plt.imsave(f"nerf_output/iter{i+1}.jpg", image)
            model.train()
                        
    # save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"checkpoints/nerf_checkpoint_{timestamp}.pt")
    
    # create PSNR plot
    plt.figure()
    plt.plot(range(25, 3001, 25), psnr_scores)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Iteration')
    plt.savefig('plots/psnr_nerf.png')
    
    render_images(model, test_dataset)

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = load_data()
    
    # prepare data
    images_train = torch.tensor(images_train).float().to(device)
    c2ws_train = torch.tensor(c2ws_train).float().to(device)
    images_val = torch.tensor(images_val).float().to(device)
    c2ws_val = torch.tensor(c2ws_val).float().to(device)
    c2ws_test = torch.tensor(c2ws_test).float().to(device)
    focal = torch.tensor(focal).float().to(device)
    K_train = intrinsic_matrix(focal.item(), focal.item(), images_train.shape[1] / 2, images_train.shape[2] / 2).unsqueeze(0).repeat(images_train.shape[0], 1, 1).to(device)
    K_val = intrinsic_matrix(focal.item(), focal.item(), images_val.shape[1] / 2, images_val.shape[2] / 2).unsqueeze(0).repeat(images_val.shape[0], 1, 1).to(device)
    K_test = intrinsic_matrix(focal.item(), focal.item(), images_val.shape[1] / 2, images_val.shape[2] / 2).unsqueeze(0).repeat(c2ws_test.shape[0], 1, 1).to(device)
    
    # training
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    batch_size = 10000
    iters = 3000
    train_dataset = RaysData(images_train, K_train, c2ws_train)
    val_dataset = RaysData(images_val, K_val, c2ws_val)
    # pass in dummy images to test_dataset, won't be used
    test_dataset = RaysData(images_train[:60], K_test, c2ws_test)
    
    train_model(model, train_dataset, val_dataset, test_dataset, optimizer, criterion, iters=iters, batch_size=batch_size, device=device)
    render_images(model, test_dataset)