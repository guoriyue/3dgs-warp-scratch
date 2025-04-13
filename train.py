import torch
import torch.nn as nn
import torch.optim as optim

class GaussianCloud(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.xyz = nn.Parameter(torch.randn(N, 3))       # Position
        self.scale = nn.Parameter(torch.ones(N, 3) * 0.01)  # Scale
        self.color = nn.Parameter(torch.rand(N, 3))       # RGB
        self.opacity = nn.Parameter(torch.ones(N) * 0.5)  # Opacity

    def forward(self):
        # Just return the parameters for now
        return {
            'xyz': self.xyz,
            'scale': self.scale,
            'color': self.color,
            'opacity': self.opacity
        }

# Dummy renderer for concept (replace with Warp or custom CUDA)
def render_gaussians(params, camera):
    # Input: dict of gaussian params + camera matrices
    # Output: image of shape (H, W, 3)
    # You can write a simplified version that rasterizes in 2D for now
    image = torch.zeros(128, 128, 3).to(params['xyz'].device)
    return image

# Photometric loss
loss_fn = nn.MSELoss()

# Training loop
model = GaussianCloud(N=10000).cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for step in range(1000):
    # Sample camera view
    camera = sample_random_camera()

    # Render current scene
    params = model()
    rendered_image = render_gaussians(params, camera)

    # Get GT image (128x128x3) for that view
    gt_image = load_gt_image(camera)

    # Compute loss
    loss = loss_fn(rendered_image, gt_image)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"[{step}] Loss: {loss.item():.4f}")
