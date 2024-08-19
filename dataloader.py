import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Union, Callable
import torch


class Dataloader():
    def __init__(self):
        if not os.path.exists('tiny_nerf_data.npz'):
            print("Dataset not found")
            exit()

        # load the dataset from https://cseweb.ucsd.edu//~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
        self.data = np.load('tiny_nerf_data.npz')
        self.images = self.data['images']
        self.poses = self.data['poses']
        self.focal = self.data['focal']

        self.height, self.width = self.images.shape[1:3]
        self.near, self.far = 2.0, 6.0

        self.training_length = 100

    def show_image(self, image_index: int):
        # get the image and the pose at the given index
        image, pose = self.images[image_index], self.poses[image_index]
        plt.imshow(image)
        print('Pose')
        print(pose)
        plt.show()

    def show_cameras(self):
        # get rotation of the camera, i.e. the direction of the z-vector, since it points along the camera axis
        rotations = np.stack([np.sum([0, 0, -1]*pose[:3, :3], -1)
                             for pose in self.poses])
        # get the origin of the camera, i.e. the translation vector
        origins = self.poses[:, :3, -1]
        figure = plt.figure(figsize=(12, 8)).add_subplot(
            projection='3d')  # create a 3D plot
        _ = figure.quiver(origins[:, 0], origins[:, 1], origins[:, 2], rotations[:, 0], rotations[:, 1],
                          rotations[:, 2], length=0.5, normalize=True)  # plot the camera positions and directions
        figure.set_xlabel('X')
        figure.set_ylabel('Y')
        figure.set_zlabel('Z')
        plt.show()

    def get_rays(self, image_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_coordinates, y_coordinates = torch.meshgrid(torch.arange(self.width, dtype=torch.float32), torch.arange(
            100, dtype=torch.float32), indexing='xy')  # make a grid of x and y coordinates of the pixels
        # calculate the direction of the rays, this matrix multiplied with distance form optical center, w, will give the u, v, w coordinates in world space
        directions = torch.stack([(x_coordinates-self.width/2)/self.focal, -(
            y_coordinates-self.height/2)/self.focal, -torch.ones_like(x_coordinates)], dim=-1)
        # Shape of directions: (width, height, 3)

        # multiply the direction of the rays with the rotation matrix of the camera to get the direction of the rays in world space
        rays_directions = torch.sum(
            directions[..., None, :] * self.poses[image_index, :3, :3], -1)
        # get the origin of the camera in world space
        rays_origin = self.poses[image_index,
                                 :3, -1].expand(rays_directions.shape)

        # Shape of rays_origin: (width, height, 3), Shape of rays_directions: (width, height, 3)
        return rays_origin, rays_directions

    def sample_stratified(self, rays_origin: torch.Tensor, rays_directions: torch.Tensor, number_of_samples: int, binning: Optional[bool] = False) -> torch.Tensor:
        points_to_sample = torch.linspace(
            0., 1., number_of_samples)  # get the points to sample
        samples = self.near + (self.far - self.near)*(points_to_sample)
        # Shape: (number_of_samples)

        if binning:
            mids = (samples[1:] + samples[:-1])/2
            upper = torch.cat([mids, samples[-1:]])
            lower = torch.cat([samples[:1], mids])
            samples = torch.rand(number_of_samples)*(upper-lower) + lower

        # expand the samples to the shape of the rays x and y dimension, and add the number of samples as the last dimension
        samples = samples.expand(
            list(rays_origin.shape[:-1]) + [number_of_samples])
        # Shape: (width, height, number_of_samples)

        samples = rays_origin[..., None, :] + rays_directions[..., None,
                                                              :]*samples[..., :, None]  # calculate the samples in world space
        # Shape: (width, height, number_of_samples, 3)

        return samples


if __name__ == '__main__':
    dataloader = Dataloader()
    dataloader.show_image(10)
    dataloader.show_cameras()