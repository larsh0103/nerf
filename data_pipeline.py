# import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# plt.imshow(images[np.random.randint(low=0, high=num_images)])

class TinyNerfDataset(torch.utils.data.Dataset):
    def __init__(self,images,poses,focal,height,width,num_samples,pos_encode_dims):
        self.images = images
        self.poses = poses
        self.focal = focal
        self.height = height
        self.width = width
        self.num_samples =num_samples
        self.pos_encode_dims=pos_encode_dims

        # self.rays_and_points = [map_fn(pose,height=self.height,width=self.width,focal=self.focal,num_samples=self.num_samples) for pose in self.poses]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        rays_and_points=[map_fn(pose,height=self.height,width=self.width,
                        focal=self.focal,num_samples=self.num_samples,
                        pos_encode_dims=self.pos_encode_dims) for pose in self.poses[[idx],:,:]]
        return (self.images[idx],rays_and_points[0])


def apply_fourier_mappping(x,pos_encode_dims=16):
    """ Takes a 3D point x, as input.
    And applies fourier feature mapping to the point,
    converting the point to higher dimensional feature space, 
    as it has been shown that this can improve NN's abililty to learn
    high frequency details. 
    outputs fourier features for point.
    Applies sparse sampling using exponents?.
    """
    features = [x]
    for i in range(pos_encode_dims):
        for fn in [torch.sin,torch.cos]:
            features.append(fn(2.0 ** i * x))
    return torch.cat(features,dim=-1)
    
def get_rays(height,width,focal,pose):


    #Make grid of pixels for 2d image
    X,Y = torch.meshgrid(
        torch.range(0,width-1,dtype=torch.float32),
    torch.range(0,height-1,dtype=torch.float32),indexing='xy')

    #Normalize
    X_normalized = (X- width *0.5) / focal
    Y_normalized = (Y -height *0.5) / focal

    #create direction unit vectors for each pixel
    direction_unit_vectors = torch.stack([X_normalized, -Y_normalized, -torch.ones_like(X_normalized)],dim=-1)

    ## camera transform that converts 
    camera_transform_matrix = pose[:3,:3]
    height_width_focal = pose[:3,-1]

    transformed_directions = direction_unit_vectors[...,None,:] * camera_transform_matrix
    ray_directions = torch.sum(transformed_directions,dim=-1)
    ray_origins = torch.broadcast_to(torch.from_numpy(height_width_focal),ray_directions.shape)
    return (ray_origins,ray_directions)

def render_flat_rays(ray_origins,ray_directions,near,far,num_samples,pos_encode_dims,rand=True):
    
    # Equation: r(t) = o+td -> Building the "t" here.
    #Compute which points to query along the ray
    t_vals = torch.linspace(near,far, num_samples)

    ## Add random noise to sample points to make the sampling continuous
    if rand:
        shape = list(ray_origins.shape[:-1]) + [num_samples]
        noise = torch.rand(size=shape) * (far-near) / num_samples
        t_vals = t_vals + noise
    
    # Equation: r(t) = o + td -> Building the "r" here
    rays = ray_origins[...,None,:] +  ray_directions[...,None,:] * t_vals[...,None]
    rays_flat = torch.reshape(rays, [-1, 3])
    rays_flat = apply_fourier_mappping(rays_flat,pos_encode_dims=pos_encode_dims)
    return (rays_flat,t_vals)

def map_fn(pose,height=100,width=100,focal=6,num_samples=32,pos_encode_dims=16):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(height=height, width=width, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=num_samples,
        pos_encode_dims=pos_encode_dims
    )
    return (rays_flat, t_vals)

def get_rgb_depth(model,rays_flat,t_vals,batch_size,H,W,num_samples,device="cuda",train=False,rand=True):
    if device!='cpu':
        rays_flat = rays_flat.to(device=device)
        t_vals = t_vals.to(device=device)
    
  
    predictions = model(rays_flat)
    predictions = torch.reshape(predictions,shape=(batch_size,H,W,num_samples,4))
    rgb = torch.sigmoid(predictions[...,:-1])
    # print("rgb mean:",rgb.mean())
    sigma_a = F.relu(predictions[...,-1])
    del predictions
    delta = t_vals[...,1:] -t_vals[...,:-1]
    if rand:
        delta = torch.cat([delta,torch.broadcast_to(torch.tensor([1e10],device=device),size=(batch_size,H,W,1))],dim=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta)
    else:
        delta = torch.cat([delta,torch.broadcast_to(torch.tensor([1e10],device=device),size=(batch_size,1))],dim=-1)
        alpha = 1.0 - torch.exp(-sigma_a * delta[:, None, None, :])

    # transmittance
    exp_term = 1.0 - alpha
    epsilon = 1e-10
    ## Need to make cumprod exclusive, meaining not a,b,c > a,b*a,c*a*b but 1,a,a*b
    transmittance = torch.cumprod(exp_term + epsilon,dim=-1)
    t = torch.zeros_like(transmittance)
    t+= transmittance 
    t[:,:,:,1:] = t[:,:,:,:-1]
    t[:,:,:,0] = torch.tensor([1])
    weights = alpha*t
    rgb = torch.sum(weights[...,None] * rgb, dim=-2)
    if rand:
        depth_map = torch.sum(weights *t_vals , dim=-1)
    else:
        depth_map = torch.sum(weights *t_vals[:,None,None] , dim=-1)
    return (rgb, depth_map)



def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ]
    return np.array(matrix, dtype=np.float32)

def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
    return np.array(matrix, dtype=np.float32)

def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [np.cos(theta), 0, -np.sin(theta), 0],
        [0, 1, 0, 0],
        [np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1],
    ]
    return np.array(matrix, dtype=np.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return np.array(c2w,dtype=np.float32)
