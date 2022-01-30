# import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
data_file = "tiny_nerf_data.npz"

data = np.load(data_file)
images = data["images"]
im_shape = images.shape
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])
print(poses[0][:3,:3])
print(poses[0][:,:])
print(H,W)
print(focal)
# plt.imshow(images[np.random.randint(low=0, high=num_images)])


NUM_FOURIER_FEATURES = 16

def apply_fourier_mappping(x):
    """ Takes a 3D point x, as input.
    And applies fourier feature mapping to the point,
    converting the point to higher dimensional feature space, 
    as it has been shown that this can improve NN's abililty to learn
    high frequency details. 
    outputs fourier features for point.
    Applies sparse sampling using exponents?.
    """
    features = [x]
    for i in range(NUM_FOURIER_FEATURES):
        for fn in [torch.cos,torch.sin]:
            features.append(fn(2*np.pi**i*x))
    print(len(features))
    return torch.cat(features,dim=-1)
    
def get_rays(height,width,focal,pose):


    #Make grid of pixels for 2d image
    X,Y = torch.meshgrid(
        torch.range(0,width-1,dtype=torch.float32),
    torch.range(0,height-1,dtype=torch.float32),indexing='xy')

    print(X.shape,Y.shape)

    #Normalize
    X_normalized = (X- width/2) / focal
    Y_normalized = (Y -height/2) / focal

    #create direction unit vectors for each pixel
    direction_unit_vectors = torch.stack([X_normalized, -Y_normalized, -torch.ones_like(X_normalized)],axis=-1)

    ## camera transform that converts 
    camera_transform_matrix = pose[:3,:3]
    height_width_focal = pose[:3,-1]

    transformed_directions = direction_unit_vectors[...,None,:] * camera_transform_matrix
    ray_directions = torch.sum(transformed_directions,dim=-1)
    ray_origins = torch.broadcast_to(torch.from_numpy(height_width_focal),ray_directions.shape)
    return (ray_origins,ray_directions)

def render_flat_rays(ray_origins,ray_directions,near,far,num_samples):
    
    # Equation: r(t) = o+td -> Building the "t" here.
    #Compute which points to query along the ray
    t_vals = torch.linspace(near,far, num_samples)

    ## Add random noise to sample points to make the sampling continuous

    shape = list(ray_origins.shape[:-1]) + [num_samples]
    print(shape)
    noise = torch.rand(size=shape) * (far-near) / num_samples
    t_vals = t_vals + noise
    
    # Equation: r(t) = o + td -> Building the "r" here
    rays = ray_origins[...,None,:] + ( 
        ray_directions[...,None,:] * t_vals[...,None]
    )
    rays_flat = torch.reshape(rays, [-1, 3])
    rays_flat = apply_fourier_mappping(rays_flat)

# ray_origins, ray_directions = get_rays(height=H,width=W,focal=focal,pose=poses[0])
# render_flat_rays(ray_origins,ray_directions,near=2.0,far=6.0,num_samples=32)

def map_fn(pose):
    """Maps individual pose to flattened rays and sample points.

    Args:
        pose: The pose matrix of the camera.

    Returns:
        Tuple of flattened rays and sample points corresponding to the
        camera pose.
    """
    (ray_origins, ray_directions) = get_rays(height=H, width=W, focal=focal, pose=pose)
    (rays_flat, t_vals) = render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=NUM_SAMPLES,
        rand=True,
    )
    return (rays_flat, t_vals)
