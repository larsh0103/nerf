from turtle import pos
import data_pipeline
import torch
from tqdm import tqdm
import numpy as np
from models import Base_Model
import cv2
from PIL import Image


rgb_frames = []
batch_flat = []
batch_t = []
H,W = (100,100)
focal  = 138.88887889922103
num_samples=16
pos_encode_dims = 16
batch_size = 2

model = Base_Model(num_pos=H*W*num_samples,pos_encode_dims=pos_encode_dims)
model.to("cuda")
model.load_state_dict(torch.load("./wandb/latest-run/files/nerf_404.pth"))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video-20.mp4',fourcc,1,(W,H))

for index, theta in tqdm(enumerate(torch.linspace(0.0,360,120))):
    c2w = data_pipeline.pose_spherical(theta,-30.0, 6)
    ray_origins, ray_directions = data_pipeline.get_rays(H,W,focal,c2w)
    (rays_flat, t_vals) = data_pipeline.render_flat_rays(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        near=2.0,
        far=6.0,
        num_samples=num_samples,
        pos_encode_dims=pos_encode_dims, rand=True
    )

    if index % batch_size ==0 and index>0:
        batched_flat = torch.stack(batch_flat, dim=0)
        batched_t = torch.stack(batch_t, dim=0)

        with torch.no_grad():
            rgb, _ = data_pipeline.get_rgb_depth(
                    model, batched_flat, t_vals=batched_t,batch_size=len(batch_flat),H=H,W=W,num_samples=num_samples,device='cuda',rand=True
                )
            
            rgb_frames = [(img.cpu().numpy() * 255.0).astype(np.uint8) for img in rgb]
            # Image.fromarray((rgb[0].cpu().numpy()*255).astype(np.uint8)).save("test-pil.jpg")
            for img in rgb_frames:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(img)
            rgb_frames=[]
            batch_flat=[]
            batch_t = []


    batch_flat.append(rays_flat)
    batch_t.append(t_vals)


cv2.destroyAllWindows()
video.release()