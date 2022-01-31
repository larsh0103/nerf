import torch
from models import Base_Model
from data_pipeline import TinyNerfDataset
from tqdm import tqdm 
import numpy as np
import wandb
import argparse
import torchvision
import os



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between image sampling")
parser.add_argument("--log_interval", type=int, default=10, help="log every n batches")
parser.add_argument("--pos_encode_dims", type=int,default=16, help="num fourier features per input dimension")
parser.add_argument("--data_file",type=str,default="tiny_nerf_data.npz")
parser.add_argument("--num_samples", type=int, default=32, help="number of samples per ray")
opt = parser.parse_args()
print(opt)


def get_rgb_depth(predictions,t_vals,batch_size,device="cuda"):
    predictions = torch.reshape(predictions,shape=(batch_size,H,W,opt.num_samples,4))
    rgb = torch.sigmoid(predictions[...,:-1])
    sigma_a = predictions[...,-1]
    delta = t_vals[...,1:] -t_vals[...,:-1]
    delta = torch.cat([delta,torch.broadcast_to(torch.tensor([1e10],device=device),size=(batch_size,H,W,1))],dim=-1)
    alpha = 1.0 - torch.exp(-sigma_a * delta)
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
    depth_map = torch.sum(weights *t_vals , dim=-1)
    return (rgb, depth_map)



data = np.load(opt.data_file)
images = data["images"]
# for im in images:
#     print(im.mean())
im_shape = images.shape
(num_images, H, W, _) = images.shape
(poses, focal) = (data["poses"], data["focal"])

# for pose in poses[[1,2,3],:,:]:
#     print(pose.shape)
config = vars(opt)

wandb.init(project="nerf",config=config)


split_index = int(num_images*0.8)

ds_train = TinyNerfDataset(images[:split_index],poses[:split_index],focal,H,W,num_samples=opt.num_samples)
train_dataloader = torch.utils.data.DataLoader(ds_train,batch_size=opt.batch_size)

ds_val = TinyNerfDataset(images[split_index:],poses[split_index:],focal,H,W,num_samples=opt.num_samples)
val_dataloader = torch.utils.data.DataLoader(ds_val,batch_size=opt.batch_size)



model = Base_Model(num_pos = H * W * opt.num_samples,pos_encode_dims=opt.pos_encode_dims)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(),lr=10-5)
criterionMSE =torch.nn.MSELoss()

for epoch in range(opt.n_epochs):
    total_loss = torch.tensor(0.).to(device)

    for i, XY_pairs  in enumerate(tqdm(train_dataloader)):
        (images,rays) = XY_pairs
        (rays_flat, t_vals) = rays
        images = images.to(device=device)
        rays_flat = rays_flat.to(device=device)
        t_vals = t_vals.to(device=device)
        predictions = model(rays_flat)
        # print(predictions)
        (rgb,depth) = get_rgb_depth(predictions,t_vals,batch_size=images.shape[0])
        # print(images[0],rgb[0])
        loss = criterionMSE(images,rgb)
        optimizer.zero_grad()
        loss.backward()
        total_loss +=loss.item()
        optimizer.step()
    wandb.log({"MSE_loss":total_loss/len(train_dataloader),"step": (epoch*len(train_dataloader) + i) })

    # batches_done = epoch * len(train_dataloader) + i
    # if batches_done % opt.sample_interval == 0:
    wandb.log({"Predicted_Image":wandb.Image(torch.moveaxis(rgb[0],-1,0)),"GT_Image": wandb.Image(torch.moveaxis(images[0],-1,0))})
    torch.save(model.state_dict(),os.path.join(wandb.run.dir,f"nerf_{(epoch*len(train_dataloader) + i)}.pth"))
    total_val_loss =torch.tensor(0.).to(device)
    for i, XY_pairs  in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            model.eval()      
            (images,rays) = XY_pairs
            (rays_flat, t_vals) = rays
            images = images.to(device=device)
            rays_flat = rays_flat.to(device=device)
            t_vals = t_vals.to(device=device)
            predictions = model(rays_flat)
            (rgb,depth) = get_rgb_depth(predictions,t_vals,batch_size=images.shape[0])
            val_loss = criterionMSE(images,rgb)
            total_val_loss +=val_loss.item()
    wandb.log({"Val_MSE_Loss":total_val_loss/len(val_dataloader) })

            

    
##todo
### optimize memory usage and increase batch size and learning rate