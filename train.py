from random import shuffle
import torch
from models import Base_Model
from data_pipeline import TinyNerfDataset
from tqdm import tqdm 
import numpy as np
import wandb
import argparse
import torchvision
import os
from torchsummary import summary
import sys
import torch.nn.functional as F
from data_pipeline import get_rgb_depth


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--sample_interval", type=int, default=15, help="interval between image sampling")
parser.add_argument("--log_interval", type=int, default=5, help="log every n batches")
parser.add_argument("--pos_encode_dims", type=int,default=16, help="num fourier features per input dimension")
parser.add_argument("--data_file",type=str,default="tiny_nerf_data.npz")
parser.add_argument("--num_samples", type=int, default=16, help="number of samples per ray")
opt = parser.parse_args()
print(opt)

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

ds_train = TinyNerfDataset(images[:split_index],poses[:split_index],focal,H,W,num_samples=opt.num_samples,pos_encode_dims=opt.pos_encode_dims)
train_dataloader = torch.utils.data.DataLoader(ds_train,batch_size=opt.batch_size,shuffle=True)

ds_val = TinyNerfDataset(images[split_index:],poses[split_index:],focal,H,W,num_samples=opt.num_samples,pos_encode_dims=opt.pos_encode_dims)
val_dataloader = torch.utils.data.DataLoader(ds_val,batch_size=opt.batch_size,shuffle=True)



model = Base_Model(num_pos = H * W * opt.num_samples,pos_encode_dims=opt.pos_encode_dims)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
summary(model,input_size=(opt.num_samples*H*W,opt.pos_encode_dims*2*3+3))
optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
criterionMSE =torch.nn.MSELoss()

for epoch in range(opt.n_epochs):
    total_loss = torch.tensor(0.).to(device)
    for i, XY_pairs  in enumerate(tqdm(train_dataloader)):
        (ims,rays) = XY_pairs
        (rays_flat, t_vals) = rays
        ims = ims.to(device=device)
        # print(predictions)
        (rgb,depth) = get_rgb_depth(model,rays_flat,t_vals,batch_size=ims.shape[0],H=H,W=W,num_samples=opt.num_samples)
        loss = criterionMSE(rgb,ims)
        # print("rgb mean",rgb.mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    # if (epoch * len(train_dataloader) + i) % opt.log_interval == 0:
    wandb.log({"MSE_loss":total_loss.item()/(len(train_dataloader)*opt.batch_size)})

    # if (epoch * len(train_dataloader) + i) % opt.sample_interval == 0:

    total_val_loss = torch.tensor(0.).to(device)
    batches_done = epoch * len(train_dataloader) + i
    for i, XY_pairs  in enumerate(tqdm(val_dataloader)):
        with torch.no_grad():
            (ims,rays) = XY_pairs
            (rays_flat, t_vals) = rays
            ims = ims.to(device=device)
            (rgb,depth) = get_rgb_depth(model,rays_flat,t_vals,batch_size=ims.shape[0],H=H,W=W,num_samples=opt.num_samples)
            val_loss = criterionMSE(rgb,ims)
            total_val_loss += val_loss.item()
    wandb.log({"Val_MSE_Loss":total_val_loss.item()/(len(val_dataloader)*opt.batch_size)})
    wandb.log({"Predicted_Image":wandb.Image(torch.moveaxis(rgb[0],-1,0)),
    "Predicted_Depth": wandb.Image(torch.moveaxis(depth[0,...,None],-1,0)),
    "GT_Image": wandb.Image(torch.moveaxis(ims[0],-1,0))})
torch.save(model.state_dict(),os.path.join(wandb.run.dir,f"nerf_{(epoch*len(train_dataloader) + i)}.pth"))

            

    
# ##todo
# ### optimize memory usage and increase batch size and learning rate