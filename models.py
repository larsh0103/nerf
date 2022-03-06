from turtle import pos
import torch
import torch.nn.functional as F


class Base_Model(torch.nn.Module):
    def __init__(self,num_pos,pos_encode_dims):

        super(Base_Model,self).__init__()
        self.num_pos = num_pos
        self.pos_encode_dims = pos_encode_dims
        # self.linear1 = torch.nn.Linear(in_features = (input_dim, 2 * 3 * POS_ENCODE_DIMS + 3),out_features=(256))
        # 2*3*pos_encode_dims+
        self.linear1 = torch.nn.Linear(in_features=99,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        self.linear3 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        self.linear4 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear4.weight)
        self.linear5 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear5.weight)
        self.linear6 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear6.weight)
        self.linear7 = torch.nn.Linear(in_features=64,out_features=64)
        torch.nn.init.xavier_uniform_(self.linear7.weight)
        self.linear8 = torch.nn.Linear(in_features=163,out_features=4)
        torch.nn.init.xavier_uniform_(self.linear8.weight)

    def forward(self,x):
    #     ray_origins,ray_directions = self.get_rays(100,100,6,pose)
    #     (rays_flat, t_vals) = self.render_flat_rays(
    #     ray_origins=ray_origins,
    #     ray_directions=ray_directions,
    #     near=2.0,
    #     far=6.0,
    #     num_samples=self.num_samples,
    #     pos_encode_dims=self.pos_encode_dims
    # )
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        out = F.relu(self.linear5(out))
        out = F.relu(self.linear6(out))
        out = F.relu(self.linear7(out))
        out = torch.cat((out,x),dim=-1)
        out = self.linear8(out)
        return out


class PreprocessingModel(torch.nn.Module):
    def __init__(self,height=100,width=100,focal=6,num_samples=32,pos_encode_dims=16):
        super(PreprocessingModel).__init__()
        self.height=height
        self.width=width
        self.focal=focal
        self.num_samples=num_samples
        self.pos_encode_dims=pos_encode_dims
        self.sigmoid = torch.nn.Sigmoid
        self.L

    def forward(self,pose):
        #Make grid of pixels for 2d image
        X,Y = torch.meshgrid(
            torch.range(0,self.width-1,dtype=torch.float32),
        torch.range(0,self.height-1,dtype=torch.float32),indexing='xy')

        #Normalize
        X_normalized = (X- self.width *0.5) / self.focal
        Y_normalized = (Y -self.height *0.5) / self.focal

        #create direction unit vectors for each pixel
        direction_unit_vectors = torch.stack([X_normalized, -Y_normalized, -torch.ones_like(X_normalized)],dim=-1)

        ## camera transform that converts 
        camera_transform_matrix = pose[:3,:3]
        height_width_focal = pose[:3,-1]

        transformed_directions = direction_unit_vectors[...,None,:] * camera_transform_matrix
        ray_directions = torch.sum(transformed_directions,dim=-1)
        ray_origins = torch.broadcast_to(torch.from_numpy(height_width_focal),ray_directions.shape)
        return self.sigmoid(ray_origins)
