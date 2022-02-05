import torch
import torch.nn.functional as F


class Base_Model(torch.nn.Module):
    def __init__(self,num_pos,pos_encode_dims):

        super(Base_Model,self).__init__()
        self.num_pos = num_pos
        self.pos_encode_dims = pos_encode_dims
        # self.linear1 = torch.nn.Linear(in_features = (input_dim, 2 * 3 * POS_ENCODE_DIMS + 3),out_features=(256))
        self.linear1 = torch.nn.Linear(in_features=2*3*pos_encode_dims+3,out_features=64)
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
        self.linear8 = torch.nn.Linear(in_features=64,out_features=4)
        torch.nn.init.xavier_uniform_(self.linear8.weight)

    def forward(self,x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        # out = torch.cat((out,x),dim=-1)
        out = F.relu(self.linear5(out))
        out = F.relu(self.linear6(out))
        out = F.relu(self.linear7(out))
        out = self.linear8(out)
        return out
