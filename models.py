import torch



class Base_Model(torch.nn.Module):
    def __init__(self,num_pos,pos_encode_dims):

        super(Base_Model,self).__init__()
        self.num_pos = num_pos
        self.pos_encode_dims = pos_encode_dims
        # self.linear1 = torch.nn.Linear(in_features = (input_dim, 2 * 3 * POS_ENCODE_DIMS + 3),out_features=(64))
        self.linear1 = torch.nn.Linear(in_features=2*3*pos_encode_dims+3,out_features=256)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=256,out_features=256)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(in_features=256,out_features=256)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(in_features=256,out_features=256)
        self.relu4 = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(in_features=355,out_features=256)
        self.relu5 = torch.nn.ReLU()
        self.linear6 = torch.nn.Linear(in_features=256,out_features=256)
        self.relu6 = torch.nn.ReLU()
        self.linear7 = torch.nn.Linear(in_features=256,out_features=256)
        self.relu7 = torch.nn.ReLU()
        self.linear8 = torch.nn.Linear(in_features=256,out_features=4)
        self.relu8 = torch.nn.ReLU()

    def forward(self,x):

       out = self.linear1(x)
       out = self.relu1(out)
       out = self.linear2(out)
       out = self.relu2(out)
       out = self.linear3(out)
       out = self.relu3(out)
       out = self.linear4(out)
       out = self.relu4(out)
       out = torch.cat((out,x),dim=-1)
       out = self.linear5(out)
       out = self.relu5(out)
       out = self.linear6(out)
       out = self.relu6(out)
       out = self.linear7(out)
       out = self.relu7(out)
       out = self.linear8(out)
       out = self.relu8(out)
       return out
