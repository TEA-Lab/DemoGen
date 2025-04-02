import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetOriginEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginEncoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x
        
class PointNetOriginNoBNEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginNoBNEncoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = F.relu(x)


        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x

class PointNetOriginNoTEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginNoTEncoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        # trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        # x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x
  

class PointNetOriginMLPEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginMLPEncoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 1024)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        
        x = self.conv1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x.transpose(2, 1)).transpose(2, 1)))
        x = self.bn3(self.conv3(x.transpose(2, 1)).transpose(2, 1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x

   
class PointNetOriginLinearNoBNNo1024Encoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginLinearNoBNNo1024Encoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 256)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = self.conv1(x)
        x = F.relu(x)


        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 256)
        x = self.projection(x)
        return x
 
class PointNetOriginLinear256Encoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginLinear256Encoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        
        x = self.conv1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x.transpose(2, 1)).transpose(2, 1)))
        x = self.bn3(self.conv3(x.transpose(2, 1)).transpose(2, 1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = self.projection(x)
        return x

class PointNetOriginLinearNoTEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=1024,
                 **kwargs
                 ):
        super(PointNetOriginLinearNoTEncoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 1024)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        # trans = self.stn(x)
        x = x.transpose(2, 1)
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        x = self.conv1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x.transpose(2, 1)).transpose(2, 1)))
        x = self.bn3(self.conv3(x.transpose(2, 1)).transpose(2, 1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x
      
class PointNetOriginLinearNoTNo1024Encoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=1024,
                 **kwargs
                 ):
        super(PointNetOriginLinearNoTNo1024Encoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        # trans = self.stn(x)
        x = x.transpose(2, 1)
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        x = self.conv1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x.transpose(2, 1)).transpose(2, 1)))
        x = self.bn3(self.conv3(x.transpose(2, 1)).transpose(2, 1))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = self.projection(x)
        return x



class PointNetOriginNoTNoBNEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=1024,
                 **kwargs
                 ):
        super(PointNetOriginNoTNoBNEncoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = F.relu(x)


        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x


class PointNetOriginLinearNoTNoBNEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=1024,
                 **kwargs
                 ):
        super(PointNetOriginLinearNoTNoBNEncoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 1024)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(1024, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        # trans = self.stn(x)
        x = x.transpose(2, 1)
        # if D > 3:
        #     feature = x[:, :, 3:]
        #     x = x[:, :, :3]
        # x = torch.bmm(x, trans)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        x = self.conv1(x)
        x = F.relu(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.projection(x)
        return x

class PointNetOrigin256Encoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOrigin256Encoder, self).__init__()
        self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 256)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)

        self.feature_transform = False
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: B, N, D
        # transposed: B, D, N
        x = x.transpose(2, 1)
        
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        
        x = self.conv1(x)
        x = x.transpose(2, 1)
        x = F.relu(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.conv2(x.transpose(2, 1)).transpose(2, 1))
        x = self.conv3(x.transpose(2, 1)).transpose(2, 1)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = self.projection(x)
        return x

 
class PointNetOriginSimpleEncoder(nn.Module):
    def __init__(self, in_channels: int=3,
                 out_channels: int=256,
                 **kwargs
                 ):
        super(PointNetOriginSimpleEncoder, self).__init__()
        # self.stn = STN3d(in_channels)
        self.conv1 = torch.nn.Linear(in_channels, 64)
        self.conv2 = torch.nn.Linear(64, 128)
        self.conv3 = torch.nn.Linear(128, 256)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)

        # self.feature_transform = False
        # if self.feature_transform:
        #     self.fstn = STNkd(k=64)
        
        # projection
        self.projection = nn.Linear(256, out_channels)

    def forward(self, x):
        # x: B, N, D
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 256)
        
        x = self.projection(x)
        return x

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss