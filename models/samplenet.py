from __future__ import print_function

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from knn_cuda import KNN

# try:
# from .src.soft_projection import SoftProjection
# from .src.chamfer_distance.chamfer_distance import ChamferDistance
# from .src import sputils

# debug, python samplenet.py
# from .src.soft_projection import SoftProjection
# from .src.chamfer_distance.chamfer_distance import ChamferDistance
# from .src import sputils
from src.soft_projection import SoftProjection # 计算flops用
from src.chamfer_distance.chamfer_distance import ChamferDistance
from src import sputils
# except (ModuleNotFoundError, ImportError) as err:
#     print(err.__repr__())
#     from models.src.soft_projection import SoftProjection
#     from chamfer_distance import ChamferDistance
#     import models.src.sputils as sputils


class SampleNet(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        # complete_fps=True,
        complete="fps",
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        # self.complete_fps = complete_fps    # true
        self.complete = complete    # fps


        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y))) # 256
        y = F.relu(self.bn_fc2(self.fc2(y))) # 256
        y = F.relu(self.bn_fc3(self.fc3(y))) # 256
        y = self.fc4(y) # B,96

        y = y.view(-1, 3, self.num_out_points) # B,3,32

        # Simplified points
        simp = y
        match = None
        proj = None

        # Projected points
        if self.training:
            if not self.skip_projection:
                proj = self.project(point_cloud=x, query_cloud=y)
            else:
                proj = simp

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())  # 用KNN找最近的1个点，idx=[1,1,32]

            """Notice that we detach the tensors and do computations in numpy,
            and then convert back to Tensors.
            This should have no effect as the network is in eval() mode
            and should require no gradients.
            """

            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            y = y.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx, axis=1)

            z = sputils.nn_matching(
                x, idx, self.num_out_points, complete=self.complete
            )

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":    # "bnc" in training
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        out = proj if self.training else match

        return simp, out        # B,3,32

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(sigma)
        return sigma



class SampleNet_nomatch(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps    # true

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y))) # 256
        y = F.relu(self.bn_fc2(self.fc2(y))) # 256
        # y = F.relu(self.bn_fc3(self.fc3(y))) # 256 有bug，y全变为0了
        y = self.fc3(y)
        y = self.bn_fc3(y)   # 经过BN之后全部为负数，所以relu后全0
        y = F.relu(y)

        y = self.fc4(y) # B,96

        y = y.view(-1, 3, self.num_out_points) # B,3,32

        # Simplified points
        simp = y
        proj = None

        # Projected points
        
        if not self.skip_projection:
            proj = self.project(point_cloud=x, query_cloud=y)
        else:
            proj = simp

        # Change to output shapes
        if self.output_shape == "bnc":    # "bnc" in training
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
      
        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()

        out = proj 

        return simp, out        # B,3,32

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(sigma)
        return sigma


class PureMLPNet(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(256)

        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps    # true

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(y1)))
        y3 = F.relu(self.bn3(self.conv3(y2)))  # Batch x 128 x NumInPoints
        y = torch.cat([y1, y2, y3], dim=1) # 消融实验residual


        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y))) # 256
        y = F.relu(self.bn_fc2(self.fc2(y))) # 256
        y = F.relu(self.bn_fc3(self.fc3(y))) # 256
        y = self.fc4(y) # B,96

        y = y.view(-1, 3, self.num_out_points) # B,3,32

        # Simplified points
        simp = y
        match = None
        proj = None

        # Projected points
        if self.training:
            if not self.skip_projection:
                proj = self.project(point_cloud=x, query_cloud=y)
            else:
                proj = simp

        # Matched points
        else:  # Inference
            # Retrieve nearest neighbor indices
            _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())  # 用KNN找最近的1个点，idx=[1,1,32]

            """Notice that we detach the tensors and do computations in numpy,
            and then convert back to Tensors.
            This should have no effect as the network is in eval() mode
            and should require no gradients.
            """

            # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
            x = x.permute(0, 2, 1).cpu().detach().numpy()
            y = y.permute(0, 2, 1).cpu().detach().numpy()

            idx = idx.cpu().detach().numpy()
            idx = np.squeeze(idx, axis=1)

            z = sputils.nn_matching(
                x, idx, self.num_out_points, complete_fps=self.complete_fps
            )

            # Matched points are in B x N x 3 format.
            match = torch.tensor(z, dtype=torch.float32).cuda()

        # Change to output shapes
        if self.output_shape == "bnc":    # "bnc" in training
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
        elif self.output_shape == "bcn" and match is not None:
            match = match.permute(0, 2, 1)
            match = match.contiguous()

        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()
        if match is not None:
            match = match.contiguous()

        out = proj if self.training else match

        return simp, out        # B,3,32

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(sigma)
        return sigma



# debug,
class SampleNet_nomatch_debug(nn.Module):
    def __init__(
        self,
        num_out_points,
        bottleneck_size,
        group_size,
        initial_temperature=1.0,
        is_temperature_trainable=True,
        min_sigma=1e-2,
        input_shape="bcn",
        output_shape="bcn",
        complete_fps=True,
        skip_projection=False,
        linear_dim=256,
        conv_dim=64,
        conv_dim2=128
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.name = "samplenet"

        self.conv1 = torch.nn.Conv1d(3, conv_dim, 1)
        self.conv2 = torch.nn.Conv1d(conv_dim, conv_dim, 1)
        self.conv3 = torch.nn.Conv1d(conv_dim, conv_dim, 1)
        self.conv4 = torch.nn.Conv1d(conv_dim, conv_dim2, 1)
        self.conv5 = torch.nn.Conv1d(conv_dim2, bottleneck_size, 1)

        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.bn3 = nn.BatchNorm1d(conv_dim)
        self.bn4 = nn.BatchNorm1d(conv_dim2)
        self.bn5 = nn.BatchNorm1d(bottleneck_size)

        self.fc1 = nn.Linear(bottleneck_size, linear_dim) # 全改成128
        self.fc2 = nn.Linear(linear_dim, linear_dim)
        self.fc3 = nn.Linear(linear_dim, linear_dim)
        self.fc4 = nn.Linear(linear_dim, 3 * num_out_points)

        self.bn_fc1 = nn.BatchNorm1d(linear_dim)
        self.bn_fc2 = nn.BatchNorm1d(linear_dim)
        self.bn_fc3 = nn.BatchNorm1d(linear_dim)

        # projection and matching
        self.project = SoftProjection(
            group_size, initial_temperature, is_temperature_trainable, min_sigma
        )
        self.skip_projection = skip_projection
        self.complete_fps = complete_fps    # true

        # input / output shapes
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if output_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "allowed shape are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        if input_shape != output_shape:
            warnings.warn("SampleNet: input_shape is different to output_shape.")
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor):
        # x shape should be B x 3 x N
        if self.input_shape == "bnc":
            x = x.permute(0, 2, 1)

        if x.shape[1] != 3:
            raise RuntimeError("shape of x must be of [Batch x 3 x NumInPoints]")

        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        y = F.relu(self.bn4(self.conv4(y)))
        y = F.relu(self.bn5(self.conv5(y)))  # Batch x 128 x NumInPoints

        # Max pooling for global feature vector:
        y = torch.max(y, 2)[0]  # Batch x 128

        y = F.relu(self.bn_fc1(self.fc1(y))) # 256
        y = F.relu(self.bn_fc2(self.fc2(y))) # 256
        # y = F.relu(self.bn_fc3(self.fc3(y))) # 256 有bug，y全变为0了
        y = self.fc3(y)
        y = self.bn_fc3(y)   # 经过BN之后全部为负数，所以relu后全0
        y = F.relu(y)

        y = self.fc4(y) # B,96

        y = y.view(-1, 3, self.num_out_points) # B,3,32

        # Simplified points
        simp = y
        proj = None

        # Projected points
        
        if not self.skip_projection:
            proj = self.project(point_cloud=x, query_cloud=y)
        else:
            proj = simp

        # Change to output shapes
        if self.output_shape == "bnc":    # "bnc" in training
            simp = simp.permute(0, 2, 1)
            if proj is not None:
                proj = proj.permute(0, 2, 1)
      
        # Assert contiguous tensors
        simp = simp.contiguous()
        if proj is not None:
            proj = proj.contiguous()

        out = proj 

        return simp, out        # B,3,32

    def sample(self, x):
        simp, proj, match, feat = self.__call__(x)
        return proj

    # Losses:
    # At inference time, there are no sampling losses.
    # When evaluating the model, we'd only want to asses the task loss.

    def get_simplification_loss(self, ref_pc, samp_pc, pc_size, gamma=1, delta=0):
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(ref_pc)
        # ref_pc and samp_pc are B x N x 3 matrices
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(samp_pc, ref_pc)
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss

    def get_projection_loss(self):
        sigma = self.project.sigma()
        # if self.skip_projection or not self.training:
        #     return torch.tensor(0).to(sigma)
        return sigma


if __name__ == "__main__":
    # point_cloud = np.random.randn(1, 3, 1024)
    # point_cloud_pl = torch.tensor(point_cloud, dtype=torch.float32).cuda()
    # net = SampleNet(5, 128, group_size=10, initial_temperature=0.1, complete_fps=True)

    # net.cuda()
    # net.eval()

    # for param in net.named_modules():
    #     print(param)

    # simp, proj, match = net.forward(point_cloud_pl)
    # simp = simp.detach().cpu().numpy()
    # proj = proj.detach().cpu().numpy()
    # match = match.detach().cpu().numpy()

    # print("*** SIMPLIFIED POINTS ***")
    # print(simp)
    # print("*** PROJECTED POINTS ***")
    # print(proj)
    # print("*** MATCHED POINTS ***")
    # print(match)

    # mse_points = np.sum((proj - match) ** 2, axis=1)
    # print("projected points vs. matched points error per point:")
    # print(mse_points)

    from thop import profile
    # model =SampleNet_nomatch(32, 128, group_size=7, initial_temperature=1.0, input_shape="bcn", output_shape="bcn",complete_fps=True,skip_projection=False).cuda()
    # input = torch.randn(1, 3, 1024)
    # model =PureMLPNet(8, 384, group_size=7, initial_temperature=1.0, input_shape="bcn", output_shape="bcn",complete_fps=True,skip_projection=True).cuda()
    model =SampleNet_nomatch_debug(512, 128, group_size=7, initial_temperature=1.0, input_shape="bcn", output_shape="bcn",
                                    skip_projection=True,linear_dim=128,conv_dim=128,conv_dim2=128).cuda()
    # Paper里 SampleNet的1024->512 FLOPs=167M，Params=0.46M，说是embedding维度为128
    # SampleNet 1024->512, 原模型 FLOPs=72.29M Params=0.595M
    # SampleNet 1024->32, 原模型 FLOPs=71.56M Params=0.225M (减少out point，让Params大幅减少，FLOPs不怎么变化)
    # SampleNet 1024->512, Conv和原文一样，Linear维度全部256，bottelsize=128 FLOPs=72.29M Params=0.595M
    # SampleNet 1024->512, Conv和原文一样，Linear维度全部128，bottelsize=128 FLOPs=71.667M Params=0.283M
    # SampleNet 1024->512, Conv维度全部128，Linear维度全部256，bottelsize=128,FLOPs=141.4M Params=0.629M
    # SampleNet 1024->512, Conv维度全部128，Linear维度全部128，bottelsize=128,FLOPs=140.74M Params=0.316M
    # SNet 1024->512, Conv维度全部128，Linear维度全部128，bottelsize=128,FLOPs=140.74M Params=0.316M  应该是SoftProjection的FLOPs和Params没被计算



    point_cloud = np.random.randn(1, 3, 1024)
    input = torch.tensor(point_cloud, dtype=torch.float32).cuda()
    model.train()
    # model(input)
    # print(input.shape)
    flops, params = profile(model, inputs=(input,))
    
       
    print("FLOPs=", str(2*flops/1000**2)+'M')
    print("params=", str(params/1000**2)+'M')
    # for param in model.named_modules():
    #     print(param)
    # for names, parameters in model.named_parameters():
    #     print(names,": ",parameters.shape)   # diffpool不保存BN参数且使用BN但不参与训练，samplenet保存BN参数