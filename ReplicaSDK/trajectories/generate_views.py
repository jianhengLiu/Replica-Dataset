# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [SHINEMapping] by Yue Pan.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def compute_obb(point_cloud):
    # 使用主成分分析（PCA）计算点云的旋转包围盒（OBB）
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    center = pca.mean_  # OBB的中心点
    axes = pca.components_  # OBB的主轴向量
    print(axes.shape)
    # calculate norm of axes
    print(np.linalg.norm(axes, axis=1))
    lengths = np.sqrt(pca.explained_variance_)  # OBB的长度
    # lengths = np.sqrt(pca.singular_values_)  # OBB的长度
    # lengths = np.max(point_cloud.dot(axes), axis=0) - np.min(
    #     point_cloud.dot(axes), axis=0
    # )  # OBB的长度
    obb = {
        "center": center,
        "axes": axes,
        "lengths": lengths,
    }
    return obb


import open3d as o3d


def compute_aabb(point_cloud):
    # 使用Open3D库计算轴对齐包围盒（AABB）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    aabb = pcd.get_axis_aligned_bounding_box()
    return aabb


# pip3 install pymeshlab
import pymeshlab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", help="folder containing the rgb images")
    # evaluation parameters
    parser.add_argument("--down_sample_vox", type=float, default=1.0)
    parser.add_argument("--boundary", type=float, default=0.0)
    args = parser.parse_args()

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load a new mesh in the MeshSet, and sets it as current mesh
    # the path of the mesh can be absolute or relative
    ms.load_new_mesh(args.mesh)

    # set the first mesh (id 0) as current mesh
    ms.set_current_mesh(0)

    m = ms.current_mesh()
    vertex = m.vertex_matrix()
    # downsampling
    vertex = vertex[::100, :]
    print(vertex.shape)
    obb = compute_obb(vertex)

    bbx = m.bounding_box()
    print(bbx.diagonal())

    bbox_min = bbx.min()
    bbox_max = bbx.max()

    bbox_size = bbox_max - bbox_min
    step_num = (bbox_size / args.down_sample_vox).astype(int)

    # evenly generate points in the bounding box
    x = np.linspace(bbox_min[0], bbox_max[0], step_num[0])
    y = np.linspace(bbox_min[1], bbox_max[1], step_num[1])
    z = bbox_min[2] + bbox_size[2] * 0.6
    pos = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    print(pos.shape)

    # pos should in obb's bounding box define by obb["center"] obb["axes"]
    # specifically, the pos to center vector, which dot with each axis should be less than each axis's length
    pos_vec = pos - obb["center"]
    print(pos_vec.shape)
    print(obb["axes"].shape)
    pos_proj = np.matmul(pos_vec, obb["axes"].T)
    # abs
    pos_proj = np.abs(pos_proj)
    print(pos_proj.shape)
    print(obb["lengths"].shape)
    pos_proj = pos_proj - obb["lengths"]
    # retain all pos_proj < -args.boundary
    mask = pos_proj < args.boundary
    mask = np.all(mask, axis=1)
    pos = pos[mask]
    print(pos.shape)
    
    
    # # 可视化旋转包围盒
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # # 绘制点云
    # ax.scatter(vertex[:, 0], vertex[:, 1], vertex[:, 2], c="blue", alpha=0.6)
    # # 绘制旋转包围盒
    # center = obb["center"]
    # axes = obb["axes"]
    # lengths = obb["lengths"]
    # for i in range(3):
    #     ax.quiver(
    #         center[0],
    #         center[1],
    #         center[2],
    #         axes[i][0] * lengths[i],
    #         axes[i][1] * lengths[i],
    #         axes[i][2] * lengths[i],
    #         color="red",
    #     )
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # # plot pos
    # ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", alpha=0.6)
    # plt.show()

    # evenly generate rotation matrix in 3D 360 degree per 45 degree
    # -z formward, y up
    rpy = np.array(
        [
            [0, 0, 0],
            [45, 0, 0],
            [90, 0, 0],
            [135, 0, 0],
            [180, 0, 0],
            [225, 0, 0],
            [270, 0, 0],
            [315, 0, 0],
            [0, 0, 45],
            [45, 0, 45],
            [90, 0, 45],
            [135, 0, 45],
            [180, 0, 45],
            [225, 0, 45],
            [270, 0, 45],
            [315, 0, 45],
            [0, 0, -45],
            [45, 0, -45],
            [90, 0, -45],
            [135, 0, -45],
            [180, 0, -45],
            [225, 0, -45],
            [270, 0, -45],
            [315, 0, -45],
            [0, 0, 90],
            [0, 0, -90],
        ]
    )
    # euler to rotation matrix
    from scipy.spatial.transform import Rotation as R

    # x y z w
    rot_0 = (
        R.from_euler("xzy", np.array([-90, 0, 0]), degrees=True)
        .as_matrix()
        .reshape(1, 3, 3)
    )
    rot = R.from_euler("yzx", rpy, degrees=True).as_matrix()
    rot = np.matmul(rot_0, rot)
    print(rot.shape)
    print(pos.shape)
    quat = R.from_matrix(rot).as_quat()

    # for each pos generate a pose compose by pos and rot
    poses = []
    for p in pos:
        for r in quat:
            poses.append(np.concatenate([p, r]))
    # export poses into "args.mesh.filename.txt"
    filename = Path(args.mesh).name.replace(".ply", ".txt")
    print(filename)
    print(len(poses))
    np.savetxt(filename, poses)
