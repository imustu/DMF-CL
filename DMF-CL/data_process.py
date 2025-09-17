#把同构图的特征加上
import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from utils import DTADataset, sparse_mx_to_torch_sparse_tensor, minMaxNormalize, denseAffinityRefine
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch import nn

"""
新加的序列代码，其中PSSM还是占位符
# 计算摩根指纹
def compute_morgan_fingerprint(smiles,radius=2,n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol,radius,nBits=n_bits),dtype=np.float32)
    else:
        return np.zeros((n_bits,),dtype=np.float32)


# 计算 PSSM 特征 (占位符)
def compute_pssm(sequence):
    # 生成 PSSM 特征的占位符
    return np.random.rand(100).astype(np.float32)  # 示例100维 PSSM 向量


# MLP 类，作为特征映射层
class MLP(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(MLP,self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.fc(x))


# 处理特征函数
def process_features(drug_smiles,protein_sequence,mlp_drug,mlp_protein):
    # 计算摩根指纹和 PSSM 特征，并映射为张量
    drug_fp = compute_morgan_fingerprint(drug_smiles)
    protein_pssm = compute_pssm(protein_sequence)

    # 转换为 PyTorch 张量
    drug_fp_tensor = torch.tensor(drug_fp,dtype=torch.float32)
    protein_pssm_tensor = torch.tensor(protein_pssm,dtype=torch.float32)

    # 使用 MLP 进行映射
    drug_embedding = mlp_drug(drug_fp_tensor)
    protein_embedding = mlp_protein(protein_pssm_tensor)

    return drug_embedding,protein_embedding
"""

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0

    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def load_data(dataset):
    affinity = pickle.load(open('data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)

    return affinity


def process_data(affinity_mat, dataset, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'

    train_file = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_index = []
    #len(train_file)=4.也就是说train_file中[[是有5份数据的，每份数据5009个。他文章写的是30056个数据，一共分成6份，其中5份是训练集，1份是测试集
    for i in range(len(train_file)):
        train_index += train_file[i]
    test_index = json.load(open(dataset_path + 'S1_test_set.txt'))

    rows, cols = np.where(np.isnan(affinity_mat) == False)
    train_rows, train_cols = rows[train_index], cols[train_index]
    train_Y = affinity_mat[train_rows, train_cols]
    #训练集包含药物ID、目标ID和亲和性分数
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)
    test_rows, test_cols = rows[test_index], cols[test_index]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity_mat = np.zeros_like(affinity_mat)
    train_affinity_mat[train_rows, train_cols] = train_Y
    #输入进get_affinity_graph的是train_affinity_mat，也就是训练集的亲和力部分，但是train_affinity_mat的行列结构是按照原始的affinity_mat（训练+测试）构建的，所以train_affinity_mat的行列数是全部数据，只不过其中的边关系是只包含训练集中的边
    affinity_graph, drug_pos, target_pos = get_affinity_graph(dataset, train_affinity_mat, num_pos, pos_threshold)

    return train_dataset, test_dataset, affinity_graph, drug_pos, target_pos


def get_affinity_graph(dataset, adj, num_pos, pos_threshold):
    dataset_path = 'data/' + dataset + '/'
    num_drug, num_target = adj.shape[0], adj.shape[1]
    # dt_是亲和力矩阵的复制
    dt_ = adj.copy()
    #将亲和性矩阵中的值根据阈值    pos_threshold    转换为0-1矩阵，表示是否存在相互作用
    dt_ = np.where(dt_ >= pos_threshold, 1.0, 0.0)
    dtd = np.matmul(dt_, dt_.T)
    dtd = dtd / dtd.sum(axis=-1).reshape(-1, 1)
    dtd = np.nan_to_num(dtd)
    dtd += np.eye(num_drug, num_drug)
    dtd = dtd.astype("float32")
    #dtd是基于网络结构的药物相似度，d_d是pubmed上的药物相似度
    d_d = np.loadtxt(dataset_path + 'drug-drug-sim.txt', delimiter=',')
    dAll = dtd + d_d
    drug_pos = np.zeros((num_drug, num_drug))
    for i in range(len(dAll)):
        one = dAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-dAll[i, one])
            sele = one[oo[:num_pos]]
            drug_pos[i, sele] = 1
        else:
            drug_pos[i, one] = 1
    drug_pos = sp.coo_matrix(drug_pos)
    drug_pos = sparse_mx_to_torch_sparse_tensor(drug_pos)
#以上是对于每个药物，根据相似度矩阵 dAll 选择前 num_pos 个最相似的药物。- 将这些位置在 drug_pos 矩阵中标记为1
    td_ = adj.T.copy()
    td_ = np.where(td_ >= pos_threshold, 1.0, 0.0)
    tdt = np.matmul(td_, td_.T)
    tdt = tdt / tdt.sum(axis=-1).reshape(-1, 1)
    tdt = np.nan_to_num(tdt)
    tdt += np.eye(num_target, num_target)
    tdt = tdt.astype("float32")
    t_t = np.loadtxt(dataset_path + 'target-target-sim.txt', delimiter=',')
    tAll = tdt + t_t
    target_pos = np.zeros((num_target, num_target))
    for i in range(len(tAll)):
        one = tAll[i].nonzero()[0]
        if len(one) > num_pos:
            oo = np.argsort(-tAll[i, one])
            sele = one[oo[:num_pos]]
            target_pos[i, sele] = 1
        else:
            target_pos[i, one] = 1
    target_pos = sp.coo_matrix(target_pos)
    target_pos = sparse_mx_to_torch_sparse_tensor(target_pos)
#处理davis数据集，将为5的地方-5，也就是置0了
#但这个处理是在蛋白质-药物异构图中，将边去掉用的
    if dataset == "davis":
        adj[adj != 0] -= 5
        # 对亲和性矩阵进行归一化
        adj_norm = minMaxNormalize(adj, 0)
#如果数据集是 "kiba"，则对亲和性矩阵进行精细化处理，以增强相似度。
    elif dataset == "kiba":
        adj_refine = denseAffinityRefine(adj.T, 150)
        adj_refine = denseAffinityRefine(adj_refine.T, 40)
        #对亲和性矩阵进行归一化
        adj_norm = minMaxNormalize(adj_refine, 0)
    adj_1 = adj_norm
    adj_2 = adj_norm.T
    #adj是药物-蛋白质二分网络，所以横纵数都是（蛋白质+药物）数量
    adj = np.concatenate((
        np.concatenate((np.zeros([num_drug, num_drug]), adj_1), 1),
        np.concatenate((adj_2, np.zeros([num_target, num_target])), 1)
    ), 0)
    train_row_ids, train_col_ids = np.where(adj != 0)
    edge_indexs = np.concatenate((
        np.expand_dims(train_row_ids, 0),
        np.expand_dims(train_col_ids, 0)
    ), 0)
    #亲和力值是边权重
    edge_weights = adj[train_row_ids, train_col_ids]
    #节点类型的特征 [1 0]是药物，[0 1]是蛋白质
    node_type_features = np.concatenate((
        np.tile(np.array([1, 0]), (num_drug, 1)),
        np.tile(np.array([0, 1]), (num_target, 1))
    ), 0)
    adj_features = np.zeros_like(adj)
    #做成0-1的
    adj_features[adj != 0] = 1
    #前两列代表节点类型（蛋白质还是药物），后边510列是邻接矩阵（0-1）
    features = np.concatenate((node_type_features, adj_features), 1)
    #异构图中features是节点特征，有510*512行是节点，列是特征
#改为全1向量
    graph_node_features = features.shape
    # 创建一个与 target_feature 形状相同且所有元素为1的数组
    temp = np.ones(graph_node_features)
    #affinity_graph = DATA.Data(x=torch.Tensor(features), adj=torch.Tensor(adj), edge_index=torch.LongTensor(edge_indexs))
    affinity_graph = DATA.Data(x=torch.Tensor(temp),adj=torch.Tensor(adj),edge_index=torch.LongTensor(edge_indexs))
    affinity_graph.__setitem__("edge_weight", torch.Tensor(edge_weights))
    affinity_graph.__setitem__("num_drug", num_drug)
    affinity_graph.__setitem__("num_target", num_target)

    return affinity_graph, drug_pos, target_pos


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """
    print("原子类型："+str(atom.GetSymbol()))
    print("原子度："+str(atom.GetDegree()))
    print("连接的氢原子数："+str(atom.GetTotalNumHs()))
    print("原子隐含价："+str(atom.GetImplicitValence()))
    print("原子芳香性:"+str(atom.GetIsAromatic()))
    """

    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

class AtomFeatureAggregator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AtomFeatureAggregator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, atomfeatures):
        # 假设 atomfeatures 是一个形状为 (num_atoms, feature_dim) 的张量
        # 其中 num_atoms 是药物中原子的数量，feature_dim 是每个原子的特征维度
        drug_features = atomfeatures.view(-1, atomfeatures.size(-1))  # 展平原子特征
        aggregated_features = self.fc(drug_features)  # 映射到统一维度
        return aggregated_features
def get_drug_molecule_graph(ligands,device):
    smile_graph = OrderedDict()
    all_drug_embeddings=[]
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        smile_graph[d] = smile_to_graph(lg)
        c_size,atomfeatures = smile_to_graph2(lg)
        aggregator = AtomFeatureAggregator(input_dim=78*c_size,output_dim=128).to(device)
        atomfeatures = atomfeatures.to(device)
        drug_embedding = aggregator(atomfeatures.float())
        all_drug_embeddings.append(drug_embedding)
    all_drug_embeddings = torch.cat(all_drug_embeddings,dim=0)
    return smile_graph,all_drug_embeddings

def smile_to_graph2(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features_tensor=[]
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        feature_tensor = torch.from_numpy(feature / sum(feature))
        features_tensor.append(feature_tensor)
    return c_size,torch.cat(features_tensor,dim=0)
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []

    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

#为了改成全1特征
    feature_length = len(features[0])
    # 创建一个新的特征向量，所有元素位置均为1
    temp_feature = np.ones(feature_length)
    # 创建 temp 列表，其结构与 features 相同，但所有元素位置均为1
    temp = [temp_feature.copy() for _ in features]

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
# 为了改成全1特征
    return c_size, features, edge_index
    #return c_size,temp,edge_index

"""
1.从 contact_dir 加载接触图文件：
    contact_dir 下的 .npy 文件包含的是每个蛋白质的氨基酸残基之间的接触图，也就是表示氨基酸残基之间是否存在相互作用或接触。这个接触图实际上是蛋白质的边（edges）信息，表示哪些氨基酸残基之间有边（关系）。
加载这个 .npy 文件后，代码通过 np.where(contact_map >= 0.5) 来筛选出相互作用关系的边，并将这些氨基酸残基之间的关系作为图的边（target_edge_index）。
2.从 aln_dir 提取氨基酸特征：
    aln_dir 目录下的比对文件提供了蛋白质的氨基酸特征。这些特征可以通过比对文件提取出每个氨基酸残基的保守性、物理化学特性、或者其它相关的序列信息。代码通过 target_to_feature 函数从这些比对文件中提取每个氨基酸的特征，最终得到每个蛋白质的节点特征（target_feature）。
这些特征代表了每个氨基酸残基的属性，可能包括序列保守性、演化信息等，这些信息将作为图中每个节点（氨基酸残基）的特征。
3.整合：
    图的节点和边的组合：
        节点：每个氨基酸残基作为图中的节点，节点的特征来自 aln 文件中的比对信息。
        边：节点之间的边来自接触图（contact_map），边表示氨基酸残基之间的物理接触或结构上的相互作用关系。
最终，蛋白质的图结构由两个部分组成：
    节点特征（target_feature），来自 aln 文件的比对特征。
    节点之间的边（target_edge_index），来自 contact_map 中的接触图
"""
def get_target_molecule_graph(proteins, dataset,device):
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'

    target_graph = OrderedDict()
    all_target_embeddings=[]

    for t in proteins.keys():
        g = target_to_graph(t, proteins[t], contac_path, msa_path)
        target_graph[t] = g
        target_size,targetfeature=target_to_graph2(t, proteins[t], msa_path)
        aggregator = AtomFeatureAggregator(input_dim=54 * target_size,output_dim=128).to(device)
        flattened_arr = targetfeature.reshape(-1)
        feature_tensor = torch.from_numpy(flattened_arr)
        feature_tensor=feature_tensor.to(device)
        target_embedding = aggregator(feature_tensor.float())
        all_target_embeddings.append(target_embedding)
    all_target_embeddings = torch.cat(all_target_embeddings,dim=0)
    return target_graph,all_target_embeddings


def target_to_graph2(target_key, target_sequence, aln_dir):
    target_size = len(target_sequence)
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    return target_size,target_feature
#靶点_>蛋白质分子图

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')

    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    # 为了改成全1特征
    target_shape = target_feature.shape
    # 创建一个与 target_feature 形状相同且所有元素为1的数组
    temp = np.ones(target_shape)


    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(target_size))
    index_row, index_col = np.where(contact_map >= 0.5)
    target_edge_index = []
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    # 为了改成全1特征
    #return target_size,temp,target_edge_index
    return target_size, target_feature, target_edge_index


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
#pssm.shape:(21, 氨基酸数量)   other_feature.shape:(氨基酸数量, 33)
    #other_feature是蛋白质的序列特征,有33列
    other_feature = seq_feature(pro_seq)


    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)

    return feature


def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat

    return pssm_mat


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]

    return np.array(res_property1 + res_property2)


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))

    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)




