import torch
import torch.nn as nn

from torch_geometric.nn import DenseGCNConv, GCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj

import torch
from torch import nn
from torch_geometric.nn import GATConv





class GCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(GCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = GCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))

        return embeddings


class GCNModel(nn.Module):
    def __init__(self, layers_dim):
        super(GCNModel, self).__init__()

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=list(range(self.num_layers)))

    def forward(self, graph_batchs):
        embedding_batchs = list(
                map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))
        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class DenseGCNBlock(nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            conv_layer = DenseGCNConv(gcn_layers_dim[i], gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj):
        output = x
        embeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))

        return embeddings


class DenseGCNModel(nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0.):
        super(DenseGCNModel, self).__init__()

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=list(range(self.num_layers)),
                                        dropout_layers_index=list(range(self.num_layers)))

    def forward(self, graph):
        xs, adj, num_d, num_t = graph.x, graph.adj, graph.num_drug, graph.num_target
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs],
                                                                p=self.edge_dropout_rate, force_undirected=True,
                                                                num_nodes=num_d + num_t, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout)

        return embeddings


class LinearBlock(nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0., relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = nn.Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(output)

        return embeddings


class Contrast(nn.Module):
    def __init__(self, hidden_dim, output_dim, tau, lam):
        super(Contrast, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim))
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)

        return sim_matrix

    def forward(self, za, zb, pos):
        za_proj = self.proj(za)
        zb_proj = self.proj(zb)
        matrix_a2b = self.sim(za_proj, zb_proj)
        matrix_b2a = matrix_a2b.t()

        matrix_a2b = matrix_a2b / (torch.sum(matrix_a2b, dim=1).view(-1, 1) + 1e-8)
        lori_a = -torch.log(matrix_a2b.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_b2a = matrix_b2a / (torch.sum(matrix_b2a, dim=1).view(-1, 1) + 1e-8)
        lori_b = -torch.log(matrix_b2a.mul(pos.to_dense()).sum(dim=-1)).mean()

        return self.lam * lori_a + (1 - self.lam) * lori_b, torch.cat((za_proj, zb_proj), 1)


class DMFCL(nn.Module):
    def __init__(self, tau, lam, ns_dims, drug_input_dim,protein_input_dim, device, embedding_dim=128, dropout_rate=0.2):
        super(DMFCL, self).__init__()

        self.output_dim = embedding_dim * 2
        new_dim=64
        self.affinity_graph_conv = DenseGCNModel(ns_dims, dropout_rate)
        #self.drug_graph_conv = GCNModel(d_ms_dims)
        #self.target_graph_conv = GCNModel(t_ms_dims)
        self.drug_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        self.target_contrast = Contrast(ns_dims[-1], embedding_dim, tau, lam)
        # Initialize MLPs for drug and protein sequence features
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_input_dim,new_dim).to(device),
            nn.ReLU(),
            nn.Linear(new_dim,embedding_dim).to(device)
        )

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_dim,new_dim).to(device),
            nn.ReLU(),
            nn.Linear(new_dim,embedding_dim).to(device)
        )
    def forward(self, affinity_graph, drug_fp, protein_pssm, drug_pos, target_pos,device):

        drug_fp=drug_fp.to(device)
        protein_pssm=protein_pssm.to(device)
        num_d = affinity_graph.num_drug
        #affinity_graph_embedding是异构图嵌入，drug_graph_embedding和target_graph_embedding分别是药物和蛋白质的同构图嵌入代码
        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        #drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        #target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]
#序列特征做映射
        drug_seq_embedding = self.drug_mlp(drug_fp)
        protein_seq_embedding = self.protein_mlp(protein_pssm)
        dru_loss, drug_embedding = self.drug_contrast(affinity_graph_embedding[:num_d], drug_seq_embedding, drug_pos)
        tar_loss, target_embedding = self.target_contrast(affinity_graph_embedding[num_d:], protein_seq_embedding,
                                                          target_pos)

        return dru_loss + tar_loss, drug_embedding, target_embedding
        # return drug_graph_embedding, target_graph_embedding
class HomFea(nn.Module):
    def __init__(self, d_ms_dims, t_ms_dims, embedding_dim=128):
        super(HomFea, self).__init__()
        self.output_dim = embedding_dim * 2
        self.drug_graph_conv = GCNModel(d_ms_dims)
        self.target_graph_conv = GCNModel(t_ms_dims)


    def forward(self, drug_graph_batchs, target_graph_batchs):
        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]
        return drug_graph_embedding, target_graph_embedding
        # return drug_graph_embedding, target_graph_embedding

class PredictModule(nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1):
        super(PredictModule, self).__init__()

        self.prediction_func, prediction_dim_func = (lambda x, y: torch.cat((x, y), -1), lambda dim: 4 * dim)
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):
        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)
        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings


class DrugProteinHeterograph(nn.Module):
    def __init__(self,drug_input_dim,protein_input_dim,embedding_dim=256):
        super(DrugProteinHeterograph,self).__init__()

        # Initialize MLPs for drug and protein sequence features
        self.drug_mlp = nn.Sequential(
            nn.Linear(drug_input_dim,embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim,embedding_dim)
        )

        self.protein_mlp = nn.Sequential(
            nn.Linear(protein_input_dim,embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim,embedding_dim)
        )

        # Graph attention layers for heterograph embeddings
        self.drug_gat = GATConv(embedding_dim,embedding_dim,heads=4,concat=False)
        self.protein_gat = GATConv(embedding_dim,embedding_dim,heads=4,concat=False)

        # Final layers for combined embeddings
        self.fc1 = nn.Linear(embedding_dim,embedding_dim)
        self.fc2 = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,drug_fp,protein_pssm,drug_graph,protein_contact_map):
        # Apply MLP on sequence features
        drug_seq_embedding = self.drug_mlp(drug_fp)
        protein_seq_embedding = self.protein_mlp(protein_pssm)

        # Apply GAT on molecular and contact graphs
        drug_graph_embedding = self.drug_gat(drug_graph.x,drug_graph.edge_index)
        protein_graph_embedding = self.protein_gat(protein_contact_map.x,protein_contact_map.edge_index)

        # Contrastive learning between MLP-mapped sequence and graph embeddings
        contrastive_loss = self.calculate_contrastive_loss(drug_seq_embedding,drug_graph_embedding) + \
                           self.calculate_contrastive_loss(protein_seq_embedding,protein_graph_embedding)

        return contrastive_loss

    def calculate_contrastive_loss(self,seq_embedding,graph_embedding,margin=1.0):
        # Example contrastive loss calculation (could use other metrics)
        loss = torch.nn.functional.mse_loss(seq_embedding,graph_embedding) + margin
        return loss