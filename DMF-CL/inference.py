import copy
import time

import numpy as np
import pandas as pd

def print_parameter_stats(model):
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean()}, max={param.data.max()}, min={param.data.min()}")
def train2(model,predictor,device,train_loader,lr,epoch,
          batch_size,affinity_graph,drug_pos,target_pos,drug_fp,protein_pssm):
    #print('Training on {} samples for final train...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad,chain(model.parameters(),predictor.parameters())),lr=lr,weight_decay=0)
    
    """# 假设model和predictor已经被定义并初始化
    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    predictor_parameters = filter(lambda p: p.requires_grad,predictor.parameters())

    # 计算模型参数量
    model_params = sum(p.numel() for p in model_parameters)
    predictor_params = sum(p.numel() for p in predictor_parameters)

    # 计算总参数量
    total_params = model_params + predictor_params
    print(f"Total parameters: {total_params}")
    """
    drug_fp = drug_fp.detach()
    protein_pssm = protein_pssm.detach()
    #drug_graph_batchs = list(map(lambda graph: graph.to(device),drug_graphs_DataLoader))
    #target_graph_batchs = list(map(lambda graph: graph.to(device),target_graphs_DataLoader))
    
    for batch_idx,data in enumerate(train_loader):
        optimizer2.zero_grad()
        ssl_loss,drug_embedding,target_embedding = model(affinity_graph.to(device),drug_fp,protein_pssm,drug_pos,
                                                         target_pos,device)
       
        # Detach embeddings to avoid reusing graph
        #drug_embedding = drug_embedding.detach()
        #target_embedding = target_embedding.detach()
        output,_ = predictor(data.to(device),drug_embedding,target_embedding)
        #contrastive_loss = model2(drug_fp,protein_pssm,drug_graph,protein_contact_map)
        
        #ssl_loss = ssl_loss.detach()
        
        #print(f"Drug embedding requires grad: {drug_embedding.requires_grad}")
        #print(f"Target embedding requires grad: {target_embedding.requires_grad}")
        #print(f"SSL Loss requires grad: {ssl_loss.requires_grad}")
        #print(f"{batch_idx}batch模型参数---loss_final.backward()前")
        #print_parameter_stats(model)
        ssl_loss,drug_embedding,target_embedding,drug_yigou_embedding,drug_graph_embedding,target_yigou_embedding,target_graph_embedding = model(
            affinity_graph.to(device),drug_graph_batchs,target_graph_batchs,drug_pos,target_pos)
        output_yigou,_ = predictor(data.to(device),drug_yigou_embedding,target_yigou_embedding)
        output_graph,_ = predictor(data.to(device),drug_graph_embedding,target_graph_embedding)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        lajin = loss_fn(output_yigou,output_graph)
        loss_final = loss_fn(output, data.y.view(-1, 1).float().to(device)) + ssl_loss+lajin
        loss_final.backward()
        optimizer.step()
        #print(loss_final.item())
        loss_final.backward()
        #print(f"{batch_idx}batch模型参数---loss_final.backward()后")
        #print_parameter_stats(model)
        #print(f"{batch_idx}batch模型predictor参数---loss_final.backward()后")
        #print_parameter_stats(predictor)
        optimizer2.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx * batch_size,len(train_loader.dataset),100. * batch_idx / len(train_loader),
                loss_final.item()))
    """
    # 累积损失
    total_loss = 0
    total_samples = 0
    
    # 直接迭代整个数据集
    for data in train_data:
        optimizer2.zero_grad()
        ssl_loss, drug_embedding, target_embedding = model(affinity_graph.to(device), drug_fp, protein_pssm, drug_pos,
                                                           target_pos, device)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss_1 = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss_final=loss_1+ ssl_loss
        
        loss_final.backward()
        optimizer2.step()
        total_loss += loss_final.item()
        total_samples += 1
    
    # 计算平均损失
    average_loss = total_loss / total_samples
    print('Train epoch: {} Average Loss: {:.6f}'.format(epoch, average_loss))
"""

def test2(model,predictor,device,loader,affinity_graph,drug_pos,
         target_pos,drug_fp,protein_pssm):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples for final test...'.format(len(loader.dataset)))
    #drug_graph_batchs = list(map(lambda graph: graph.to(device),drug_graphs_DataLoader))  # drug graphs
    #target_graph_batchs = list(map(lambda graph: graph.to(device),target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            _,drug_embedding,target_embedding = model(affinity_graph.to(device),drug_fp,protein_pssm,
                                                      drug_pos,target_pos,device)
            output,_ = predictor(data.to(device),drug_embedding,target_embedding)
            total_preds = torch.cat((total_preds,output.cpu()),0)
            total_labels = torch.cat((total_labels,data.y.view(-1,1).cpu()),0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train(model,predictor,device,train_loader,drug_graphs_DataLoader,target_graphs_DataLoader,lr,epoch,
           batch_size):
    print('Training on {} samples...for affinity_graph.x'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,chain(model.parameters(),predictor.parameters())),lr=lr,weight_decay=0)
    """
    # 假设model和predictor已经被定义并初始化
    model_parameters = filter(lambda p: p.requires_grad,model.parameters())
    predictor_parameters = filter(lambda p: p.requires_grad,predictor.parameters())

    # 计算模型参数量
    model_params = sum(p.numel() for p in model_parameters)
    predictor_params = sum(p.numel() for p in predictor_parameters)

    # 计算总参数量
    total_params = model_params + predictor_params
    print(f"Total parameters: {total_params}")
    """
    drug_graph_batchs = list(map(lambda graph: graph.to(device),drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device),target_graphs_DataLoader))
    drug_embedding = []
    target_embedding = []
    best_loss=float('inf')
    best_drug_embedding = []
    best_target_embedding = []
    for batch_idx,data in enumerate(train_loader):
        optimizer.zero_grad()
        drug_embedding,target_embedding = model(drug_graph_batchs,target_graph_batchs)
        output,_ = predictor(data.to(device),drug_embedding,target_embedding)
        #contrastive_loss = model2(drug_fp,protein_pssm,drug_graph,protein_contact_map)
        loss = loss_fn(output,data.y.view(-1,1).float().to(device))
        loss.backward()
        optimizer.step()
        #print("模型参数----图特征提取时")
        #print_parameter_stats(model)
        if batch_idx==len(train_loader)-1:
            if loss<best_loss:
                best_loss = loss
                best_drug_embedding=drug_embedding
                best_target_embedding=target_embedding
        """
        if batch_idx < 10 or batch_idx == len(train_loader) - 1:
            # 假设 all_drug_embeddings 和 all_target_embeddings 是所有批次的嵌入的累积结果
            # 确保它们已经在 CPU 上并且是 NumPy 数组
            all_drug_embeddings = drug_embedding.detach().cpu().numpy()
            all_target_embeddings = target_embedding.detach().cpu().numpy()

            # 将 NumPy 数组转换为 pandas 数据帧
            drug_embeddings_df = pd.DataFrame(all_drug_embeddings)
            target_embeddings_df = pd.DataFrame(all_target_embeddings)

            # 保存数据帧为 CSV 文件
            drug_embeddings_df.to_csv(f'drug_embeddings_{epoch}_{batch_idx}.csv',index=False)
            target_embeddings_df.to_csv(f'target_embeddings_{epoch}_{batch_idx}.csv',index=False)
        """
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,batch_idx * batch_size,len(train_loader.dataset),100. * batch_idx / len(train_loader),
                loss.item()))

    return best_drug_embedding,best_target_embedding


def test(model,predictor,device,loader,drug_graphs_DataLoader,target_graphs_DataLoader):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples for affinity_graph.x...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device),drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device),target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            drug_embedding,target_embedding = model(drug_graph_batchs,target_graph_batchs)
            output,_ = predictor(data.to(device),drug_embedding,target_embedding)
            total_preds = torch.cat((total_preds,output.cpu()),0)
            total_labels = torch.cat((total_labels,data.y.view(-1,1).cpu()),0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


def train_predict():
    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)
    train_data,test_data,affinity_graph,drug_pos,target_pos = process_data(affinity_mat,args.dataset,args.num_pos,
                                                                           args.pos_threshold)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    print("device")
    print(device)
    drug_graphs_dict,drug_fp = get_drug_molecule_graph(
        json.load(open(f'data/{args.dataset}/drugs.txt'),object_pairs_hook=OrderedDict),device)
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict,dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data,shuffle=False,collate_fn=collate,
                                                         batch_size=affinity_graph.num_drug)
    target_graphs_dict,protein_pssm = get_target_molecule_graph(
        json.load(open(f'data/{args.dataset}/targets.txt'),object_pairs_hook=OrderedDict),args.dataset,device)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict,dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data,shuffle=False,collate_fn=collate,
                                                           batch_size=affinity_graph.num_target)

    print("Model preparation... ")


    model = HomFea(d_ms_dims=[78,78,78 * 2,256],
                    t_ms_dims=[54,54,54 * 2,256],embedding_dim=128)
    #model.eval()
    model.eval()
    predictor = PredictModule()

    model.to(device)
    predictor.to(device)
    drug_embeddings = []
    target_embeddings = []
    print("Start training for affinity_graph.x...")
    for epoch in range(args.epochs):
        drug_embeddings,target_embeddings = train(model,predictor,device,train_loader,drug_graphs_DataLoader,
                                                   target_graphs_DataLoader,args.lr,epoch + 1,
                                                   args.batch_size)
        #G, P = test2(model2, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader)
        """
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr, epoch+1,
              args.batch_size, affinity_graph, drug_pos, target_pos)
        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos, target_pos)
        """
        #r = model_evaluate(G, P)
        #print(r)

    # 合并 drug_embedding 和 target_embedding
    drug_embeddings = drug_embeddings.detach()
    target_embeddings = target_embeddings.detach()
    combined_features = torch.cat((drug_embeddings,target_embeddings),dim=0)

    # 更新 affinity_graph 的 x
    affinity_graph.x = combined_features
    print('\npredicting for train data for final train and test')
    model2 = DMFCL(tau=args.tau,
                    lam=args.lam,
                    ns_dims=[256,256,128],
                    drug_input_dim=128,
                    protein_input_dim=128,
                    device=device,
                    embedding_dim=128,
                    dropout_rate=args.edge_dropout_rate)
    model2.eval()
    predictor2 = PredictModule()
    predictor2.to(device)
    drug_pos = drug_pos.to(device)
    target_pos = target_pos.to(device)

    model2.to(device)
    for epoch in range(args.epochs2):
        start_time = time.time()
        train2(model2,predictor2,device,train_loader,args.lr,epoch + 1,
              args.batch_size,affinity_graph,drug_pos,target_pos,drug_fp,protein_pssm)
        G,P = test2(model2,predictor2,device,test_loader,
                   affinity_graph,drug_pos,target_pos,drug_fp,protein_pssm)

        r = model_evaluate(G,P)
        end_time = time.time()  # 记录本轮结束时间
        epoch_time = end_time - start_time  # 计算本轮耗时
        print(f"Epoch {epoch + 1}, Time taken: {epoch_time:.2f} seconds, {r}")
    #G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_pos, target_pos)
    #result = model_evaluate(G, P)
    #print("result:", result)


if __name__ == '__main__':
    import os
    import argparse
    import torch
    import json
    import warnings
    from collections import OrderedDict
    from torch import nn
    from itertools import chain
    from data_process import load_data,process_data,get_drug_molecule_graph,get_target_molecule_graph
    from utils import GraphDataset,collate,model_evaluate
    from models import DMFCL,PredictModule,HomFea

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--dataset',type=str,default='davis')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--epochs2',type=int,default=2000)
    #parser.add_argument('--epochs',type=int,default=2500)  # --kiba 3000
    parser.add_argument('--batch_size',type=int,default=512)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--edge_dropout_rate',type=float,default=0.1)  # --kiba 0.
    parser.add_argument('--tau',type=float,default=0.8)
    parser.add_argument('--lam',type=float,default=0.5)
    parser.add_argument('--num_pos',type=int,default=3)  # --kiba 10
    parser.add_argument('--pos_threshold',type=float,default=8.0)
    args,_ = parser.parse_known_args()
    # args = parser.parse_args()

    train_predict()
