import pickle
from transformers import BertTokenizer, AutoTokenizer,BertModel,RobertaModel,RobertaTokenizer
import argparse
import os
import os.path
import numpy as np
import pandas as pd
import torch
_cached_tokenizer = None
_cached_model = None
_cached_model_path = None
def get_smiles_embedding(smiles, model_path="DeepChem/ChemBERTa-77M-MLM"):
    global _cached_tokenizer, _cached_model, _cached_model_path
    if _cached_tokenizer is not None and _cached_model is not None and _cached_model_path == model_path:
        tokenizer_ = _cached_tokenizer
        model_ = _cached_model
    else:
        print(f"正在加载模型: {model_path}")
        try:
            if os.path.exists(model_path) and os.path.isdir(model_path):
                print(f"使用本地模型: {model_path}")
                tokenizer_ = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                model_ = RobertaModel.from_pretrained(model_path, local_files_only=True)
            else:
                print(f"本地模型不存在，尝试在线下载: {model_path}")
                tokenizer_ = AutoTokenizer.from_pretrained(model_path)
                model_ = RobertaModel.from_pretrained(model_path)
            _cached_tokenizer = tokenizer_
            _cached_model = model_
            _cached_model_path = model_path
            print("模型加载成功！")
        except Exception as e:
            print(f"无法加载模型: {e}")
            print("请考虑手动下载模型或检查网络连接")
            raise e
    model_.eval()
    with torch.no_grad():
        outputs_ = model_(**tokenizer_(smiles, return_tensors='pt'))
    return outputs_.last_hidden_state[0][1:outputs_.last_hidden_state.shape[1]-1]
def generate_feature(args):
    data_path = args.root_data_path
    dataset = args.dataset
    model_path = getattr(args, 'model_path', "DeepChem/ChemBERTa-77M-MLM")
    output_data_path = data_path + '/' + dataset + '/compound/'
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
        print(f"{output_data_path} created")
    else:
        print(f"{output_data_path} exists")
    dict_list = {}
    count = 0
    opts = ['train', 'test']
    for o in opts:
        if dataset == 'davis':
            raw_data = pd.read_csv(f'{data_path}/{dataset}.csv')
        else:
            raw_data = pd.read_csv(f'{data_path}/{dataset}_{o}.csv')
        smiles_values = raw_data['compound_iso_smiles'].values
        for s in smiles_values:
          if s in dict_list.keys():
              continue
          smiles_embedding = get_smiles_embedding(s, model_path)
          dict_list[s] = smiles_embedding
          count += 1
        with open(f'{output_data_path}/mol_dict.pkl', 'wb') as file:
            pickle.dump(dict_list, file)
    print(count)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', type=str, default='../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='davis', help='Datasets')
    parser.add_argument('--model_path', type=str, default='DeepChem/ChemBERTa-77M-MLM',
                       help='Path to ChemBERTa model (local path or HuggingFace model name)')
    return parser.parse_args()
if __name__ == '__main__':
    params = parse_args()
    print(params)
    generate_feature(params)
