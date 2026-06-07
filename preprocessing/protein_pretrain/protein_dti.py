import os.path
import numpy as np
import pandas as pd
import torch
import esm
import argparse
import os
def get_pretrained_embedding(s):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", s)])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].cpu())
    del model, batch_tokens, token_representations, results
    torch.cuda.empty_cache()
    return sequence_representations[0]
def generate_feature(args):
    data_path = args.root_data_path
    dataset = args.dataset
    output_data_path = data_path + '/' + dataset + '/protein/'
    opts = ['train', 'test']
    for o in opts:
        sub_output_path = f'{output_data_path}/{o}/'
        if not os.path.exists(sub_output_path):
            os.makedirs(sub_output_path)
            print(f"{sub_output_path} created")
        else:
            print(f"{sub_output_path} exists")
        count = 0
        id_list = []
        raw_data = pd.read_csv(f'{data_path}/{dataset}_{o}.csv')
        sequence_values = raw_data['target_sequence'].values
        print(f"Processing {o} data: {len(sequence_values)} sequences")
        for i, s in enumerate(sequence_values):
            id = str(raw_data['target_id'][i])
            if id in id_list:
                print(f"Skipping duplicate ID: {id}")
                continue
            if os.path.isfile(f'{output_data_path}/{o}/' + id + '.npy'):
                print(f"File already exists, skipping: {id}")
                continue
            print(f"Processing sequence {i+1}/{len(sequence_values)}, ID: {id}")
            seq_emb = get_pretrained_embedding(s.upper())
            print(f"Embedding shape: {seq_emb.shape}")
            np.save(f'{output_data_path}/{o}/' + id, seq_emb)
            print(f"Saved: {id}")
            id_list.append(id)
            count += 1
        print(f"Processed {count} new sequences for {o}")
    print(f"Total new sequences processed: {count}")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_data_path', type=str, default='../../data', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='bindingdb', help='Datasets')
    return parser.parse_args()
if __name__ == '__main__':
    params = parse_args()
    print(f"Parameters: {params}")
    data_path = params.root_data_path
    dataset = params.dataset
    output_data_path = data_path + '/' + dataset + '/protein/'
    o = 'test'
    sub_output_path = f'{output_data_path}/{o}/'
    if not os.path.exists(sub_output_path):
        os.makedirs(sub_output_path)
        print(f"{sub_output_path} created")
    else:
        print(f"{sub_output_path} exists")
    count = 0
    id_list = []
    raw_data = pd.read_csv(f'{data_path}/{dataset}_{o}.csv')
    sequence_values = raw_data['target_sequence'].values
    print(f"Processing {o} data: {len(sequence_values)} sequences")
    for i, s in enumerate(sequence_values):
        id = str(raw_data['target_id'][i])
        if id in id_list:
            print(f"Skipping duplicate ID: {id}")
            continue
        if os.path.isfile(f'{output_data_path}/{o}/' + id + '.npy'):
            print(f"File already exists, skipping: {id}")
            continue
        print(f"Processing sequence {i+1}/{len(sequence_values)}, ID: {id}")
        seq_emb = get_pretrained_embedding(s.upper())
        print(f"Embedding shape: {seq_emb.shape}")
        np.save(f'{output_data_path}/{o}/' + id, seq_emb)
        print(f"Saved: {id}")
        id_list.append(id)
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} sequences so far...")
    print(f"Processed {count} new sequences for {o}")
    print("Finished!")
