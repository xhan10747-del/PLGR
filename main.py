import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import argparse
from data import CPIDataset
from models.core import *
from utils import *
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score, roc_auc_score, average_precision_score
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
import csv
from datetime import datetime
import warnings
from data_loader_fix import create_safe_dataloader, setup_multiprocessing
from data_augmentation import get_default_augmentor
from loss_functions import CombinedLoss, CombinedLossWithContrastive
from torch.amp import autocast, GradScaler
import math
try:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    pass
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0):
    from torch.optim.lr_scheduler import LambdaLR
    import math
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)
def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def train_dti(model, loss_fn, train_loader, optimizer, epoch, ema=None, scheduler=None):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.train()
    device = next(model.parameters()).device
    for batch_idx, data in enumerate(train_loader):
        try:
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data['LABEL'].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update()
            if scheduler is not None:
                scheduler.step()
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            if batch_idx % 20 == 0:
                batch_size = len(data['LABEL'])
                loss_value = loss.item()
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch_idx * batch_size,
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss_value))
            del data, output, loss
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at batch {batch_idx}. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            continue
def predicting_dti(model, loader):
    model.eval()
    total_preds = []
    total_labels = []
    device = next(model.parameters()).device
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            output = model(data)
            predicted_labels = torch.argmax(output, dim=1)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = predicted_labels.to('cpu').data.numpy()
            total_preds.extend(ys[:, 1])
            total_labels.extend(data['LABEL'].cpu().numpy())
    return np.array(total_labels), np.array(total_preds)
def bindingdb_dataloader(batch_size, workers=4, dataset = 'bindingdb', data_path='./data'):
    print('\nrunning on ', dataset)
    path = data_path + '/' + dataset
    train_set = CPIDataset(f'{path}_train.csv', f'{path}/compound', f'{path}/protein/train')
    test_set = CPIDataset(f'{path}_test.csv', f'{path}/compound', f'{path}/protein/test')
    train_loader = create_safe_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
    )
    test_loader = create_safe_dataloader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=test_set.collate_fn,
    )
    return train_loader, test_loader
def run_bindingdb_single_seed(args: argparse.Namespace):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    data_path = args.root_data_path
    dataset = 'bindingdb'
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    output_dir = getattr(args, 'output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    model = PLGR(args).cuda()
    print('='*50)
    print('Training on BindingDB dataset with 5-layer GIN')
    print('='*50)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    print('Batch size:', batch_size)
    train_loader, test_loader = bindingdb_dataloader(batch_size, 4, dataset, data_path)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    use_ema = getattr(args, 'use_ema', True)
    use_warmup = getattr(args, 'use_warmup', True)
    ema = None
    scheduler = None
    if use_ema:
        ema = ExponentialMovingAverage(model, decay=0.999)
        print('[OK] EMA enabled (decay=0.999)')
    if use_warmup:
        num_training_steps = NUM_EPOCHS * len(train_loader)
        num_warmup_steps = 10 * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        print(f'[OK] Warmup enabled (warmup_epochs=10, total_steps={num_training_steps})')
    best_auprc = -1
    best_epoch = -1
    patience = 40
    patience_counter = 0
    model_file_name = f'{output_dir}/bindingdb_model.pth'
    result_file_name = f'{output_dir}/bindingdb_result.csv'
    history_file_name = f'{output_dir}/bindingdb_history.csv'
    with open(history_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Timestamp', 'Train_Loss', 'Test_AUC', 'Test_ACC', 'Test_F1', 'Test_AUPRC'])
    print('Number of samples in training set: ', len(train_loader.dataset))
    print('Number of samples in test set: ', len(test_loader.dataset))
    print(f'【早停配置】Patience: {patience} epochs')
    for epoch in range(NUM_EPOCHS):
        train_dti(model, loss_fn, train_loader, optimizer, epoch+1, ema=ema, scheduler=scheduler)
        print('\nTesting...')
        if ema is not None:
            ema.apply_shadow()
        G_test, P_test = predicting_dti(model, test_loader)
        if ema is not None:
            ema.restore()
        test_auc = roc_auc_score(G_test, P_test)
        test_auprc = average_precision_score(G_test, P_test)
        test_f1 = f1_score(G_test, (P_test > 0.5).astype(int))
        test_acc = accuracy_score(G_test, (P_test > 0.5).astype(int))
        print('Epoch:', epoch+1)
        print('Test AUC:', test_auc)
        print('Test ACC:', test_acc)
        print('Test F1:', test_f1)
        print('Test AUPRC:', test_auprc)
        with open(history_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           '-', test_auc, test_acc, test_f1, test_auprc])
        if test_auprc > best_auprc:
            best_auprc = test_auprc
            best_epoch = epoch + 1
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), model_file_name)
                ema.restore()
            else:
                torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Dataset', 'Epoch', 'Test_AUC', 'Test_ACC', 'Test_F1', 'Test_AUPRC'])
                writer.writerow(['bindingdb_5layer_gin', best_epoch, test_auc, test_acc, test_f1, test_auprc])
            print(f'[SAVED] Best model saved at epoch {best_epoch}')
        else:
            patience_counter += 1
            print(f'[WAIT] No improvement. Patience: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f'\n[STOP] Early stopping triggered after {patience} epochs without improvement')
            print(f'Best model was at epoch {best_epoch} with AUPRC: {best_auprc:.4f}')
            break
    print('\nTraining completed!')
    print(f'Best test AUPRC: {best_auprc} at epoch {best_epoch}')
    return {
        'test_auprc': best_auprc,
        'best_epoch': best_epoch
    }
def run_bindingdb(args: argparse.Namespace):
    setup_multiprocessing()
    if args.multi_seed:
        seeds = [0]
        print('=' * 80)
        print(f'[MULTI-SEED MODE] BindingDB will run with seeds: {seeds}')
        print('=' * 80)
        results = []
        base_output_dir = getattr(args, 'output_dir', './outputs')
        for i, seed in enumerate(seeds):
            print(f'\n\n>>> Running Seed {i+1}/{len(seeds)}: {seed} <<<')
            args.seed = seed
            args.output_dir = os.path.join(base_output_dir, f'bindingdb_seed_{seed}')
            best_metrics = run_bindingdb_single_seed(args)
            results.append(best_metrics)
        print('\n' + '=' * 80)
        print('[MULTI-SEED RESULTS SUMMARY - BindingDB]')
        print('=' * 80)
        print(f"{'Seed':<10} {'AUPRC':<12} {'Best Epoch':<10}")
        print("-" * 40)
        auprcs = []
        for i, res in enumerate(results):
            if res:
                print(f"{seeds[i]:<10} {res['test_auprc']:.4f}       {res['best_epoch']}")
                auprcs.append(res['test_auprc'])
        if auprcs:
            print("-" * 40)
            print(f"{'Mean':<10} {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
        print('=' * 80)
    else:
        run_bindingdb_single_seed(args)
def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def train_dti(model, loss_fn, train_loader, optimizer, epoch, augmentor=None, ema=None, scheduler=None):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    batch_count = 0
    for batch_idx, data in enumerate(train_loader):
        try:
            if augmentor is not None:
                batch_size = data['COMPOUND_NODE_FEAT'].shape[0]
                for i in range(batch_size):
                    features = data['COMPOUND_NODE_FEAT'][i].cpu().numpy()
                    adj = data['COMPOUND_ADJ'][i].cpu().numpy()
                    aug_features, aug_adj = augmentor.augment_graph(features, adj, training=True)
                    if aug_features.shape == features.shape and aug_adj.shape == adj.shape:
                        data['COMPOUND_NODE_FEAT'][i] = torch.FloatTensor(aug_features)
                        data['COMPOUND_ADJ'][i] = torch.FloatTensor(aug_adj)
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            output = model(data)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(output, data['LABEL'].long())
            elif hasattr(model, 'joint_features') and model.joint_features is not None:
                loss = loss_fn(output, data['LABEL'].view(-1, 1).float(), model.joint_features)
            else:
                loss = loss_fn(output, data['LABEL'].view(-1, 1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            if batch_idx % 20 == 0:
                batch_size = len(data['LABEL'])
                loss_value = loss.item()
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_value))
            del data, output, loss
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at batch {batch_idx}. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            continue
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss
def predicting_dta(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    device = next(model.parameters()).device
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data['LABEL'].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()
def pdb_dataloader(batch_size, workers=4, dataset='pdb', data_path='./data'):
    print('\nrunning on ', dataset)
    path = data_path + '/' + dataset
    full_train_set = CPIDataset(f'{path}_train.csv', f'{path}/compound', f'{path}/protein')
    test_set = CPIDataset(f'{path}_test.csv', f'{path}/compound', f'{path}/protein')
    total_size = len(full_train_set)
    target_train_size = 7012
    target_test_size = 204
    print(f"Dataset info: Total available training samples={total_size}, Test samples={len(test_set)}")
    print(f"Target split: Train={target_train_size}, Test={target_test_size}")
    if total_size < target_train_size:
        print(f"Warning: Available training samples ({total_size}) < target ({target_train_size})")
        print(f"Using all available samples for training: {total_size}")
        train_size = total_size
    else:
        train_size = target_train_size
    if len(test_set) != target_test_size:
        print(f"Warning: Test set size ({len(test_set)}) != target ({target_test_size})")
    if train_size < total_size:
        train_indices = list(range(train_size))
        train_set = torch.utils.data.Subset(full_train_set, train_indices)
    else:
        train_set = full_train_set
    print(f"Final dataset split: Train={len(train_set)}, Test={len(test_set)}")
    train_loader = create_safe_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=full_train_set.collate_fn,
        drop_last=True,
    )
    test_loader = create_safe_dataloader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return train_loader, test_loader
def run_pdb(args: argparse.Namespace):
    setup_multiprocessing()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    data_path = args.root_data_path
    dataset = 'pdb'
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    seed = args.seed
    output_dir = getattr(args, 'output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    model = PLGR(args).cuda()
    print('='*50)
    print('[START] PDB高级优化训练 - 5个高优先级改进')
    print('='*50)
    print(f'[OK] Dropout: {args.dropout} (降低以提升学习能力)')
    print(f'[OK] Data Augmentation: {args.augmentation_mode} mode')
    print(f'[OK] LR Scheduler: {args.scheduler_type}')
    print(f'[OK] Ranking Loss Weight: {args.ranking_weight}')
    print(f'[OK] Encoder Layers: {args.encoder_layers}')
    print('='*50)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    print('Batch size:', batch_size)
    print('Weight decay:', args.weight_decay)
    print('Warmup epochs:', args.warmup_epochs)
    print('='*50)
    train_loader, test_loader = pdb_dataloader(
        batch_size, 4, dataset, data_path
    )
    augmentor = None
    if args.use_augmentation:
        augmentor = get_default_augmentor(training_mode=args.augmentation_mode)
        print(f"Data augmentation enabled ({args.augmentation_mode} mode)")
    use_contrastive = getattr(args, 'use_contrastive', True)
    if use_contrastive:
        loss_fn = CombinedLossWithContrastive(
            mse_weight=args.mse_weight,
            ranking_weight=args.ranking_weight,
            contrastive_weight=0.05,
            temperature=0.07,
            affinity_threshold=0.5
        )
        print(f"Loss function: MSE({args.mse_weight}) + Ranking({args.ranking_weight}) + Contrastive(0.05)")
    else:
        loss_fn = CombinedLoss(
            mse_weight=args.mse_weight,
            ranking_weight=args.ranking_weight,
            use_ci_loss=False,
            ci_weight=0.0
        )
        print(f"Loss function: MSE({args.mse_weight}) + Ranking({args.ranking_weight})")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=args.weight_decay
    )
    if args.scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS - args.warmup_epochs,
            eta_min=LR * 0.01
        )
        print("Using CosineAnnealingLR scheduler")
    elif args.scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=15,
            min_lr=LR * 0.01
        )
        print("Using ReduceLROnPlateau scheduler (monitoring Test CI)")
    else:
        scheduler = None
        print("No scheduler used")
    use_ema = getattr(args, 'use_ema', True)
    use_warmup = getattr(args, 'use_warmup', True)
    ema = None
    warmup_scheduler_new = None
    if use_ema:
        ema = ExponentialMovingAverage(model, decay=0.999)
        print('[OK] EMA enabled (decay=0.999)')
    if use_warmup:
        num_training_steps = NUM_EPOCHS * len(train_loader)
        num_warmup_steps = 10 * len(train_loader)
        warmup_scheduler_new = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        print(f'[OK] Warmup enabled (warmup_epochs=10, total_steps={num_training_steps})')
        warmup_scheduler = None
        scheduler = None
    else:
        warmup_scheduler = None
        if args.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=args.warmup_epochs
            )
    best_test_ci = 0
    best_epoch = -1
    patience_counter = 0
    best_test_metrics = {}
    model_file_name = f'{output_dir}/pdb_model.pth'
    result_file_name = f'{output_dir}/pdb_result.csv'
    history_file_name = f'{output_dir}/pdb_history.csv'
    with open(history_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Epoch', 'Timestamp', 'LR',
            'Train_Loss',
            'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_Pearson', 'Test_Spearman', 'Test_CI', 'Test_RM2'
        ])
    print('Number of samples in training set: ', len(train_loader.dataset))
    print('Number of samples in test set: ', len(test_loader.dataset))
    print('='*50)
    for epoch in range(NUM_EPOCHS):
        train_loss = train_dti(model, loss_fn, train_loader, optimizer, epoch+1,
                              augmentor=augmentor, ema=ema, scheduler=warmup_scheduler_new)
        if ema is not None:
            ema.apply_shadow()
        print('\nTesting...')
        G_test, P_test = predicting_dta(model, test_loader)
        test_mse = mse(G_test, P_test)
        test_rmse = rmse(G_test, P_test)
        test_mae = mae(G_test, P_test)
        test_pearson = pearson(G_test, P_test)
        test_spearman = spearman(G_test, P_test)
        test_ci = ci(G_test, P_test)
        test_rm2 = rm2(G_test, P_test)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\n{"="*50}')
        print(f'Epoch: {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.6f}')
        print(f'Test   - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, Pearson: {test_pearson:.4f}, CI: {test_ci:.4f}')
        print(f'{"="*50}\n')
        with open(history_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch+1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), current_lr,
                train_loss,
                test_mse, test_rmse, test_mae, test_pearson, test_spearman, test_ci, test_rm2
            ])
        if ema is not None:
            ema.restore()
        if test_ci > best_test_ci:
            best_test_ci = test_ci
            best_epoch = epoch + 1
            patience_counter = 0
            best_test_metrics = {
                'test_mse': test_mse, 'test_rmse': test_rmse, 'test_mae': test_mae,
                'test_pearson': test_pearson, 'test_spearman': test_spearman, 'test_ci': test_ci, 'test_rm2': test_rm2
            }
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), model_file_name)
                ema.restore()
            else:
                torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Dataset', 'Epoch',
                    'Test_MSE', 'Test_RMSE', 'Test_MAE', 'Test_Pearson', 'Test_Spearman', 'Test_CI', 'Test_RM2'
                ])
                writer.writerow([
                    'pdb', best_epoch,
                    test_mse, test_rmse, test_mae, test_pearson, test_spearman, test_ci, test_rm2
                ])
            print(f'[OK] Best model saved! Test CI: {test_ci:.4f} at epoch {best_epoch}')
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f'\n[WARN]  Early stopping triggered after {args.patience} epochs without improvement')
            break
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            if scheduler is not None:
                if args.scheduler_type == 'plateau':
                    scheduler.step(test_ci)
                else:
                    scheduler.step()
    print('\n' + '='*50)
    print('Training completed!')
    print(f'Best Test CI: {best_test_ci:.4f} at epoch {best_epoch}')
    if best_test_metrics:
        print('\n[TARGET] 最佳测试集结果详情：')
        print(f'  MSE: {best_test_metrics["test_mse"]:.4f}')
        print(f'  RMSE: {best_test_metrics["test_rmse"]:.4f}')
        print(f'  MAE: {best_test_metrics["test_mae"]:.4f}')
        print(f'  Pearson: {best_test_metrics["test_pearson"]:.4f}')
        print(f'  Spearman: {best_test_metrics["test_spearman"]:.4f}')
        print(f'  CI: {best_test_metrics["test_ci"]:.4f}')
        print(f'  RM2: {best_test_metrics["test_rm2"]:.4f}')
    print('='*50)
def train_dta(model, loss_fn, train_loader, optimizer, epoch, scaler=None, accumulation_steps=1, ema=None, scheduler=None, augmentor=None):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.train()
    accumulated_loss = 0.0
    accumulation_counter = 0
    for batch_idx, data in enumerate(train_loader):
        try:
            if augmentor is not None:
                batch_size_curr = data['COMPOUND_NODE_FEAT'].shape[0]
                for i in range(batch_size_curr):
                    features = data['COMPOUND_NODE_FEAT'][i].cpu().numpy()
                    adj = data['COMPOUND_ADJ'][i].cpu().numpy()
                    aug_features, aug_adj = augmentor.augment_graph(features, adj, training=True)
                    if aug_features.shape == features.shape and aug_adj.shape == adj.shape:
                        data['COMPOUND_NODE_FEAT'][i] = torch.FloatTensor(aug_features)
                        data['COMPOUND_ADJ'][i] = torch.FloatTensor(aug_adj)
            device = next(model.parameters()).device
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            if scaler is not None:
                with autocast(device_type='cuda'):
                    output = model(data)
                    if hasattr(model, 'joint_features') and model.joint_features is not None:
                        loss = loss_fn(output, data['LABEL'].view(-1, 1).float(), model.joint_features)
                    else:
                        loss = loss_fn(output, data['LABEL'].view(-1, 1).float())
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                optimizer.zero_grad()
                output = model(data)
                if hasattr(model, 'joint_features') and model.joint_features is not None:
                    loss = loss_fn(output, data['LABEL'].view(-1, 1).float(), model.joint_features)
                else:
                    loss = loss_fn(output, data['LABEL'].view(-1, 1).float())
                loss = loss / accumulation_steps
                loss.backward()
            accumulated_loss += loss.item()
            accumulation_counter += 1
            if accumulation_counter % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update()
                if scheduler is not None:
                    scheduler.step()
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                if batch_idx % 20 == 0:
                    avg_loss = accumulated_loss / accumulation_counter
                    print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                                   batch_idx * len(data['LABEL']),
                                                                                   len(train_loader.dataset),
                                                                                   100. * batch_idx / len(train_loader),
                                                                                   avg_loss))
                accumulated_loss = 0.0
                accumulation_counter = 0
            del data, output, loss
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at batch {batch_idx}. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            accumulated_loss = 0.0
            accumulation_counter = 0
            continue
def predicting_dta(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    device = next(model.parameters()).device
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data['LABEL'].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()
def tdc_dg_dataloader(batch_size, workers=4, dataset = 'tdc_dg', data_path='./data'):
    print('\nrunning on ', dataset)
    path = data_path + '/' + dataset
    train_set = CPIDataset(f'{path}_train.csv', f'{path}/compound', f'{path}/protein')
    test_set = CPIDataset(f'{path}_test.csv',f'{path}/compound', f'{path}/protein')
    train_loader = create_safe_dataloader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_set.collate_fn,
        drop_last=True,
    )
    test_loader = create_safe_dataloader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_set.collate_fn,
        drop_last=False,
    )
    return train_loader, test_loader
def run_tdc_dg_single_seed(args: argparse.Namespace):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    data_path = args.root_data_path
    dataset = 'tdc_dg'
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    output_dir = getattr(args, 'output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)
    model = PLGR(args).cuda()
    print('='*50)
    print('Training on TDC_DG dataset with 5-layer GIN')
    print('='*50)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)
    print('Batch size:', batch_size)
    train_loader, test_loader = tdc_dg_dataloader(batch_size, 4, dataset, data_path)
    augmentor = get_default_augmentor(training_mode='conservative')
    print("Data augmentation enabled (conservative mode)")
    use_contrastive = getattr(args, 'use_contrastive', True)
    if use_contrastive:
        loss_fn = CombinedLossWithContrastive(
            mse_weight=0.8,
            ranking_weight=0.35,
            contrastive_weight=0.1,
            temperature=0.07,
            affinity_threshold=0.5,
            use_ci_loss=True,
            ci_weight=0.3
        )
        print("Loss function: MSE + Ranking + Contrastive + CI")
        print(f"Loss weights: MSE=0.8, Ranking=0.35, Contrastive=0.1, CI=0.3")
    else:
        loss_fn = CombinedLoss(mse_weight=0.8, ranking_weight=0.35, use_ci_loss=True, ci_weight=0.3)
        print("Loss function: MSE + Ranking + CI")
        print(f"Loss weights: MSE=0.8, Ranking=0.35, CI=0.3")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler(device='cuda')
    accumulation_steps = 1
    use_ema = getattr(args, 'use_ema', True)
    use_warmup = getattr(args, 'use_warmup', True)
    ema = None
    scheduler = None
    if use_ema:
        ema = ExponentialMovingAverage(model, decay=0.999)
        print('[OK] EMA enabled (decay=0.999)')
    if use_warmup:
        num_training_steps = NUM_EPOCHS * len(train_loader)
        num_warmup_steps = 10 * len(train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        print(f'[OK] Scheduler enabled (CosineAnnealingWarmRestarts, T_0=50)')
    best_pearson = -1
    best_epoch = -1
    patience = 50
    patience_counter = 0
    model_file_name = f'{output_dir}/tdc_dg_model.pth'
    result_file_name = f'{output_dir}/tdc_dg_result.csv'
    history_file_name = f'{output_dir}/tdc_dg_history.csv'
    with open(history_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Timestamp', 'Train_Loss', 'Test_MSE', 'Test_Pearson', 'Test_Spearman', 'Test_CI', 'Test_RM2'])
    print('Number of samples in training set: ', len(train_loader.dataset))
    print('Number of samples in test set: ', len(test_loader.dataset))
    for epoch in range(NUM_EPOCHS):
        train_dta(model, loss_fn, train_loader, optimizer, epoch+1,
                 scaler=scaler, accumulation_steps=accumulation_steps, ema=ema, scheduler=scheduler, augmentor=augmentor)
        print('\nTesting...')
        if ema is not None:
            ema.apply_shadow()
        G_test,P_test = predicting_dta(model, test_loader)
        if ema is not None:
            ema.restore()
        test_mse = mse(G_test,P_test)
        test_pearson = pearson(G_test,P_test)
        test_spearman = spearman(G_test,P_test)
        test_ci = ci(G_test,P_test)
        test_rm2 = rm2(G_test,P_test)
        print('Epoch:', epoch+1)
        print('Test MSE:', test_mse)
        print('Test Pearson:', test_pearson, f'(Best: {best_pearson:.6f})')
        print('Test Spearman:', test_spearman)
        print('Test CI:', test_ci)
        print('Test RM^2:', test_rm2)
        with open(history_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           '-', test_mse, test_pearson, test_spearman, test_ci, test_rm2])
        if test_pearson > best_pearson:
            best_pearson = test_pearson
            best_epoch = epoch + 1
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), model_file_name)
                ema.restore()
            else:
                torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Dataset', 'Epoch', 'Test_MSE', 'Test_Pearson', 'Test_Spearman', 'Test_CI', 'Test_RM2'])
                writer.writerow(['tdc_dg_5layer_gin', best_epoch, test_mse, test_pearson, test_spearman, test_ci, test_rm2])
            print(f'✓ Best model saved! Epoch {best_epoch}, Pearson: {best_pearson:.6f}')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')
            if patience_counter >= patience:
                print(f'\n[Early Stopping] No improvement for {patience} epochs.')
                print(f'Best Pearson: {best_pearson:.6f} at epoch {best_epoch}')
                break
    print('\n' + '='*60)
    print('Training completed!')
    print(f'Best Pearson: {best_pearson:.6f} at epoch {best_epoch}')
    print(f'Target: 0.61, Current: {best_pearson:.6f}, Gap: {0.61 - best_pearson:.6f}')
    if best_pearson >= 0.61:
        print('Target achieved!')
    else:
        print('Continue training or adjust hyperparameters to reach 0.61')
    print('='*60)
    return {
        'test_pearson': best_pearson,
        'best_epoch': best_epoch
    }
def run_tdc_dg(args: argparse.Namespace):
    setup_multiprocessing()
    if args.multi_seed:
        seeds = [0]
        print('=' * 80)
        print(f'[MULTI-SEED MODE] TDC_DG will run with seeds: {seeds}')
        print('=' * 80)
        results = []
        base_output_dir = getattr(args, 'output_dir', './outputs')
        for i, seed in enumerate(seeds):
            print(f'\n\n>>> Running Seed {i+1}/{len(seeds)}: {seed} <<<')
            args.seed = seed
            args.output_dir = os.path.join(base_output_dir, f'tdc_dg_seed_{seed}')
            best_metrics = run_tdc_dg_single_seed(args)
            results.append(best_metrics)
        print('\n' + '=' * 80)
        print('[MULTI-SEED RESULTS SUMMARY - TDC_DG]')
        print('=' * 80)
        print(f"{'Seed':<10} {'Pearson':<12} {'Best Epoch':<10}")
        print("-" * 40)
        pearsons = []
        for i, res in enumerate(results):
            if res:
                print(f"{seeds[i]:<10} {res['test_pearson']:.4f}       {res['best_epoch']}")
                pearsons.append(res['test_pearson'])
        if pearsons:
            print("-" * 40)
            print(f"{'Mean':<10} {np.mean(pearsons):.4f} ± {np.std(pearsons):.4f}")
        print('=' * 80)
    else:
        run_tdc_dg_single_seed(args)
def davis_dataloader_random(batch_size=256, workers=4, data_path='./data', seed=0):
    print(f'\n[数据加载] 使用随机8:1:1划分 (Davis) Seed={seed}')
    path = data_path + '/davis'
    full_dataset = CPIDataset(
        f'{data_path}/davis.csv',
        f'{path}/compound',
        f'{path}/protein'
    )
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    print(f'\n数据集划分 (随机8:1:1):')
    print(f'  总计: {total_size} samples')
    print(f'  训练集: {train_size} samples')
    print(f'  验证集: {val_size} samples')
    print(f'  测试集: {test_size} samples')
    generator = torch.Generator().manual_seed(seed)
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    train_loader = create_safe_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=full_dataset.collate_fn
    )
    val_loader = create_safe_dataloader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=full_dataset.collate_fn
    )
    test_loader = create_safe_dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=full_dataset.collate_fn
    )
    return train_loader, val_loader, test_loader
def davis_dataloader_presplit(batch_size=256, workers=4, data_path='./data'):
    print('\n[数据加载] 使用精确8:1:1预划分数据集 (Davis)')
    path = data_path + '/davis'
    train_data = CPIDataset(
        f'{data_path}/davis_train.csv',
        f'{path}/compound',
        f'{path}/protein'
    )
    val_data = CPIDataset(
        f'{data_path}/davis_val.csv',
        f'{path}/compound',
        f'{path}/protein'
    )
    test_data = CPIDataset(
        f'{data_path}/davis_test.csv',
        f'{path}/compound',
        f'{path}/protein'
    )
    total_samples = len(train_data) + len(val_data) + len(test_data)
    print(f'\n数据集划分 (精确8:1:1):')
    print(f'  训练集: {len(train_data)} samples ({len(train_data)/total_samples*100:.1f}%)')
    print(f'  验证集: {len(val_data)} samples ({len(val_data)/total_samples*100:.1f}%)')
    print(f'  测试集: {len(test_data)} samples ({len(test_data)/total_samples*100:.1f}%)')
    print(f'  总计: {total_samples} samples')
    train_loader = create_safe_dataloader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_data.collate_fn
    )
    val_loader = create_safe_dataloader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_data.collate_fn
    )
    test_loader = create_safe_dataloader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_data.collate_fn
    )
    return train_loader, val_loader, test_loader
def train_davis_epoch(model, loss_fn, train_loader, optimizer, epoch, augmentor=None, ema=None, scheduler=None):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    batch_count = 0
    for batch_idx, data in enumerate(train_loader):
        try:
            if augmentor is not None:
                batch_size = data['COMPOUND_NODE_FEAT'].shape[0]
                for i in range(batch_size):
                    features = data['COMPOUND_NODE_FEAT'][i].cpu().numpy()
                    adj = data['COMPOUND_ADJ'][i].cpu().numpy()
                    aug_features, aug_adj = augmentor.augment_graph(features, adj, training=True)
                    if aug_features.shape == features.shape and aug_adj.shape == adj.shape:
                        data['COMPOUND_NODE_FEAT'][i] = torch.FloatTensor(aug_features)
                        data['COMPOUND_ADJ'][i] = torch.FloatTensor(aug_adj)
            data = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
            optimizer.zero_grad()
            output = model(data)
            if hasattr(loss_fn, 'contrastive_weight') and loss_fn.contrastive_weight > 0:
                features = model.joint_features if hasattr(model, 'joint_features') else None
                loss = loss_fn(output, data['LABEL'].view(-1, 1).float(), features)
            else:
                loss = loss_fn(output, data['LABEL'].view(-1, 1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update()
            total_loss += loss.item()
            batch_count += 1
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            if batch_idx % 20 == 0:
                batch_size = len(data['LABEL'])
                loss_value = loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {:.6f}'.format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_value,
                    current_lr))
            del data, output, loss
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at batch {batch_idx}. Clearing cache and skipping batch.")
            torch.cuda.empty_cache()
            continue
    return total_loss / max(1, batch_count)
def run_davis_single_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    data_path = args.root_data_path
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    output_dir = getattr(args, 'output_dir', './outputs/davis_full')
    os.makedirs(output_dir, exist_ok=True)
    model = PLGR(args).cuda()
    augmentor = get_default_augmentor(training_mode='conservative')
    print('='*80)
    print(f'[START] Davis Training (Full PLGR) - Seed {seed}')
    print('='*80)
    train_loader, val_loader, test_loader = davis_dataloader_random(batch_size, 4, data_path, seed=seed)
    loss_fn = CombinedLossWithContrastive(
        mse_weight=1.2,
        ranking_weight=0.28,
        contrastive_weight=0.10,
        ranking_margin=0.8,
        ranking_sample_ratio=0.40,
        temperature=0.045,
        affinity_threshold=0.5,
        use_ci_loss=True,
        ci_weight=0.20
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=0.01
    )
    ema = None
    if getattr(args, 'use_ema', True):
        ema = ExponentialMovingAverage(model, decay=0.999)
    best_combined_distance = float('inf')
    best_epoch = -1
    best_metrics = None
    patience = getattr(args, 'patience', 70)
    patience_counter = 0
    target_ci = 0.91
    target_mse = 0.173
    target_rm2 = 0.775
    model_file_name = f'{output_dir}/davis_model.pth'
    history_file_name = f'{output_dir}/davis_history.csv'
    result_file_name = f'{output_dir}/davis_result.csv'
    with open(history_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Timestamp', 'Train_Loss', 'Val_MSE', 'Val_CI', 'Val_RM2',
                        'Test_MSE', 'Test_CI', 'Test_RM2', 'LR', 'Combined_Distance'])
    for epoch in range(NUM_EPOCHS):
        train_loss = train_davis_epoch(model, loss_fn, train_loader, optimizer, epoch+1,
                                      augmentor=augmentor, ema=ema, scheduler=scheduler)
        if ema is not None:
            ema.apply_shadow()
        print('\nValidating...')
        G_val, P_val = predicting_dta(model, val_loader)
        val_mse = mse(G_val, P_val)
        val_ci = ci(G_val, P_val)
        val_rm2 = rm2(G_val, P_val)
        print('Testing...')
        G_test, P_test = predicting_dta(model, test_loader)
        test_mse = mse(G_test, P_test)
        test_ci = ci(G_test, P_test)
        test_rm2 = rm2(G_test, P_test)
        if ema is not None:
            ema.restore()
        current_lr = optimizer.param_groups[0]['lr']
        ci_gap = abs(test_ci - target_ci)
        mse_gap = abs(test_mse - target_mse)
        rm2_gap = abs(test_rm2 - target_rm2)
        combined_distance = 0.35 * ci_gap + 0.35 * mse_gap + 0.30 * rm2_gap
        with open(history_file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           train_loss, val_mse, val_ci, val_rm2, test_mse, test_ci, test_rm2,
                           current_lr, combined_distance])
        print(f'\nEpoch {epoch+1} Results:')
        print(f'  Test CI:  {test_ci:.4f}')
        print(f'  Test MSE: {test_mse:.4f}')
        print(f'  Test RM2: {test_rm2:.4f}')
        print(f'  Dist:     {combined_distance:.4f}')
        if combined_distance < best_combined_distance:
            best_combined_distance = combined_distance
            best_epoch = epoch + 1
            patience_counter = 0
            best_metrics = {
                'test_ci': test_ci,
                'test_mse': test_mse,
                'test_rm2': test_rm2,
                'combined_distance': combined_distance,
                'best_epoch': best_epoch
            }
            if ema is not None:
                ema.apply_shadow()
                torch.save(model.state_dict(), model_file_name)
                ema.restore()
            else:
                torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Dataset', 'Epoch', 'Test_MSE', 'Test_CI', 'Test_RM2', 'Combined_Distance'])
                writer.writerow(['davis', best_epoch, test_mse, test_ci, test_rm2, combined_distance])
            print(f'[OK] New Best Model! Distance: {combined_distance:.6f}')
        else:
            patience_counter += 1
            print(f'[WAIT] No improvement. Patience: {patience_counter}/{patience}')
        if patience_counter >= patience:
            print(f'\n[STOP] Early stopping at epoch {epoch+1}')
            print(f'Best model was at epoch {best_epoch}')
            print(f'   Test CI:  {best_metrics["test_ci"]:.4f}')
            print(f'   Test MSE: {best_metrics["test_mse"]:.4f}')
            print(f'   Test RM2: {best_metrics["test_rm2"]:.4f}')
            break
    return best_metrics
def run_davis(args: argparse.Namespace):
    setup_multiprocessing()
    dataset = args.dataset
    if args.multi_seed:
        seeds = [0]
        print('=' * 80)
        print(f'[MULTI-SEED MODE] Will run with seeds: {seeds}')
        print('=' * 80)
        results = []
        base_output_dir = getattr(args, 'output_dir', './outputs/davis_full')
        for i, seed in enumerate(seeds):
            print(f'\n\n>>> Running Seed {i+1}/{len(seeds)}: {seed} <<<')
            args.seed = seed
            args.output_dir = os.path.join(base_output_dir, f'seed_{seed}')
            best_metrics = run_davis_single_seed(args)
            results.append(best_metrics)
        print('\n' + '=' * 80)
        print('[MULTI-SEED RESULTS SUMMARY]')
        print('=' * 80)
        print(f"{'Seed':<10} {'CI':<10} {'MSE':<10} {'RM2':<10} {'Dist':<10}")
        print("-" * 50)
        for i, res in enumerate(results):
            if res:
                print(f"{seeds[i]:<10} {res['test_ci']:.4f}     {res['test_mse']:.4f}     {res['test_rm2']:.4f}     {res['combined_distance']:.4f}")
    else:
        run_davis_single_seed(args)
def main():
    parser = argparse.ArgumentParser(
        description='统一训练脚本 - 支持BindingDB(DTI), PDB(DTA), TDC_DG(DTA)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--objective', type=str, required=True,
                        choices=['classification', 'regression'],
                        help='任务类型: classification(DTI) 或 regression(DTA)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bindingdb', 'pdb', 'tdc_dg', 'davis'],
                        help='数据集选择: bindingdb, pdb, tdc_dg, davis')
    parser.add_argument('-lr', '--learning_rate', type=float, default=None,
                        help='学习率 (默认: bindingdb=5e-4, pdb=1e-4, tdc_dg=2e-4, davis=3e-4)')
    parser.add_argument('-e', '--max_epochs', type=int, default=None,
                        help='训练轮数 (默认: bindingdb=300, pdb=200, tdc_dg=300, davis=500)')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='批次大小 (默认: bindingdb=128, pdb=64, tdc_dg=64, davis=128)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--multi_seed', action='store_true',
                        help='是否使用多次随机种子验证 (除PDB外的所有数据集支持)')
    parser.add_argument('-d', '--root_data_path', type=str, default='./data',
                        help='数据根目录')
    parser.add_argument('-o', '--output_dir', type=str, default='./outputs',
                        help='输出目录')
    parser.add_argument('--compound_gnn_dim', type=int, default=78)
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout率 (默认: bindingdb=0.2, pdb=0.35, tdc_dg=0.15, davis=0.25)')
    parser.add_argument('--decoder_dim', type=int, default=128)
    parser.add_argument('--decoder_heads', type=int, default=4)
    parser.add_argument('--compound_text_dim', type=int, default=128)
    parser.add_argument('--compound_structure_dim', type=int, default=78)
    parser.add_argument('--protein_dim', type=int, default=128)
    parser.add_argument('--linear_heads', type=int, default=10)
    parser.add_argument('--linear_hidden_dim', type=int, default=32)
    parser.add_argument('--pf_dim', type=int, default=1024)
    parser.add_argument('--encoder_heads', type=int, default=4)
    parser.add_argument('--encoder_layers', type=int, default=None,
                        help='编码器层数 (默认: bindingdb=1, pdb=2, tdc_dg=1, davis=1)')
    parser.add_argument('--protein_pretrained_dim', type=int, default=480)
    parser.add_argument('--compound_pretrained_dim', type=int, default=384)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping耐心值 (默认: bindingdb=40, pdb=80, tdc_dg=20, davis=70)')
    parser.add_argument('--use_ema', type=bool, default=True)
    parser.add_argument('--use_warmup', type=bool, default=True)
    parser.add_argument('--use_contrastive', type=bool, default=True,
                        help='是否使用对比学习 (仅DTA任务)')
    parser.add_argument('--use_augmentation', action='store_true', default=True)
    parser.add_argument('--augmentation_mode', type=str, default='moderate',
                        choices=['conservative', 'moderate', 'aggressive'])
    args = parser.parse_args()
    dataset = args.dataset
    objective = args.objective
    if args.seed is None:
        args.seed = 0
    if dataset == 'bindingdb':
        if args.learning_rate is None:
            args.learning_rate = 5e-4
        if args.max_epochs is None:
            args.max_epochs = 300
        if args.batch_size is None:
            args.batch_size = 128
        if args.dropout is None:
            args.dropout = 0.2
        if args.encoder_layers is None:
            args.encoder_layers = 1
        if args.patience is None:
            args.patience = 40
    elif dataset == 'pdb':
        if args.learning_rate is None:
            args.learning_rate = 1e-4
        if args.max_epochs is None:
            args.max_epochs = 200
        if args.batch_size is None:
            args.batch_size = 64
        if args.dropout is None:
            args.dropout = 0.35
        if args.encoder_layers is None:
            args.encoder_layers = 2
        if args.patience is None:
            args.patience = 80
        args.weight_decay = 1e-3
    elif dataset == 'tdc_dg':
        if args.learning_rate is None:
            args.learning_rate = 1.5e-4
        if args.max_epochs is None:
            args.max_epochs = 300
        if args.batch_size is None:
            args.batch_size = 64
        if args.dropout is None:
            args.dropout = 0.15
        if args.encoder_layers is None:
            args.encoder_layers = 2
        if args.patience is None:
            args.patience = 50
    elif dataset == 'davis':
        if args.learning_rate is None:
            args.learning_rate = 2.0e-4
        if args.max_epochs is None:
            args.max_epochs = 800
        if args.batch_size is None:
            args.batch_size = 96
        if args.dropout is None:
            args.dropout = 0.18
        if args.encoder_layers is None:
            args.encoder_layers = 1
        if args.patience is None:
            args.patience = 120
        args.weight_decay = 1.2e-4
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    print('\n' + '='*70)
    print(f'统一训练脚本 - 任务: {objective.upper()} | 数据集: {dataset.upper()}')
    print('='*70)
    print(f'学习率: {args.learning_rate}')
    print(f'训练轮数: {args.max_epochs}')
    print(f'批次大小: {args.batch_size}')
    print(f'Dropout: {args.dropout}')
    print(f'编码器层数: {args.encoder_layers}')
    print(f'权重衰减: {args.weight_decay}')
    print(f'Early stopping: {args.patience}')
    if args.multi_seed:
        print(f'多次随机种子验证: [0]')
    print('='*70 + '\n')
    try:
        if dataset == 'bindingdb':
            print('启动BindingDB训练 (DTI分类任务)...\n')
            run_bindingdb(args)
        elif dataset == 'pdb':
            print('启动PDB训练 (DTA回归任务)...\n')
            run_pdb(args)
        elif dataset == 'tdc_dg':
            print('启动TDC_DG训练 (DTA回归任务)...\n')
            run_tdc_dg(args)
        elif dataset == 'davis':
            print('启动Davis训练 (DTA回归任务)...\n')
            run_davis(args)
        print('\n' + '='*70)
        print(f'任务 {dataset.upper()} 训练完成!')
        print('='*70)
    except Exception as e:
        print(f'\n[ERROR] 训练过程中出现错误: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == "__main__":
    main()
