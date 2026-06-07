import torch
import torch.nn as nn
import torch.nn.functional as F
class WeightedMSELoss(nn.Module):
    def __init__(self, threshold=5.0, high_weight=3.0, low_weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.high_weight = high_weight
        self.low_weight = low_weight
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        weights = torch.where(
            target > self.threshold,
            torch.tensor(self.high_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.low_weight, device=target.device, dtype=target.dtype)
        )
        weighted_mse = mse * weights
        return weighted_mse.mean()
class FocalMSELoss(nn.Module):
    def __init__(self, gamma=2.0, threshold=5.0, minority_weight=2.0):
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.minority_weight = minority_weight
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        difficulty = torch.abs(pred - target)
        focal_weight = (1 + difficulty) ** self.gamma
        is_minority = (target > self.threshold).float()
        minority_boost = 1.0 + self.minority_weight * is_minority
        total_weight = focal_weight * minority_boost
        weighted_mse = mse * total_weight
        return weighted_mse.mean()
class RankingLoss(nn.Module):
    def __init__(self, margin=1.0, sample_ratio=0.3):
        super(RankingLoss, self).__init__()
        self.margin = margin
        self.sample_ratio = sample_ratio
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        pred = predictions.view(-1)
        target = targets.view(-1)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        significant_pairs = torch.abs(target_diff) > 0.5
        ranking_loss = F.relu(self.margin - pred_diff * torch.sign(target_diff))
        ranking_loss = ranking_loss * significant_pairs.float()
        if self.sample_ratio < 1.0:
            num_pairs = int(batch_size * batch_size * self.sample_ratio)
            mask = torch.zeros_like(ranking_loss)
            indices = torch.randperm(batch_size * batch_size)[:num_pairs]
            mask.view(-1)[indices] = 1
            ranking_loss = ranking_loss * mask
        num_valid_pairs = significant_pairs.float().sum()
        if num_valid_pairs > 0:
            return ranking_loss.sum() / num_valid_pairs
        else:
            return torch.tensor(0.0, device=predictions.device)
class ConcordanceIndexLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ConcordanceIndexLoss, self).__init__()
        self.margin = margin
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)
        pred = predictions.view(-1)
        target = targets.view(-1)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        valid_pairs = target_diff > 0
        correct_prob = torch.sigmoid(pred_diff / self.margin)
        if valid_pairs.sum() > 0:
            ci_loss = 1.0 - (correct_prob * valid_pairs.float()).sum() / valid_pairs.float().sum()
            return ci_loss
        else:
            return torch.tensor(0.0, device=predictions.device)
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, affinity_threshold=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.affinity_threshold = affinity_threshold
    def forward(self, features, labels):
        device = features.device
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        affinity_diff = torch.abs(labels - labels.T)
        positive_mask = (affinity_diff < self.affinity_threshold).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(positive_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        positive_mask = positive_mask * logits_mask
        has_positive = positive_mask.sum(1) > 0
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=device)
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-9)
        mean_log_prob_pos = mean_log_prob_pos[has_positive]
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss
class CombinedLoss(nn.Module):
    def __init__(self,
                 mse_weight=1.0,
                 ranking_weight=0.1,
                 ranking_margin=1.0,
                 ranking_sample_ratio=0.3,
                 use_ci_loss=False,
                 ci_weight=0.05):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.use_ci_loss = use_ci_loss
        self.ci_weight = ci_weight
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = RankingLoss(margin=ranking_margin, sample_ratio=ranking_sample_ratio)
        if use_ci_loss:
            self.ci_loss = ConcordanceIndexLoss(margin=0.5)
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        ranking = self.ranking_loss(predictions, targets)
        total_loss = self.mse_weight * mse + self.ranking_weight * ranking
        if self.use_ci_loss:
            ci = self.ci_loss(predictions, targets)
            total_loss = total_loss + self.ci_weight * ci
        return total_loss
    def get_loss_components(self, predictions, targets):
        with torch.no_grad():
            mse = self.mse_loss(predictions, targets).item()
            ranking = self.ranking_loss(predictions, targets).item()
            components = {
                'mse': mse,
                'ranking': ranking,
                'total': self.mse_weight * mse + self.ranking_weight * ranking
            }
            if self.use_ci_loss:
                ci = self.ci_loss(predictions, targets).item()
                components['ci'] = ci
                components['total'] += self.ci_weight * ci
            return components
class CombinedLossWithContrastive(nn.Module):
    def __init__(self,
                 mse_weight=1.0,
                 ranking_weight=0.1,
                 contrastive_weight=0.05,
                 ranking_margin=1.0,
                 ranking_sample_ratio=0.3,
                 temperature=0.07,
                 affinity_threshold=0.5,
                 use_ci_loss=False,
                 ci_weight=0.05):
        super(CombinedLossWithContrastive, self).__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        self.use_ci_loss = use_ci_loss
        self.ci_weight = ci_weight
        self.mse_loss = nn.MSELoss()
        self.ranking_loss = RankingLoss(margin=ranking_margin, sample_ratio=ranking_sample_ratio)
        self.contrastive_loss = SupConLoss(temperature=temperature, affinity_threshold=affinity_threshold)
        if use_ci_loss:
            self.ci_loss = ConcordanceIndexLoss(margin=0.5)
    def forward(self, predictions, targets, features=None):
        mse = self.mse_loss(predictions, targets)
        ranking = self.ranking_loss(predictions, targets)
        total_loss = self.mse_weight * mse + self.ranking_weight * ranking
        if features is not None and self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(features, targets)
            total_loss = total_loss + self.contrastive_weight * contrastive
        if self.use_ci_loss:
            ci = self.ci_loss(predictions, targets)
            total_loss = total_loss + self.ci_weight * ci
        return total_loss
    def get_loss_components(self, predictions, targets, features=None):
        with torch.no_grad():
            mse = self.mse_loss(predictions, targets).item()
            ranking = self.ranking_loss(predictions, targets).item()
            components = {
                'mse': mse,
                'ranking': ranking,
                'total': self.mse_weight * mse + self.ranking_weight * ranking
            }
            if features is not None and self.contrastive_weight > 0:
                contrastive = self.contrastive_loss(features, targets).item()
                components['contrastive'] = contrastive
                components['total'] += self.contrastive_weight * contrastive
            if self.use_ci_loss:
                ci = self.ci_loss(predictions, targets).item()
                components['ci'] = ci
                components['total'] += self.ci_weight * ci
            return components
class CombinedLossWithWeighting(nn.Module):
    def __init__(self,
                 mse_weight=1.0,
                 ranking_weight=0.2,
                 contrastive_weight=0.11,
                 ranking_margin=1.0,
                 ranking_sample_ratio=0.3,
                 temperature=0.048,
                 affinity_threshold=0.5,
                 use_ci_loss=False,
                 ci_weight=0.08,
                 use_weighted_mse=True,
                 sample_weight_ratio=3.0,
                 threshold=5.0,
                 use_focal=False,
                 focal_gamma=2.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        self.use_ci_loss = use_ci_loss
        self.ci_weight = ci_weight
        self.use_weighted_mse = use_weighted_mse
        self.use_focal = use_focal
        if use_focal:
            self.mse_loss = FocalMSELoss(
                gamma=focal_gamma,
                threshold=threshold,
                minority_weight=sample_weight_ratio - 1.0
            )
            print(f'   [加权策略] 使用Focal MSE (gamma={focal_gamma}, minority_weight={sample_weight_ratio-1.0})')
        elif use_weighted_mse:
            self.mse_loss = WeightedMSELoss(
                threshold=threshold,
                high_weight=sample_weight_ratio,
                low_weight=1.0
            )
            print(f'   [加权策略] 使用Weighted MSE (非5.0样本权重={sample_weight_ratio}x)')
        else:
            self.mse_loss = nn.MSELoss()
            print(f'   [加权策略] 使用标准MSE (无加权)')
        self.ranking_loss = RankingLoss(margin=ranking_margin, sample_ratio=ranking_sample_ratio)
        self.contrastive_loss = SupConLoss(temperature=temperature, affinity_threshold=affinity_threshold)
        if use_ci_loss:
            self.ci_loss = ConcordanceIndexLoss(margin=0.5)
    def forward(self, predictions, targets, features=None):
        mse = self.mse_loss(predictions, targets)
        ranking = self.ranking_loss(predictions, targets)
        total_loss = self.mse_weight * mse + self.ranking_weight * ranking
        if features is not None and self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(features, targets)
            total_loss = total_loss + self.contrastive_weight * contrastive
        if self.use_ci_loss:
            ci = self.ci_loss(predictions, targets)
            total_loss = total_loss + self.ci_weight * ci
        return total_loss
def get_loss_function(loss_type='combined', **kwargs):
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'ranking':
        return RankingLoss(**kwargs)
    elif loss_type == 'ci':
        return ConcordanceIndexLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedLoss(**kwargs)
    elif loss_type == 'weighted':
        return CombinedLossWithWeighting(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
if __name__ == "__main__":
    print("Testing Loss Functions...")
    batch_size = 16
    predictions = torch.randn(batch_size, 1) * 2 + 5
    targets = torch.randn(batch_size, 1) * 2 + 5
    mse_loss = nn.MSELoss()
    print(f"MSE Loss: {mse_loss(predictions, targets).item():.4f}")
    ranking_loss = RankingLoss(margin=1.0, sample_ratio=0.5)
    print(f"Ranking Loss: {ranking_loss(predictions, targets).item():.4f}")
    ci_loss = ConcordanceIndexLoss(margin=0.5)
    print(f"CI Loss: {ci_loss(predictions, targets).item():.4f}")
    combined_loss = CombinedLoss(mse_weight=1.0, ranking_weight=0.1)
    loss_value = combined_loss(predictions, targets)
    print(f"Combined Loss: {loss_value.item():.4f}")
    components = combined_loss.get_loss_components(predictions, targets)
    print(f"Loss Components: {components}")
