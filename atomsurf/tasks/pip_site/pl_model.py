import os
import sys

import torch
import torch.nn.functional as F
# project
from atomsurf.tasks.pip_site.model import PIPsiteNet,PINDERNet,PINDERNet_seed
from atomsurf.utils.learning_utils import AtomPLModule
from atomsurf.utils.metrics import compute_auroc, compute_accuracy

def compute_BCE(pred, target):
    """
    Compute Binary Cross Entropy Loss with class imbalance handling.

    :param pred:    Tensor of predictions (logits).
    :param target:  Tensor of target labels (0 or 1).
    :return:        Computed BCE loss.
    """
    num_pos = target.sum()
    numels = len(target)

    # Avoid division by zero
    if num_pos.item() == 0:
        return torch.tensor(0.0, device=pred.device)

    # Compute positive weight (balancing factor)
    weight = (numels - num_pos) / max(num_pos, 1)
    weight = torch.tensor(weight, dtype=torch.float, device=pred.device)

    # Compute BCE Loss with pos_weight adjustment
    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=weight)
    return loss 
def focal_bce(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    logits: raw scores (before sigmoid) of shape [N,1]
    targets: binary labels in {0,1}, same shape
    alpha: balancing factor for class 1
    gamma: focusing parameter
    """
    # BCE with logits, per-element
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # p_t = sigmoid(logits) if target=1; 1-sigmoid if target=0
    p_t = torch.exp(-bce_loss)
    # focal weight
    weight = alpha * (1 - p_t) ** gamma
    loss = weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
class PIPsiteModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PIPsiteNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch):
        runtype = 'both'
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
        outputs = self(batch)
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss = self.criterion(outputs, labels)
        elif runtype=='l':
            labels= labels_l
            loss = self.criterion(outputs, labels)
        elif runtype=='r':
            labels= labels_r
            loss = self.criterion(outputs, labels)
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
        return loss, outputs, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/train": acc, "auroc/train": auroc}, on_epoch=True, batch_size=len(logits))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/val": acc, "auroc/val": auroc}, on_epoch=True, batch_size=len(logits))
        self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None:
            self.log("acc/test", 0.5, on_epoch=True)
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)
        self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True, batch_size=len(logits))



class PINDERModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)

    def step(self, batch):
        runtype = 'both'
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return  None, None, None,None, None, None,None
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
        outputs_site,outputs_pair = self(batch)
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss_site = compute_BCE(outputs_site, labels)
        elif runtype=='l':
            labels= labels_l
            loss_site = compute_BCE(outputs_site, labels)
        elif runtype=='r':
            labels= labels_r
            loss_site = compute_BCE(outputs_site, labels)
        if isinstance(batch.labels_pair, list):
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss = loss_site + loss_pair
        
        if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any():
            print('Nan loss')
            return None, None, None,None, None, None,None
        return loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair

    def training_step(self, batch, batch_idx):
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item(),"loss_site/train":loss_site.item(),"loss_pair/train":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/train": acc_site, "auroc_site/train": auroc_site,"acc_pair/train":acc_pair,"auroc_pair/train":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item(),"loss_site/val":loss_site.item(),"loss_pair/val":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/val": acc_site, "auroc_site/val": auroc_site,"acc_pair/val":acc_pair,"auroc_pair/val":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        # self.log("auroc_val", auroc, prog_bar=True, on_step=False, on_epoch=True, logger=False, batch_size=len(logits))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair, outputs_site,outputs_pair, labels,labels_pair = self.step(batch)
        if loss is None:
            self.log_dict({"acc_site/val": 0.5, "auroc_site/val": 0.5,"acc_pair/val":0.5,"auroc_pair/val":0.5}, on_epoch=True, batch_size=0)
            return None
        self.log_dict({"loss/test": loss.item(),"loss_site/test":loss_site.item(),"loss_pair/test":loss_pair.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        self.log_dict({"acc_site/test": acc_site, "auroc_site/test": auroc_site,"acc_pair/test":acc_pair,"auroc_pair/test":auroc_pair}, on_epoch=True, batch_size=len(outputs_site))
        # self.log_dict({"acc/test": acc, "auroc/test": auroc}, on_epoch=True, batch_size=len(logits))

import torch
import torch.nn.functional as F

def focal_bce(logits, labels, alpha=0.01, gamma=2.0):
    """Focal loss variant of binary cross entropy."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()

def compute_BCE(outputs, labels):
    """Standard binary cross entropy loss."""
    return F.binary_cross_entropy_with_logits(outputs, labels)

def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
    """Contrastive loss for Euclidean distances."""
    loss_pos = pos_weight * labels * distances.pow(2)
    loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
    return (loss_pos + loss_neg).mean()

def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
    """Contrastive loss for cosine similarities."""
    loss_pos = pos_weight * labels * (1 - similarities).pow(2)
    loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)
    return (loss_pos + loss_neg).mean()

class PINDERModule_seed(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet_seed(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)
        self.save_embed = False #cfg.save_embed
        self.save_embed_dir = None #cfg.save_embed
    def _check_nan(self, tensor, name):
        """Check for NaN values and provide diagnostic information."""
        if torch.isnan(tensor).any():
            print(f"⚠️  NaN detected in {name}")
            print(f"   - Shape: {tensor.shape}")
            print(f"   - Min: {tensor[~torch.isnan(tensor)].min() if (~torch.isnan(tensor)).any() else 'all NaN'}")
            print(f"   - Max: {tensor[~torch.isnan(tensor)].max() if (~torch.isnan(tensor)).any() else 'all NaN'}")
            print(f"   - NaN count: {torch.isnan(tensor).sum().item()}/{tensor.numel()}")
            return True
        return False
    
    def _safe_normalize(self, tensor, dim=1, eps=1e-8):
        """Safely normalize tensors with NaN protection."""
        norm = tensor.norm(p=2, dim=dim, keepdim=True)
        norm = torch.clamp(norm, min=eps)  # Prevent division by zero
        return tensor / norm
    
    def _prepare_labels(self, batch, runtype='both'):
        """Prepare labels based on run type."""
        # Handle list or tensor labels
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
            s_site_label_L = torch.cat(batch.s_site_label_L).reshape(-1, 1)
            s_site_label_R = torch.cat(batch.s_site_label_R).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
            s_site_label_L = batch.s_site_label_L.reshape(-1, 1)
            s_site_label_R = batch.s_site_label_R.reshape(-1, 1)
        
        # Combine based on runtype
        if runtype == 'both':
            labels = torch.cat([labels_l, labels_r])
            s_site_label = torch.cat([s_site_label_L, s_site_label_R])
        elif runtype == 'l':
            labels = labels_l
            s_site_label = s_site_label_L
        else:  # runtype == 'r'
            labels = labels_r
            s_site_label = s_site_label_R
        
        return labels, s_site_label
    
    def _compute_complementarity_losses(self, emb1, emb2, emb1_inv, emb2_inv, 
                                       s_emb1, s_emb2, s_emb1_inv, s_emb2_inv,
                                       labels_pair, s_label):
        """Compute complementarity losses with NaN protection."""
        
        # Safely normalize all embeddings
        emb1 = self._safe_normalize(emb1)
        emb1_inv = self._safe_normalize(emb1_inv)
        emb2 = self._safe_normalize(emb2)
        emb2_inv = self._safe_normalize(emb2_inv)
        s_emb1 = self._safe_normalize(s_emb1)
        s_emb1_inv = self._safe_normalize(s_emb1_inv)
        s_emb2 = self._safe_normalize(s_emb2)
        s_emb2_inv = self._safe_normalize(s_emb2_inv)
        
        # Check for NaN after normalization
        if self._check_nan(emb1, "emb1 after normalization"):
            return None, None, None, None, None, None, None, None
        
        # Cosine similarities
        cos_dists_1_2 = F.cosine_similarity(emb1, emb2_inv).view(-1, 1)
        cos_dists_2_1 = F.cosine_similarity(emb2, emb1_inv).view(-1, 1)
        cos_s_dists_1_2 = F.cosine_similarity(s_emb1, s_emb2_inv).view(-1, 1)
        cos_s_dists_2_1 = F.cosine_similarity(s_emb2, s_emb1_inv).view(-1, 1)
        
        # Focal BCE losses
        loss_comp_g_cos = (
            focal_bce(cos_dists_1_2, labels_pair, alpha=0.01, gamma=2.0) +
            focal_bce(cos_dists_2_1, labels_pair, alpha=0.01, gamma=2.0)
        )
        
        loss_comp_s_cos = (
            focal_bce(cos_s_dists_1_2, s_label, alpha=0.01, gamma=2.0) +
            focal_bce(cos_s_dists_2_1, s_label, alpha=0.01, gamma=2.0)
        )
        
        # Euclidean distances (with gradient clipping to prevent explosion)
        dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True).clamp(max=10.0)
        dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True).clamp(max=10.0)
        s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True).clamp(max=10.0)
        s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True).clamp(max=10.0)
        
        # Contrastive losses
        loss_comp_g = (
            contrastive_loss(dists_1_2, labels_pair, margin=2.0) +
            contrastive_loss(dists_2_1, labels_pair, margin=2.0) +
            cos_contrastive_loss(cos_dists_1_2, labels_pair) +
            cos_contrastive_loss(cos_dists_2_1, labels_pair)
        )
        
        loss_comp_s = (
            contrastive_loss(s_dists_1_2, s_label, margin=2.0, pos_weight=50.0) +
            contrastive_loss(s_dists_2_1, s_label, margin=2.0, pos_weight=50.0) +
            cos_contrastive_loss(cos_s_dists_1_2, s_label, pos_weight=50.0) +
            cos_contrastive_loss(cos_s_dists_2_1, s_label, pos_weight=50.0)
        )
        
        return (loss_comp_g, loss_comp_s, loss_comp_g_cos, loss_comp_s_cos,
                cos_dists_1_2, cos_dists_2_1, cos_s_dists_1_2, cos_s_dists_2_1)
    
    def step(self, batch, save_emb=False):
        """
        Main training step with improved error handling and NaN detection.
        
        Returns None tuple if batch is invalid or NaN is detected.
        """
        runtype = 'both'
        none_return = (None,) * 19
        
        # Validate batch
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return none_return
        
        # Check batch size
        total_nodes = batch.surface_1.x.shape[0] + batch.surface_2.x.shape[0]
        print(f"Batch size: {total_nodes} nodes")
        
        if total_nodes > 180000:
            print("⚠️  Batch too large, skipping...")
            return none_return
        
        # Prepare labels
        labels, s_site_label = self._prepare_labels(batch, runtype)
        
        # Handle pair labels
        if isinstance(batch.labels_pair, list):
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        
        if isinstance(batch.s_label, list):
            s_label = torch.cat(batch.s_label).reshape(-1, 1)
        else:
            s_label = batch.s_label.reshape(-1, 1)
        
        # Forward pass

        try:
            (outputs_site, outputs_pair, outputs_surface_site, outputs_surface_pair,
             emb1, emb2, emb1_inv, emb2_inv,
             s_emb1, s_emb2, s_emb1_inv, s_emb2_inv) = self.model(batch, save_emb)
        except RuntimeError as e:
            print(f"❌ Forward pass failed: {e}")
            return none_return
        
        # Check for NaN in model outputs
        if self._check_nan(outputs_site, "outputs_site"):
            print("💡 Solution: Check model architecture, reduce learning rate, or use gradient clipping")
            return none_return
        
        # Compute site losses
        loss_site = compute_BCE(outputs_site, labels)
        loss_surf_site = compute_BCE(outputs_surface_site, s_site_label)
        
        # Compute pair losses
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss_surface_pair = compute_BCE(outputs_surface_pair, s_label)
        
        # Check for NaN in basic losses
        if any(torch.isnan(l).any() for l in [loss_site, loss_pair, loss_surf_site, loss_surface_pair]):
            print("❌ NaN detected in basic losses")
            self._check_nan(loss_site, "loss_site")
            self._check_nan(loss_pair, "loss_pair")
            self._check_nan(loss_surf_site, "loss_surf_site")
            self._check_nan(loss_surface_pair, "loss_surface_pair")
            print("💡 Solution: Check for extreme values in labels or outputs, consider label smoothing")
            return none_return
        
        # Compute complementarity losses
        comp_results = self._compute_complementarity_losses(
            emb1, emb2, emb1_inv, emb2_inv,
            s_emb1, s_emb2, s_emb1_inv, s_emb2_inv,
            labels_pair, s_label
        )
        
        if comp_results[0] is None:
            print("❌ NaN detected in complementarity losses")
            print("💡 Solution: Reduce embedding dimensionality or add batch normalization")
            return none_return
        
        (loss_comp_g, loss_comp_s, loss_comp_g_cos, loss_comp_s_cos,
         cos_dists_1_2, cos_dists_2_1, cos_s_dists_1_2, cos_s_dists_2_1) = comp_results
        
        # Combine losses with gradient clipping on individual components
        loss = (
            10* loss_site + 
            loss_pair + 
            10* loss_surf_site + 
            loss_surface_pair + 
            torch.clamp(loss_comp_g, max=10.0) + 
            torch.clamp(loss_comp_s, max=10.0) + 
            5 * (torch.clamp(loss_comp_g_cos, max=10.0) + torch.clamp(loss_comp_s_cos, max=10.0))
        )
        
        # Final NaN check
        if torch.isnan(loss).any():
            print("❌ NaN in final loss")
            self._check_nan(loss_comp_g, "loss_comp_g")
            self._check_nan(loss_comp_s, "loss_comp_s")
            self._check_nan(loss_comp_g_cos, "loss_comp_g_cos")
            self._check_nan(loss_comp_s_cos, "loss_comp_s_cos")
            
            self.nan_count += 1
            if self.nan_count <= self.max_nan_warnings:
                print("\n" + "="*60)
                print("🔍 NaN DEBUGGING RECOMMENDATIONS:")
                print("="*60)
                print("1. Reduce learning rate (try 1e-5 or 1e-6)")
                print("2. Add gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
                print("3. Use mixed precision training with loss scaling")
                print("4. Check for zero or very small batch statistics")
                print("5. Add LayerNorm or BatchNorm before embeddings")
                print("6. Reduce pos_weight from 50.0 to 5.0 or 10.0")
                print("7. Initialize weights with smaller variance")
                print("8. Add dropout for regularization")
                print("="*60 + "\n")
            
            return none_return
        
        return (
            loss, loss_site, loss_pair, loss_surf_site, loss_surface_pair,
            loss_comp_g, loss_comp_s,
            outputs_site, outputs_pair, outputs_surface_site, outputs_surface_pair,
            labels, labels_pair, s_site_label, s_label,
            cos_dists_1_2, cos_dists_2_1, cos_s_dists_1_2, cos_s_dists_2_1
        )        
    # def step(self, batch,save_emb=False):
    #     runtype = 'both'
    #     # import pdb
    #     # pdb.set_trace()
    #     if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
    #         return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
    #     print(batch.surface_1.x.shape[0]+ batch.surface_2.x.shape[0])
    #     if batch.surface_1.x.shape[0]+ batch.surface_2.x.shape[0] > 180000:
    #         print('too big batch, skip')
    #         return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None

    #     if isinstance(batch.label_l, list):
    #         labels_l = torch.cat(batch.label_l).reshape(-1, 1)
    #         labels_r = torch.cat(batch.label_r).reshape(-1, 1)
    #         s_site_label_L = torch.cat(batch.s_site_label_L).reshape(-1, 1)
    #         s_site_label_R = torch.cat(batch.s_site_label_R).reshape(-1, 1)
    #     else:
    #         labels_l = batch.label_l.reshape(-1, 1)
    #         labels_r = batch.label_r.reshape(-1, 1)
    #         s_site_label_L = batch.s_site_label_L.reshape(-1, 1)
    #         s_site_label_R = batch.s_site_label_R.reshape(-1, 1)
    #     outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair,emb1,emb2,emb1_inv,emb2_inv,s_emb1,s_emb2,s_emb1_inv,s_emb2_inv = self.model(batch,save_emb)
    #     if runtype=='both':
    #         labels = torch.cat([labels_l,labels_r])
    #         loss_site = compute_BCE(outputs_site, labels)
    #         s_site_label = torch.cat([s_site_label_L,s_site_label_R])
    #         loss_surf_site = compute_BCE(outputs_surface_site, s_site_label)
    #     elif runtype=='l':
    #         labels= labels_l
    #         loss_site = compute_BCE(outputs_site, labels)
    #         s_site_label = s_site_label_L
    #         loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_L)
    #     elif runtype=='r':
    #         labels= labels_r
    #         s_site_label = s_site_label_R
    #         loss_site = compute_BCE(outputs_site, labels)
    #         loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_R)
    #     if isinstance(batch.labels_pair, list):
    #         label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.labels_pair)]).to(batch.labels_pair[0].device)
    #         labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
    #     else:
    #         labels_pair = batch.labels_pair.reshape(-1, 1)
    #     if isinstance(batch.s_label, list):
    #         s_label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.s_label)]).to(batch.s_label[0].device)
    #         s_label = torch.cat(batch.s_label).reshape(-1, 1)
    #     else:
    #         s_label = batch.s_label.reshape(-1, 1)
    #     loss_pair = compute_BCE(outputs_pair, labels_pair)
    #     loss_surface_pair= compute_BCE(outputs_surface_pair, s_label)
    #     ## add complementarity loss, dot product for similarity
    #     ## old loss for complementary
    #     complementray =  'euclidean' #'cosine'
    #     if complementray=='cosine':
    #         dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
    #         dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
    #         loss_complementartiy_g = compute_BCE(dists_1_2, labels_pair) + compute_BCE(dists_2_1, labels_pair) 

    #         s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
    #         s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
    #         loss_complementartiy_s = compute_BCE(s_dists_1_2, s_label) + compute_BCE(s_dists_2_1, s_label)
    #     elif complementray=='euclidean':
    #         # emb1 = F.normalize(emb1, dim=1)
    #         # emb1_inv = F.normalize(emb1_inv, dim=1)
    #         # emb2 = F.normalize(emb2, dim=1)
    #         # emb2_inv = F.normalize(emb2_inv, dim=1)
    #         # s_emb1 = F.normalize(s_emb1, dim=1)
    #         # s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
    #         # s_emb2 = F.normalize(s_emb2, dim=1)
    #         # s_emb2_inv = F.normalize(s_emb2_inv, dim=1)

    #         # cos_dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
    #         # cos_dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
    #         # loss_complementartiy_g_cos = compute_BCE(cos_dists_1_2, labels_pair) + compute_BCE(cos_dists_2_1, labels_pair) 
    #         # cos_s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
    #         # cos_s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
    #         # loss_complementartiy_s_cos = compute_BCE(cos_s_dists_1_2, s_label) + compute_BCE(cos_s_dists_2_1, s_label)
    #         # def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
    #         #     loss_pos = pos_weight * labels * distances.pow(2)
    #         #     loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
    #         #     loss = loss_pos + loss_neg
    #         #     return loss.mean()
    #         # def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
    #         #     loss_pos = pos_weight * labels * (1 - similarities).pow(2)  # Pull positives to 1
    #         #     loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)  # Push negatives below margin
    #         #     return (loss_pos + loss_neg).mean()
            
    #         # dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True)
    #         # dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True)

    #         # # Apply contrastive loss for ground truth pairs
    #         # loss_complementartiy_g = contrastive_loss(dists_1_2, labels_pair,margin=2.0) + contrastive_loss(dists_2_1, labels_pair,margin=2.0)+cos_contrastive_loss(cos_dists_1_2, labels_pair)+cos_contrastive_loss(cos_dists_2_1, labels_pair)
            
    #         # # Compute Euclidean distances for the secondary embeddings
    #         # s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True)
    #         # s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True)

    #         # # Apply contrastive loss for secondary pairs
    #         # loss_complementartiy_s = contrastive_loss(s_dists_1_2, s_label,margin=2.0,pos_weight=50.0) + contrastive_loss(s_dists_2_1, s_label,margin=2.0,pos_weight=50.0)+cos_contrastive_loss(cos_s_dists_1_2, s_label,pos_weight=50.0)+cos_contrastive_loss(cos_s_dists_2_1, s_label,pos_weight=50.0)
    #         emb1 = F.normalize(emb1, dim=1)
    #         emb1_inv = F.normalize(emb1_inv, dim=1)
    #         emb2 = F.normalize(emb2, dim=1)
    #         emb2_inv = F.normalize(emb2_inv, dim=1)
    #         s_emb1 = F.normalize(s_emb1, dim=1)
    #         s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
    #         s_emb2 = F.normalize(s_emb2, dim=1)
    #         s_emb2_inv = F.normalize(s_emb2_inv, dim=1)

    #         # cosine similarities as “logits”
    #         cos_dists_1_2 = F.cosine_similarity(emb1, emb2_inv).view(-1,1)
    #         cos_dists_2_1 = F.cosine_similarity(emb2, emb1_inv).view(-1,1)
    #         cos_s_dists_1_2 = F.cosine_similarity(s_emb1, s_emb2_inv).view(-1,1)
    #         cos_s_dists_2_1 = F.cosine_similarity(s_emb2, s_emb1_inv).view(-1,1)

    #         # replace compute_BCE with focal_bce
    #         loss_complementarity_g_cos = (
    #             focal_bce(cos_dists_1_2, labels_pair, alpha=0.01, gamma=2.0)
    #         + focal_bce(cos_dists_2_1, labels_pair, alpha=0.01, gamma=2.0)
    #         )

    #         loss_complementarity_s_cos = (
    #             focal_bce(cos_s_dists_1_2, s_label,    alpha=0.01, gamma=2.0)
    #         + focal_bce(cos_s_dists_2_1, s_label,    alpha=0.01, gamma=2.0)
    #         )

    #         # your existing (non-BCE) contrastive losses
    #         def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
    #             loss_pos = pos_weight * labels * distances.pow(2)
    #             loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
    #             return (loss_pos + loss_neg).mean()

    #         def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
    #             loss_pos = pos_weight * labels * (1 - similarities).pow(2)
    #             loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)
    #             return (loss_pos + loss_neg).mean()

    #         # Euclidean distances
    #         dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True)
    #         dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True)
    #         s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True)
    #         s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True)

    #         # final loss aggregation
    #         loss_complementarity_g = (
    #             contrastive_loss(dists_1_2,   labels_pair, margin=2.0)
    #             + contrastive_loss(dists_2_1,   labels_pair, margin=2.0)
    #             + cos_contrastive_loss(cos_dists_1_2, labels_pair)
    #             + cos_contrastive_loss(cos_dists_2_1, labels_pair)
    #         )

    #         loss_complementarity_s = (
    #             contrastive_loss(s_dists_1_2, s_label, margin=2.0, pos_weight=50.0)
    #             + contrastive_loss(s_dists_2_1, s_label, margin=2.0, pos_weight=50.0)
    #             + cos_contrastive_loss(cos_s_dists_1_2, s_label, pos_weight=50.0)
    #             + cos_contrastive_loss(cos_s_dists_2_1, s_label, pos_weight=50.0)
    #         )
    #     # # new loss for complementary
    #     # emb1 = F.normalize(emb1, dim=1)
    #     # emb1_inv = F.normalize(emb1_inv, dim=1)
    #     # emb2 = F.normalize(emb2, dim=1)
    #     # emb2_inv = F.normalize(emb2_inv, dim=1)
    #     # adjmatrix= label_batch[:,None]==label_batch[None,:]
    #     # mask = (torch.matmul(labels_pair,labels_pair.T) *adjmatrix* torch.eye(len(emb1)).to(labels_pair.device)).view(-1, 1)
    #     # sim1 = (torch.matmul(emb1,emb2_inv.T)/0.1).view(-1, 1)
    #     # sim2 = (torch.matmul(emb1_inv,emb2.T)/0.1).view(-1, 1)
    #     # loss_comp_g_new = compute_BCE(sim1,mask)+compute_BCE(sim2,mask)
    #     # inverse_loss_g = torch.mean((F.cosine_similarity(emb1, emb1_inv, dim=1) + 1) ** 2) + torch.mean((F.cosine_similarity(emb2, emb2_inv, dim=1) + 1) ** 2) 
        
    #     # # new s loss for complementary
    #     # s_emb1 = F.normalize(s_emb1, dim=1)
    #     # s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
    #     # s_emb2 = F.normalize(s_emb2, dim=1)
    #     # s_emb2_inv = F.normalize(s_emb2_inv, dim=1)
    #     # pos_indices = torch.nonzero(s_label == 1, as_tuple=True)[0]
    #     # num_pos = pos_indices.size(0)
    #     # sample_size=int(torch.sqrt(torch.tensor(len(s_label))))
    #     # i_idx = pos_indices[torch.randint(0, num_pos, (sample_size,))]
    #     # j_idx = pos_indices[torch.randint(0, num_pos, (sample_size,))]

    #     # s_sim1_sampled = (s_emb1[i_idx] * s_emb2_inv[j_idx]).sum(dim=1) / 0.1
    #     # s_sim2_sampled = (s_emb1_inv[i_idx] * s_emb2[j_idx]).sum(dim=1) / 0.1

    #     # s_mask_sampled = torch.tensor((i_idx==j_idx)).float().unsqueeze(1).to(s_emb1.device)
        
    #     # Compute the loss using your binary cross-entropy loss function.
    #     # loss_comp_s_new = compute_BCE(s_sim1_sampled.view(-1, 1), s_mask_sampled) + \
    #     #                     compute_BCE(s_sim2_sampled.view(-1, 1), s_mask_sampled)
    
    #     # s_adjmatrix= s_label_batch[:,None]==s_label_batch[None,:] 
    #     # s_mask = (torch.matmul(s_label,s_label.T) *s_adjmatrix* torch.eye(len(s_emb1)).to(s_label.device)).view(-1, 1)
    #     # s_sim1 = (torch.matmul(s_emb1,s_emb2_inv.T)/0.1).view(-1, 1)
    #     # s_sim2 = (torch.matmul(s_emb1_inv,s_emb2.T)/0.1).view(-1, 1)
    #     # loss_comp_s_new = compute_BCE(s_sim1,s_mask)+compute_BCE(s_sim2,s_mask)
    #     # inverse_loss_s = torch.mean((F.cosine_similarity(s_emb1, s_emb1_inv, dim=1) + 1) ** 2) + torch.mean((F.cosine_similarity(s_emb2, s_emb2_inv, dim=1) + 1) ** 2) 
    #     ### best loss :loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_complementartiy_g + loss_complementartiy_s
    #     loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_complementarity_g + loss_complementarity_s + 5*(loss_complementarity_g_cos+ loss_complementarity_s_cos)
    #     # loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_complementartiy_g + loss_complementartiy_s +inverse_loss_g+inverse_loss_s+loss_comp_g_new+loss_comp_s_new
    #     # loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_comp_g_new + loss_comp_s_new+ inverse_loss_g+inverse_loss_s
    #     # loss = loss_complementartiy_g + loss_complementartiy_s+ inverse_loss_g+inverse_loss_s
    #     # TODO: change loss pass in return and in each step logging
    #     if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any() or torch.isnan(loss_surface_pair).any() :
    #         print('Nan loss')
    #         return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
    #     return loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementarity_g,loss_complementarity_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,cos_dists_1_2,cos_dists_2_1,cos_s_dists_1_2,cos_s_dists_2_1

    def training_step(self, batch, batch_idx):
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1= self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item(),"loss_site/train":loss_site.item(),"loss_pair/train":loss_pair.item(),"loss_surface_site/train":loss_surf_site.item(),"loss_surface/train":loss_surface_pair.item(),"loss_complementartiy_g/train":loss_complementartiy_g.item(),"loss_complementartiy_s/train":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)
        #
        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/train": acc_site, "auroc_site/train": auroc_site,"acc_pair/train":acc_pair,"auroc_pair/train":auroc_pair,'acc_surface_site/train':acc_surface_site,'auroc_surface_site/train':auroc_surface_site,"acc_surface/train":acc_surface,"auroc_surface/train":auroc_surface,'auroc_graph_seed1/train':auroc_graph_seed1,'auroc_graph_seed2/train':auroc_graph_seed2,'auroc_surface_seed1/train':auroc_surface_seed1,'auroc_surface_seed2/train':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item(),"loss_site/val":loss_site.item(),"loss_pair/val":loss_pair.item(),"loss_surface/val":loss_surface_pair.item(),"loss_complementartiy_g/val":loss_complementartiy_g.item(),"loss_complementartiy_s/val":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)

        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/val": acc_site, "auroc_site/val": auroc_site,"acc_pair/val":acc_pair,"auroc_pair/val":auroc_pair,'acc_surface_site/val':acc_surface_site,'auroc_surface_site/val':auroc_surface_site,"acc_surface/val":acc_surface,"auroc_surface/val":auroc_surface,'auroc_graph_seed1/val':auroc_graph_seed1,'auroc_graph_seed2/val':auroc_graph_seed2,'auroc_surface_seed1/val':auroc_surface_seed1,'auroc_surface_seed2/val':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch,save_emb=True)
        if loss is None:
            print("test step skipped!")
            self.log("auroc/test", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/test": loss.item(),"loss_site/test":loss_site.item(),"loss_pair/test":loss_pair.item(),"loss_surface/test":loss_surface_pair.item(),"loss_complementartiy_g/test":loss_complementartiy_g.item(),"loss_complementartiy_s/test":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)
        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/test": acc_site, "auroc_site/test": auroc_site,"acc_pair/test":acc_pair,"auroc_pair/test":auroc_pair,'acc_surface_site/test':acc_surface_site,'auroc_surface_site/test':auroc_surface_site,"acc_surface/test":acc_surface,"auroc_surface/test":auroc_surface,'auroc_graph_seed1/test':auroc_graph_seed1,'auroc_graph_seed2/test':auroc_graph_seed2,'auroc_surface_seed1/test':auroc_surface_seed1,'auroc_surface_seed2/test':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
    
    def predict_step(self, batch, batch_idx: int):
        self.model.train()
        import h5py
        if batch is  None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            print('None batch')
        else:
            graph_1,graph_2,surface_1,surface_2,g_site_L_pred,g_site_R_pred,s_site_L_pred,s_site_R_pred= self.model.extract_embedding(batch)
            save_dir = "/work/lpdi/users/ymiao/code/sbatch/pinderlog/"
            os.makedirs(save_dir, exist_ok=True)
            hdf5_path = os.path.join(save_dir, "embeddings_new.h5")
            #change it to  self.hparams.save_dir and hdf5_path later
            # Open HDF5 file in append mode

            graph_1_list= graph_1.to_data_list()
            graph_2_list= graph_2.to_data_list()
            surface_1_list= surface_1.to_data_list()
            surface_2_list= surface_2.to_data_list()
            
            with h5py.File(hdf5_path, "a") as hf:
                for batch_idx in graph_1.batch.unique():
                    tmp_id2, tmp_id1 = batch.id[batch_idx].split('--')
                    # print(tmp_id2,tmp_id1)
                    # Create a single large group for everything
                    grp = hf.require_group("embedding")#hf.require_group(f"batch_{batch_idx}")
                    if f"graph/{tmp_id1}" not in grp:
                        tmp_graph= graph_1_list[batch_idx]
                        tmp_graph_d= {}
                        tmp_graph_d['edge_index']=tmp_graph.edge_index.cpu().numpy()
                        tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                        tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                        tmp_graph_d['emb']=graph_1.x_emb[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['inv_emb']=graph_1.x_inv[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['site_pred']=g_site_L_pred[graph_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['label']=batch.label_l[batch_idx].detach().cpu().numpy()
                        grp_graph1 = grp.require_group(f"graph/{tmp_id1}")
                        save_dict_to_group(grp_graph1, tmp_graph_d)
                        tmp_surface1 = surface_1_list[batch_idx]
                        tmp_surface1_d = {}
                        tmp_surface1_d['verts'] = tmp_surface1.verts.cpu().numpy() 
                        tmp_surface1_d['faces'] = tmp_surface1.faces.cpu().numpy() 
                        tmp_surface1_d['vnormals'] = tmp_surface1.vnormals.cpu().numpy() 
                        tmp_surface1_d['emb'] = surface_1.x_emb[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['inv_emb'] = surface_1.x_inv[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['site_pred'] = s_site_L_pred[surface_1.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface1_d['label'] = batch.s_site_label_L[batch_idx].detach().cpu().numpy()
                        grp_surface1 = grp.require_group(f"surface/{tmp_id1}")
                        save_dict_to_group(grp_surface1, tmp_surface1_d)

                    if f"graph/{tmp_id2}" not in grp:
                        tmp_graph= graph_2_list[batch_idx]
                        tmp_graph_d= {}
                        tmp_graph_d['edge_index'] = tmp_graph.edge_index.cpu().numpy()
                        tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                        tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                        tmp_graph_d['emb']=graph_2.x_emb[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['inv_emb']=graph_2.x_inv[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['site_pred']=g_site_R_pred[graph_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_graph_d['label']=batch.label_r[batch_idx].detach().cpu().numpy()
                        grp_graph2 = grp.require_group(f"graph/{tmp_id2}")
                        save_dict_to_group(grp_graph2, tmp_graph_d)
                        tmp_surface2 = surface_2_list[batch_idx]
                        tmp_surface2_d = {}
                        tmp_surface2_d['verts'] = tmp_surface2.verts.cpu().numpy() 
                        tmp_surface2_d['faces'] = tmp_surface2.faces.cpu().numpy() 
                        tmp_surface2_d['vnormals'] = tmp_surface2.vnormals.cpu().numpy() 
                        tmp_surface2_d['emb'] = surface_2.x_emb[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['inv_emb'] = surface_2.x_inv[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['site_pred'] = s_site_R_pred[surface_2.batch == batch_idx].detach().cpu().numpy()
                        tmp_surface2_d['label'] = batch.s_site_label_R[batch_idx].detach().cpu().numpy()
                        grp_surface2 = grp.require_group(f"surface/{tmp_id2}")
                        save_dict_to_group(grp_surface2, tmp_surface2_d) 

                    

            print(f"Saved batch {batch_idx} embeddings to {hdf5_path}")

def save_dict_to_group(h5group, data_dict):
    """
    Save each key-value pair from data_dict into the provided h5group.
    Assumes values are NumPy arrays.
    """
    for key, value in data_dict.items():
        if value is not None:
            h5group.create_dataset(key, data=value)
        else:
            print(f"Warning: Value for key {key} is None; skipping.")


from atomsurf.utils.geometric_processing import atoms_to_points_normals
from atomsurf.network_utils.misc_arch.dmasif_utils.geometry_processing import curvatures
from torch_geometric.data import Data, Batch
from pykeops.torch import LazyTensor

# def get_interface_pairs_pykeops(points_1, batch_points_1, 
#                                 points_2, batch_points_2, threshold=2.0):
#     # points_1: [N1, D], points_2: [N2, D]
#     # batch_points_1: [N1], batch_points_2: [N2]
#     N1, D = points_1.shape
#     N2, _ = points_2.shape
    
#     # Unsqueeze to add a dummy dimension for broadcasting, then wrap into LazyTensors.
#     X_i = LazyTensor(points_1.unsqueeze(1))  # shape: [N1, 1, D]
#     Y_j = LazyTensor(points_2.unsqueeze(0))  # shape: [1, N2, D]
    
#     # Compute the squared Euclidean distances. The result is a LazyTensor of shape [N1, N2].
#     D2 = ((X_i - Y_j) ** 2).sum(-1)
    
#     # For the batch indices, unsqueeze to prepare them for broadcasting.
#     # Here we use standard PyTorch tensors.
#     b1 = batch_points_1.float().view(-1, 1)  # [N1, 1]
#     b2 = batch_points_2.float().view(-1, 1)  # [N2, 1]
#     # To subtract pairwise, unsqueeze them to add an extra dimension.
#     # b1: [N1, 1] -> [N1, 1, 1] and b2: [N2, 1] -> [1, N2, 1]
#     b1_lt = LazyTensor(b1.unsqueeze(1))  # [N1, 1, 1]
#     b2_lt = LazyTensor(b2.unsqueeze(0))   
    
#     # Compute the absolute difference between batch indices.
#     diff_batch = (b1_lt - b2_lt).abs()  # shape: [N1, N2]
    
#     # Add a large constant to distances where the batch indices differ.
#     big = 1e6
#     D2 = D2 + diff_batch * big
    
#     # Create a mask for pairs with squared distances below the threshold.
#     mask = D2 < (threshold ** 2)
    
#     mask_mat = mask.torch()
#     # Extract indices where the condition holds.
#     idx_i, idx_j = torch.where(mask_mat)
#     return idx_i, idx_j

def get_interface_and_pair_indices(points_1, batch_points_1, 
                                   points_2, batch_points_2, threshold=2.0):
    # Lists to accumulate global indices and the corresponding batch id.
    interface_1_list = []
    interface_2_list = []
    batch_interface_list = []

    # Get unique batch IDs (assuming both surfaces share the same batch structure)
    batch_ids = batch_points_1.unique()

    for b in batch_ids:
        # Get the indices for this batch element in each surface
        idx1 = (batch_points_1 == b).nonzero(as_tuple=True)[0]
        idx2 = (batch_points_2 == b).nonzero(as_tuple=True)[0]
        
        # Skip if either surface is empty for this batch.
        if idx1.numel() == 0 or idx2.numel() == 0:
            continue
        
        # Extract the surface points for this batch element.
        pts1 = points_1[idx1]
        pts2 = points_2[idx2]
        
        # Compute the pairwise distance matrix between the two sets of points.
        dist_mat = torch.cdist(pts1, pts2)
        
        # Create a boolean mask of where distances are below the threshold.
        interface_mask = dist_mat < threshold
        
        # Get the indices (relative to pts1 and pts2) where the condition holds.
        rel_idx1, rel_idx2 = torch.where(interface_mask)
        
        # Map these relative indices back to global indices.
        # global_idx1 = idx1[rel_idx1]
        # global_idx2 = idx2[rel_idx2]
        
        # Append the results.
        if len(rel_idx1)>0:
            interface_1_list.append(rel_idx1)
            interface_2_list.append(rel_idx2)
        else:
            interface_1_list.append([])
            interface_2_list.append([])
        # Create a tensor of the same length with the current batch id.
        # batch_interface_list.append(torch.full((global_idx1.shape[0],), b, 
        #                                        dtype=global_idx1.dtype,
        #                                        device=global_idx1.device))
    
    # If no interfaces were found, you may return None or empty tensors.
    if not interface_1_list:
        return None, None, None
    
    # Concatenate the results from all batches.
    # interface_1 = torch.cat(interface_1_list, dim=0)
    # interface_2 = torch.cat(interface_2_list, dim=0)
    # batch_interface = torch.cat(batch_interface_list, dim=0)
    
    # return interface_1,interface_2,batch_interface #interface_1_list, interface_2_list
    return interface_1_list, interface_2_list

def sample_negative_pairs_fast(s_idxL_pos, s_idxR_pos, total_left, total_right, neg_num, device):
    """
    Vectorized negative pair sampling using a hashing trick.
    
    Parameters:
      s_idxL_pos, s_idxR_pos: 1D LongTensors with positive pairs (global indices).
      total_left: total number of vertices in surface_L.
      total_right: total number of vertices in surface_R.
      neg_num: number of negative pairs to sample.
      device: torch device.
    
    Returns:
      neg_idxL, neg_idxR: sampled negative pair indices as 1D LongTensors.
    """
    # Compute a unique hash for each positive pair.
    pos_hash = s_idxL_pos.to(torch.long) * total_right + s_idxR_pos.to(torch.long)
    # Ensure unique hashes for faster checking.
    pos_hash_unique = pos_hash.unique()

    # Oversample candidate pairs.
    candidate_count = neg_num * 5  # Oversample factor can be tuned.
    candidate_left = torch.randint(0, total_left, (candidate_count,), device=device)
    candidate_right = torch.randint(0, total_right, (candidate_count,), device=device)
    # Compute hash for candidate pairs.
    candidate_hash = candidate_left * total_right + candidate_right

    # Use torch.isin (PyTorch >= 1.10) to find candidates that are in the positive set.
    isin_mask = torch.isin(candidate_hash, pos_hash_unique)
    valid_mask = ~isin_mask  # Candidates not in positive set.
    
    # Get indices of valid candidates.
    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() < neg_num:
        # Not enough negatives found; warn and take what is available.
        print(f"Warning: Only found {valid_idx.numel()} negative pairs out of requested {neg_num}.")
        selected_idx = valid_idx
    else:
        selected_idx = valid_idx[:neg_num]
    
    neg_idxL = candidate_left[selected_idx]
    neg_idxR = candidate_right[selected_idx]
    return neg_idxL, neg_idxR
def process_interface_list(interface_1, interface_2, batch_interface,
                           surface_L, surface_R, pos_mult=200):
    """
    Process the interface list to compute per-sample pairwise indices and sample negatives.
    
    Returns a dictionary keyed by batch id with:
      - s_idx_left, s_idx_right: concatenated positive and negative pair indices.
      - s_label: binary labels (1 for positive, 0 for negative).
      - s_site_idxL_sample, s_site_idxR_sample: sampled interface site indices for each surface.
      - s_site_label_L, s_site_label_R: corresponding binary labels for site-level classification.
    """

    results = {
        's_idx_left': [],
        's_idx_right': [],
        's_label': [],
        's_site_idxL_sample': [],
        's_site_idxR_sample': [],
        's_site_label_L': [],
        's_site_label_R': [],
        's1_len':[],
        's2_len':[],
    }
    
    for b in batch_interface.unique():
        # Mask for the current batch.
        s_idxL_pos = interface_1[b]
        s_idxR_pos = interface_2[b]
        # Unique interface site indices per surface.
        s_site_idxL = s_idxL_pos.unique()
        s_site_idxR = s_idxR_pos.unique()
        
        # Create dense binary masks for interface sites.
        denseL = torch.zeros(surface_L.verts[surface_L.batch==b].shape[0], dtype=torch.uint8, device=surface_L.verts.device)
        denseR = torch.zeros(surface_R.verts[surface_R.batch==b].shape[0], dtype=torch.uint8, device=surface_R.verts.device)
        denseL[s_site_idxL] = 1 
        denseR[s_site_idxR] = 1
        
        # Negative site indices.
        s_site_idxL_neg = (denseL == 0).nonzero(as_tuple=True)[0]
        s_site_idxR_neg = (denseR == 0).nonzero(as_tuple=True)[0]
        
        # Sample negatives for sites.
        site_neg_num = min(pos_mult * len(s_site_idxL), len(s_site_idxL_neg),
                           pos_mult * len(s_site_idxR), len(s_site_idxR_neg))
        if site_neg_num > 0:
            perm_L = torch.randperm(len(s_site_idxL_neg))[:site_neg_num]
            perm_R = torch.randperm(len(s_site_idxR_neg))[:site_neg_num]
            s_site_neg_sampleL = s_site_idxL_neg[perm_L]
            s_site_neg_sampleR = s_site_idxR_neg[perm_R]
        else:
            s_site_neg_sampleL = torch.empty(0, dtype=torch.long, device=surface_L.verts.device)
            s_site_neg_sampleR = torch.empty(0, dtype=torch.long, device=surface_R.verts.device)
        
        s_site_idxL_sample = torch.cat([s_site_idxL, s_site_neg_sampleL])
        s_site_idxR_sample = torch.cat([s_site_idxR, s_site_neg_sampleR])
        s_site_label_L = torch.cat([torch.ones(len(s_site_idxL), device=surface_L.verts.device),
                                    torch.zeros(len(s_site_neg_sampleL), device=surface_L.verts.device)])
        s_site_label_R = torch.cat([torch.ones(len(s_site_idxR), device=surface_R.verts.device),
                                    torch.zeros(len(s_site_neg_sampleR), device=surface_R.verts.device)])
        
        # For pairwise contact sampling: sample negatives without building a huge dense matrix.
        neg_num = pos_mult * len(s_idxL_pos)
        neg_idxL_sample, neg_idxR_sample = sample_negative_pairs_fast(
            s_idxL_pos, s_idxR_pos,
            total_left=surface_L.verts[surface_L.batch==b].shape[0],
            total_right=surface_R.verts[surface_R.batch==b].shape[0],
            neg_num=neg_num,
            device=surface_L.verts.device
        )
        
        sidx_left = torch.cat([s_idxL_pos, neg_idxL_sample])
        sidx_right = torch.cat([s_idxR_pos, neg_idxR_sample])
        s_label = torch.cat([torch.ones(len(s_idxL_pos), device=surface_L.verts.device),
                             torch.zeros(len(neg_idxL_sample), device=surface_L.verts.device)])
        
        results['s_idx_left'].append(sidx_left)
        results['s_idx_right'].append(sidx_right)
        results['s_label'].append(s_label)
        results['s_site_idxL_sample'].append(s_site_idxL_sample)
        results['s_site_idxR_sample'].append(s_site_idxR_sample)
        results['s_site_label_L'].append(s_site_label_L)
        results['s_site_label_R'].append(s_site_label_R)
        results['s1_len'].append(surface_L.verts[surface_L.batch==b].shape[0])
        results['s2_len'].append(surface_R.verts[surface_R.batch==b].shape[0])
    return results
class PINDERModule_seed_onfly(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters()
        # self.criterion = torch.nn.BCEWithLogitsLoss()  # pos_weight=torch.tensor([hparams.model.pos_weight])
        self.model = PINDERNet_seed(hparams_encoder=cfg.encoder, hparams_head=cfg.cfg_head)
        self.save_embed= False # cfg.save_embed
        self.save_embed_dir =None # cfg.save_embed_dir
        

    def step(self, batch,save_emb=False):
        runtype = 'both'
        import time
        t0=time.time()
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None

        ### pre process surface
        points_1, normals_1, batch_points_1 = atoms_to_points_normals(batch.graph_1.node_pos,batch.graph_1.batch,distance=1.05,smoothness=0.5,resolution=2.5,nits=4,atomtypes=batch.graph_1.x[:,-12:],sup_sampling=20, variance=0.1,)
        points_2, normals_2, batch_points_2 = atoms_to_points_normals(batch.graph_2.node_pos,batch.graph_2.batch,distance=1.05,smoothness=0.5,resolution=2.5,nits=4,atomtypes=batch.graph_2.x[:,-12:],sup_sampling=20, variance=0.1,)

        P_curvatures_1 = curvatures(points_1,triangles= None,normals= normals_1,scales=[1.0, 2.0, 3.0, 5.0, 10.0],batch=batch_points_1,)
        P_curvatures_2 = curvatures(points_2,triangles= None,normals= normals_2,scales=[1.0, 2.0, 3.0, 5.0, 10.0],batch=batch_points_2,)

        print('calculate surface cost:',t0-time.time())
        t0=time.time()

        surface_data_list_1 = []
        for b in batch_points_1.unique():
            data_obj = Data(
                verts=points_1[batch_points_1==b],
                n_verts=len(points_1[batch_points_1==b]),
                x=P_curvatures_1[batch_points_1==b],
                num_nodes=len(points_1[batch_points_1==b]),
                vnormals=normals_1[batch_points_1==b]
            )
            surface_data_list_1.append(data_obj)
        surface_data_list_2 = []
        for b in batch_points_2.unique():
            data_obj = Data(
                verts=points_2[batch_points_2==b],
                n_verts=len(points_2[batch_points_2==b]),
                x=P_curvatures_2[batch_points_2==b],
                num_nodes=len(points_2[batch_points_2==b]),
                vnormals=normals_2[batch_points_2==b]
            )
            surface_data_list_2.append(data_obj)
        from torch_geometric.data import Batch
        surface_batch_1 = Batch.from_data_list(surface_data_list_1) 
        surface_batch_2 = Batch.from_data_list(surface_data_list_2) 
        batch.surface_1=surface_batch_1
        batch.surface_2=surface_batch_2
        ## cant get the interface index , give it up just use the for loop
        # interface_idx1, interface_idx2 = get_interface_pairs_pykeops(points_1, batch_points_1,points_2, batch_points_2,threshold=2.0)
        interface_1,interface_2  = get_interface_and_pair_indices(points_1, batch_points_1, points_2, batch_points_2, threshold=3.0)
        s_result = process_interface_list(interface_1, interface_2, batch_points_1,surface_batch_1, surface_batch_2,pos_mult=200)
        batch.s_label = s_result['s_label']
        batch.s_idx_left = s_result['s_idx_left']
        batch.s_idx_right = s_result['s_idx_right']
        batch.s_site_idxL = s_result['s_site_idxL_sample']
        batch.s_site_idxR = s_result['s_site_idxR_sample']
        batch.s_site_label_L= s_result['s_site_label_L']
        batch.s_site_label_R= s_result['s_site_label_R']
        batch.s1_len = torch.tensor(s_result['s1_len']).to(points_1.device)
        batch.s2_len = torch.tensor(s_result['s2_len']).to(points_1.device)
        print('create new batch:',t0-time.time())
        t0=time.time()
        if isinstance(batch.label_l, list):
            labels_l = torch.cat(batch.label_l).reshape(-1, 1)
            labels_r = torch.cat(batch.label_r).reshape(-1, 1)
            s_site_label_L = torch.cat(batch.s_site_label_L).reshape(-1, 1)
            s_site_label_R = torch.cat(batch.s_site_label_R).reshape(-1, 1)
        else:
            labels_l = batch.label_l.reshape(-1, 1)
            labels_r = batch.label_r.reshape(-1, 1)
            s_site_label_L = batch.s_site_label_L.reshape(-1, 1)
            s_site_label_R = batch.s_site_label_R.reshape(-1, 1)
        outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair,emb1,emb2,emb1_inv,emb2_inv,s_emb1,s_emb2,s_emb1_inv,s_emb2_inv = self.model(batch,save_emb)
        if runtype=='both':
            labels = torch.cat([labels_l,labels_r])
            loss_site = compute_BCE(outputs_site, labels)
            s_site_label = torch.cat([s_site_label_L,s_site_label_R])
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label)
        elif runtype=='l':
            labels= labels_l
            loss_site = compute_BCE(outputs_site, labels)
            s_site_label = s_site_label_L
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_L)
        elif runtype=='r':
            labels= labels_r
            s_site_label = s_site_label_R
            loss_site = compute_BCE(outputs_site, labels)
            loss_surf_site = compute_BCE(outputs_surface_site, s_site_label_R)
        if isinstance(batch.labels_pair, list):
            label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.labels_pair)]).to(batch.labels_pair[0].device)
            labels_pair = torch.cat(batch.labels_pair).reshape(-1, 1)
        else:
            labels_pair = batch.labels_pair.reshape(-1, 1)
        if isinstance(batch.s_label, list):
            s_label_batch = torch.cat([torch.tensor([i] * len(labels)) for i, labels in enumerate(batch.s_label)]).to(batch.s_label[0].device)
            s_label = torch.cat(batch.s_label).reshape(-1, 1)
        else:
            s_label = batch.s_label.reshape(-1, 1)
        loss_pair = compute_BCE(outputs_pair, labels_pair)
        loss_surface_pair= compute_BCE(outputs_surface_pair, s_label)
        ## add complementarity loss, dot product for similarity
        ## old loss for complementary

        complementray =  'euclidean' #'cosine'
        if complementray=='cosine':
            dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
            loss_complementartiy_g = compute_BCE(dists_1_2, labels_pair) + compute_BCE(dists_2_1, labels_pair) 

            s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
            loss_complementartiy_s = compute_BCE(s_dists_1_2, s_label) + compute_BCE(s_dists_2_1, s_label)
        elif complementray=='euclidean':
            emb1 = F.normalize(emb1, dim=1)
            emb1_inv = F.normalize(emb1_inv, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            emb2_inv = F.normalize(emb2_inv, dim=1)
            s_emb1 = F.normalize(s_emb1, dim=1)
            s_emb1_inv = F.normalize(s_emb1_inv, dim=1)
            s_emb2 = F.normalize(s_emb2, dim=1)
            s_emb2_inv = F.normalize(s_emb2_inv, dim=1)

            cos_dists_1_2= F.cosine_similarity(emb1,emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            cos_dists_2_1= F.cosine_similarity(emb2,emb1_inv).reshape(-1, 1)
            loss_complementartiy_g_cos = compute_BCE(cos_dists_1_2, labels_pair) + compute_BCE(cos_dists_2_1, labels_pair) 
            cos_s_dists_1_2= F.cosine_similarity(s_emb1,s_emb2_inv).reshape(-1, 1) # in dmasif use torch.matmul 
            cos_s_dists_2_1= F.cosine_similarity(s_emb2,s_emb1_inv).reshape(-1, 1)
            loss_complementartiy_s_cos = compute_BCE(cos_s_dists_1_2, s_label) + compute_BCE(cos_s_dists_2_1, s_label)
            def contrastive_loss(distances, labels, margin=1.0, pos_weight=1.0):
                loss_pos = pos_weight * labels * distances.pow(2)
                loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
                loss = loss_pos + loss_neg
                return loss.mean()
            def cos_contrastive_loss(similarities, labels, margin=-0.5, pos_weight=1.0):
                loss_pos = pos_weight * labels * (1 - similarities).pow(2)  # Pull positives to 1
                loss_neg = (1 - labels) * F.relu(similarities - margin).pow(2)  # Push negatives below margin
                return (loss_pos + loss_neg).mean()
            
            dists_1_2 = torch.norm(emb1 - emb2_inv, p=2, dim=1, keepdim=True)
            dists_2_1 = torch.norm(emb2 - emb1_inv, p=2, dim=1, keepdim=True)

            # Apply contrastive loss for ground truth pairs
            loss_complementartiy_g = contrastive_loss(dists_1_2, labels_pair,margin=2.0) + contrastive_loss(dists_2_1, labels_pair,margin=2.0)+cos_contrastive_loss(cos_dists_1_2, labels_pair)+cos_contrastive_loss(cos_dists_2_1, labels_pair)
            
            # Compute Euclidean distances for the secondary embeddings
            s_dists_1_2 = torch.norm(s_emb1 - s_emb2_inv, p=2, dim=1, keepdim=True)
            s_dists_2_1 = torch.norm(s_emb2 - s_emb1_inv, p=2, dim=1, keepdim=True)

            # Apply contrastive loss for secondary pairs
            loss_complementartiy_s = contrastive_loss(s_dists_1_2, s_label,margin=2.0,pos_weight=50.0) + contrastive_loss(s_dists_2_1, s_label,margin=2.0,pos_weight=50.0)+cos_contrastive_loss(cos_s_dists_1_2, s_label,pos_weight=50.0)+cos_contrastive_loss(cos_s_dists_2_1, s_label,pos_weight=50.0)

        loss = loss_site + loss_pair +loss_surf_site+loss_surface_pair+ loss_complementartiy_g + loss_complementartiy_s + 5*(loss_complementartiy_g_cos+ loss_complementartiy_s_cos)
        # TODO: change loss pass in return and in each step logging
        if torch.isnan(loss_site).any() or torch.isnan(loss_pair).any() or torch.isnan(loss_surface_pair).any() :
            print('Nan loss')
            return None, None, None,None, None, None,None,None, None,None,None,None, None,None,None,None,None,None,None
        
        print('calculating loss:',t0-time.time())
        t0=time.time()
        
        return loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,cos_dists_1_2,cos_dists_2_1,cos_s_dists_1_2,cos_s_dists_2_1

    def training_step(self, batch, batch_idx):
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1= self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item(),"loss_site/train":loss_site.item(),"loss_pair/train":loss_pair.item(),"loss_surface_site/train":loss_surf_site.item(),"loss_surface/train":loss_surface_pair.item(),"loss_complementartiy_g/train":loss_complementartiy_g.item(),"loss_complementartiy_s/train":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)
        #
        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/train": acc_site, "auroc_site/train": auroc_site,"acc_pair/train":acc_pair,"auroc_pair/train":auroc_pair,'acc_surface_site/train':acc_surface_site,'auroc_surface_site/train':auroc_surface_site,"acc_surface/train":acc_surface,"auroc_surface/train":auroc_surface,'auroc_graph_seed1/train':auroc_graph_seed1,'auroc_graph_seed2/train':auroc_graph_seed2,'auroc_surface_seed1/train':auroc_surface_seed1,'auroc_surface_seed2/train':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch)
        if loss is None:
            print("validation step skipped!")
            self.log("auroc_val", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/val": loss.item(),"loss_site/val":loss_site.item(),"loss_pair/val":loss_pair.item(),"loss_surface/val":loss_surface_pair.item(),"loss_complementartiy_g/val":loss_complementartiy_g.item(),"loss_complementartiy_s/val":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)

        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/val": acc_site, "auroc_site/val": auroc_site,"acc_pair/val":acc_pair,"auroc_pair/val":auroc_pair,'acc_surface_site/val':acc_surface_site,'auroc_surface_site/val':auroc_surface_site,"acc_surface/val":acc_surface,"auroc_surface/val":auroc_surface,'auroc_graph_seed1/val':auroc_graph_seed1,'auroc_graph_seed2/val':auroc_graph_seed2,'auroc_surface_seed1/val':auroc_surface_seed1,'auroc_surface_seed2/val':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss,loss_site,loss_pair,loss_surf_site,loss_surface_pair,loss_complementartiy_g,loss_complementartiy_s, outputs_site,outputs_pair,outputs_surface_site,outputs_surface_pair, labels,labels_pair,s_site_label,s_label,dists_1_2,dists_2_1,s_dists_1_2,s_dists_2_1 = self.step(batch,save_emb=True)
        if loss is None:
            print("test step skipped!")
            self.log("auroc/test", 0.5, prog_bar=True, on_step=False, on_epoch=True, logger=False)
            return None
        self.log_dict({"loss/test": loss.item(),"loss_site/test":loss_site.item(),"loss_pair/test":loss_pair.item(),"loss_surface/test":loss_surface_pair.item(),"loss_complementartiy_g/test":loss_complementartiy_g.item(),"loss_complementartiy_s/test":loss_complementartiy_s.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(outputs_site))
        acc_site = compute_accuracy(outputs_site, labels, add_sigmoid=True)
        auroc_site = compute_auroc(outputs_site, labels)
        acc_pair = compute_accuracy(outputs_pair, labels_pair, add_sigmoid=True)
        auroc_pair = compute_auroc(outputs_pair, labels_pair)
        acc_surface_site = compute_accuracy(outputs_surface_site,s_site_label, add_sigmoid=True)
        auroc_surface_site = compute_auroc(outputs_surface_site,s_site_label)
        acc_surface = compute_accuracy(outputs_surface_pair, s_label, add_sigmoid=True)
        auroc_surface = compute_auroc(outputs_surface_pair, s_label)
        auroc_graph_seed1 = compute_auroc(dists_1_2, labels_pair)
        auroc_graph_seed2 = compute_auroc(dists_2_1, labels_pair)
        auroc_surface_seed1 = compute_auroc(s_dists_1_2, s_label)
        auroc_surface_seed2 = compute_auroc(s_dists_2_1, s_label)

        self.log_dict({"acc_site/test": acc_site, "auroc_site/test": auroc_site,"acc_pair/test":acc_pair,"auroc_pair/test":auroc_pair,'acc_surface_site/test':acc_surface_site,'auroc_surface_site/test':auroc_surface_site,"acc_surface/test":acc_surface,"auroc_surface/test":auroc_surface,'auroc_graph_seed1/test':auroc_graph_seed1,'auroc_graph_seed2/test':auroc_graph_seed2,'auroc_surface_seed1/test':auroc_surface_seed1,'auroc_surface_seed2/test':auroc_surface_seed2}, on_epoch=True, batch_size=len(outputs_site))
    
    def predict_step(self, batch, batch_idx: int):
        self.model.train()
        import h5py
        if self.save_embed:
            if batch is  None or batch.num_graphs < self.hparams.cfg.min_batch_size:
                print('None batch')
            else:
                graph_1,graph_2,surface_1,surface_2,g_site_L_pred,g_site_R_pred,s_site_L_pred,s_site_R_pred= self.model.extract_embedding(batch)
                save_dir = "/work/lpdi/users/ymiao/code/sbatch/pinderlog/"
                os.makedirs(save_dir, exist_ok=True)
                hdf5_path = os.path.join(save_dir, "embeddings_new.h5")
                #change it to  self.hparams.save_dir and hdf5_path later
                # Open HDF5 file in append mode

                graph_1_list= graph_1.to_data_list()
                graph_2_list= graph_2.to_data_list()
                surface_1_list= surface_1.to_data_list()
                surface_2_list= surface_2.to_data_list()
                
                with h5py.File(hdf5_path, "a") as hf:
                    for batch_idx in graph_1.batch.unique():
                        tmp_id2, tmp_id1 = batch.id[batch_idx].split('--')
                        # print(tmp_id2,tmp_id1)
                        # Create a single large group for everything
                        grp = hf.require_group("embedding")#hf.require_group(f"batch_{batch_idx}")
                        if f"graph/{tmp_id1}" not in grp:
                            tmp_graph= graph_1_list[batch_idx]
                            tmp_graph_d= {}
                            tmp_graph_d['edge_index']=tmp_graph.edge_index.cpu().numpy()
                            tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                            tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                            tmp_graph_d['emb']=graph_1.x_emb[graph_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['inv_emb']=graph_1.x_inv[graph_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['site_pred']=g_site_L_pred[graph_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['label']=batch.label_l[batch_idx].detach().cpu().numpy()
                            grp_graph1 = grp.require_group(f"graph/{tmp_id1}")
                            save_dict_to_group(grp_graph1, tmp_graph_d)
                            tmp_surface1 = surface_1_list[batch_idx]
                            tmp_surface1_d = {}
                            tmp_surface1_d['verts'] = tmp_surface1.verts.cpu().numpy() 
                            tmp_surface1_d['faces'] = tmp_surface1.faces.cpu().numpy() 
                            tmp_surface1_d['vnormals'] = tmp_surface1.vnormals.cpu().numpy() 
                            tmp_surface1_d['emb'] = surface_1.x_emb[surface_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface1_d['inv_emb'] = surface_1.x_inv[surface_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface1_d['site_pred'] = s_site_L_pred[surface_1.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface1_d['label'] = batch.s_site_label_L[batch_idx].detach().cpu().numpy()
                            grp_surface1 = grp.require_group(f"surface/{tmp_id1}")
                            save_dict_to_group(grp_surface1, tmp_surface1_d)

                        if f"graph/{tmp_id2}" not in grp:
                            tmp_graph= graph_2_list[batch_idx]
                            tmp_graph_d= {}
                            tmp_graph_d['edge_index'] = tmp_graph.edge_index.cpu().numpy()
                            tmp_graph_d['edge_attr']=tmp_graph.edge_attr.cpu().numpy()
                            tmp_graph_d['node_pos']=tmp_graph.node_pos.cpu().numpy()
                            tmp_graph_d['emb']=graph_2.x_emb[graph_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['inv_emb']=graph_2.x_inv[graph_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['site_pred']=g_site_R_pred[graph_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_graph_d['label']=batch.label_r[batch_idx].detach().cpu().numpy()
                            grp_graph2 = grp.require_group(f"graph/{tmp_id2}")
                            save_dict_to_group(grp_graph2, tmp_graph_d)
                            tmp_surface2 = surface_2_list[batch_idx]
                            tmp_surface2_d = {}
                            tmp_surface2_d['verts'] = tmp_surface2.verts.cpu().numpy() 
                            tmp_surface2_d['faces'] = tmp_surface2.faces.cpu().numpy() 
                            tmp_surface2_d['vnormals'] = tmp_surface2.vnormals.cpu().numpy() 
                            tmp_surface2_d['emb'] = surface_2.x_emb[surface_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface2_d['inv_emb'] = surface_2.x_inv[surface_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface2_d['site_pred'] = s_site_R_pred[surface_2.batch == batch_idx].detach().cpu().numpy()
                            tmp_surface2_d['label'] = batch.s_site_label_R[batch_idx].detach().cpu().numpy()
                            grp_surface2 = grp.require_group(f"surface/{tmp_id2}")
                            save_dict_to_group(grp_surface2, tmp_surface2_d)
                print(f"Saved batch {batch_idx} embeddings to {hdf5_path}")
        else:
            graph_1,graph_2,surface_1,surface_2,g_site_L_pred,g_site_R_pred,s_site_L_pred,s_site_R_pred= self.model.extract_embedding(batch)
            return graph_1,graph_2
