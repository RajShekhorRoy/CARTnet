#LATEST MODEL last best
#DONT TOUCH
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from model.helper.losses import dice_loss_targetwise
from model.helper.rotary import RotaryEmbedding
from model.helper.focal_loss import FocalLoss
from model.helper.block import RecycleAttentionBlock
from sklearn.metrics import roc_auc_score, average_precision_score


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class CARTnet(pl.LightningModule):
    def __init__(self,fold= 0):
        super().__init__()
        d_common =256
        self.fold = fold
        self.hidden_dim=256
        ID = "TEMP_RESNET_clust40_{0}".format(fold)
        self.fusion_proj_d_h = nn.Linear(d_common ,       self.hidden_dim)
        self.fusion_proj_d_h_2 = nn.Linear(      self.hidden_dim,       self.hidden_dim)
        self.fusion_proj_h_1 = nn.Linear(      self.hidden_dim , 1)
        self.save_hyperparameters()
        self.apply(init_weights)
        self.attention_head_size =16
        self.MAX_LENGTH=1400
        # self.cross_attn = self.CrossAttnBlock(128, 128, 128).cuda()
        self.classifier = nn.Sigmoid( )
        # binary classification
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.loss_fn = FocalLoss(gamma=2,alpha=0.7, task_type='binary', reduction="none")
        # self.instance = nn.InstanceNorm2d( 1, affine=True)
        self.rotary_embeddings = RotaryEmbedding(dim=      self.hidden_dim)
        self.rotary_embeddings_mlp = RotaryEmbedding(dim=      self.hidden_dim)
        self.rotary_embeddings_v = RotaryEmbedding(dim=      self.hidden_dim)
        # self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=1198)

        self.track_esm = nn.Sequential(
            nn.LayerNorm(1537),
            nn.Linear(1537, 512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
            nn.Linear(512,self.hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1)
        )

        self.mlp = nn.Sequential(
            nn.Linear( self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

        self.recycle_blocks = nn.ModuleList([
            RecycleAttentionBlock(hidden_dim=256, feat_dim=32)
            for _ in range(10)
        ])

    def make_loss_mask(sefl,lengths, max_len):
        try:
            device = lengths.device
        except:
            device = lengths[0].device
        return (
                torch.arange(max_len, device=device)[None, :] < lengths[:, None]        ).float()

    def forward(self, x2,x3,x4,x5):
        x5= torch.tensor(x5, dtype=torch.float32)
        x2 = x2.permute(0, 1, 3, 2)
        K= Q = V = self.track_esm(x2)
        for block in self.recycle_blocks:
            x5 = block(V, x3, x4, x5)
        return  x5.squeeze()

    def on_train_epoch_start(self):
        self.train_losses = []


    def training_step(self, batch, batch_idx):
        x2 = batch['esm']#[B,1, L,2560]
        x3 = batch['feat']
        x4 = batch['intra']
        x5 = batch['temp']
        y = batch['ground_truth'].float()
        lengths = batch ['length']
        logits = self(x2,x3,x4,x5)
        y = y.reshape(logits.shape)
        lengths = batch['length']
        try :
            loss = self.loss_fn(logits, y.float())
            temp_loss_mask = batch['temp_loss_mask']
            temp_loss_mask = temp_loss_mask.reshape(logits.shape)
            n_loss =  (temp_loss_mask * loss).sum() / (temp_loss_mask.sum()+0.000000001)
            mask = self.make_loss_mask(lengths, 1400)
            loss = (loss * mask).sum() / mask.sum()
            dice = dice_loss_targetwise(logits,  y.float(), mask)
            loss=   loss+ 0.05 *dice + 0.2 *n_loss
        except :
            print("error")
        self.val_losses.append(float(f"{loss.item():.4f}"))
        self.log('train_loss',float(f"{loss.item():.4f}"), on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses.append(float(f"{loss.item():.4f}"))
        return loss

    def on_validation_epoch_start(self):
        self.val_outputs = []
        self.val_losses = []
        self.val_lengths = []

    def validation_step(self, batch, batch_idx):
        x2 = batch['esm']  # [B,1, L,2560]
        x3 = batch['feat']
        x4 = batch['intra']
        y = batch['ground_truth'].float()
        lengths = batch['length']
        x5 = batch['temp']
        logits = self( x2, x3,x4,x5)
        y = y.reshape(logits.shape)
        self.val_lengths.extend(lengths.tolist())
        try :
            loss =self.loss_fn(logits, y.float())
            mask = self.make_loss_mask(lengths, 1400)
            loss = (loss * mask).sum() / mask.sum()
            temp_loss_mask = batch['temp_loss_mask']
            temp_loss_mask = temp_loss_mask.reshape(logits.shape)
            n_loss =  (temp_loss_mask * loss).sum() / (temp_loss_mask.sum() +0.000000001)
            dice = dice_loss_targetwise(logits,  y.float(), mask)
            loss=loss+0.05 *dice+ 0.2 * n_loss
        except :
            print("error")
        desired_shape = batch['ground_truth'].squeeze().shape
        if len(desired_shape)==1:
            desired_shape = batch['ground_truth'].squeeze().unsqueeze(dim=0).shape
        self.val_losses.append(float(f"{loss.item():.6f}"))
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.val_outputs.append({"probs":torch.reshape(logits.detach().cpu(),desired_shape), "targets": torch.reshape(batch['ground_truth'],desired_shape),"name":batch["name"]})
        return {"probs": torch.reshape(logits.detach().cpu(),desired_shape), "targets":torch.reshape(batch['ground_truth'],desired_shape) ,"name":batch["name"]}

    def append_to_txt(self,file_path, text):
        with open(file_path, "a") as f:
            f.write(text + "\n")

    def get_top_results(self,_pred,_target,_k_values,_lengths):
        array_result = []
        for _y,_y_hat,_k,_l in zip(_target,_pred,_k_values,_lengths):
            one_pred = np.asarray(_y_hat).reshape(-1)[0:_l]
            one_target = np.asarray(_y).reshape(-1)[0:_l]
            # Get indices of top-K predictions (descending)
            topk_idx = np.argsort(one_pred)[- _k:][::-1]
            # Count positives in top-K
            tp_at_k = one_target[topk_idx].sum()
            precision_at_k = tp_at_k / _k
            array_result.append(precision_at_k)
        return np.mean(array_result)

    def on_validation_epoch_end(self):
        mask = self.make_loss_mask(torch.tensor(self.val_lengths),  self.MAX_LENGTH).bool()
        probs = torch.cat([o["probs"] for o in self.val_outputs])
        targets = torch.cat([o["targets"] for o in self.val_outputs]).bool().cpu()
        # ---- APPLY MASK ----
        probs_m = probs[mask].detach().cpu().numpy()
        targets_m = targets[mask].detach().cpu().numpy().astype(int)
        preds = (probs > 0.3).bool().cpu()
        correct = (preds == targets) & mask
        acc = correct.sum().float() / mask.sum()
        # TP / FP / FN
        tp = (preds & targets & mask).sum().float()
        fp = (preds & ~targets & mask).sum().float()
        fn = (~preds & targets & mask).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        auc = roc_auc_score(targets_m, probs_m)
        auprc = average_precision_score(targets_m, probs_m)
        l5_k = (np.asarray(self.val_lengths)/5).astype(int)
        top_10 = self.get_top_results(_pred=probs,_target=targets,_k_values=np.full(len(self.val_lengths), 10, dtype=int),_lengths=self.val_lengths)
        top_l5 = self.get_top_results(_pred=probs, _target=targets,_k_values=l5_k, _lengths=self.val_lengths)
        t_loss= 0
        if    len(self.train_losses)==0:
            t_loss=0
        else:
            t_loss = np.asarray(self.train_losses).mean()

        self.log_dict({
            "train_loss":t_loss ,
            "val_loss": np.asarray(self.val_losses).mean(),
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_auc": auc,
            "val_auprc": auprc
        }, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_outputs = []
        self.test_lengths = []

    def test_step(self, batch, batch_idx):
        x2 = batch['esm']  # [B,1, L,2560]
        x3 = batch['feat']
        x4 = batch['intra']
        y = batch['ground_truth'].float()
        x5 = batch['temp']
        logits = self( x2, x3,x4,x5)
        y = y.reshape(logits.shape)
        lengths = batch['length']
        self.test_lengths.extend(lengths.tolist())
        try:
            loss = self.loss_fn(logits, y)
            mask = self.make_loss_mask(lengths,  self.MAX_LENGTH)
            loss = (loss * mask).sum() / mask.sum()
            dice = dice_loss_targetwise(logits,  y.float(), mask)
            loss=loss+dice
        except:
            print("error")

        desired_shape = batch['ground_truth'].squeeze().shape
        if len(desired_shape) == 1:
            desired_shape = batch['ground_truth'].squeeze().unsqueeze(dim=0).shape

        self.test_losses.append(float(f"{loss.item():.6f}"))
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.test_outputs.append({"probs": torch.reshape(logits.detach().cpu(), desired_shape),
                                  "targets": torch.reshape(batch['ground_truth'], desired_shape),
                                  "name": batch["name"]})
        # print({"probs": logits.detach().cpu(), "targets": y.detach().cpu()})
        return {"probs": torch.reshape(logits.detach().cpu(), desired_shape),
                "targets": torch.reshape(batch['ground_truth'], desired_shape), "name": batch["name"]}


    def on_test_epoch_end(self):
        mask = self.make_loss_mask(torch.tensor(self.test_lengths), self.MAX_LENGTH).bool()
        probs = torch.cat([o["probs"] for o in self.test_outputs])
        targets = torch.cat([o["targets"] for o in self.test_outputs]).bool().cpu()
        # ---- APPLY MASK ----
        probs_m = probs[mask].detach().cpu().numpy()
        targets_m = targets[mask].detach().cpu().numpy().astype(int)
        preds = (probs > 0.3).bool().cpu()
        correct = (preds == targets) & mask
        acc = correct.sum().float() / mask.sum()
        # TP / FP / FN
        tp = (preds & targets & mask).sum().float()
        fp = (preds & ~targets & mask).sum().float()
        fn = (~preds & targets & mask).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        auc = roc_auc_score(targets_m, probs_m)
        auprc = average_precision_score(targets_m, probs_m)
        l5_k = (np.asarray(self.test_lengths) / 5).astype(int)
        top_10 = self.get_top_results(_pred=probs, _target=targets,
                                      _k_values=np.full(len(self.val_lengths), 10, dtype=int),
                                      _lengths=self.val_lengths)
        top_l5 = self.get_top_results(_pred=probs, _target=targets, _k_values=l5_k, _lengths=self.val_lengths)
        t_loss = 0
        if len(self.train_losses) == 0:
            t_loss = 0
        else:
            t_loss = np.asarray(self.train_losses).mean()
        self.log_dict({
            "train_loss": t_loss,
            "val_loss": np.asarray(self.val_losses).mean(),
            "val_acc": acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "val_auc": auc,
            "val_auprc": auprc
        }, prog_bar=True)

    def predict_one_sample(self, feat_tensor: torch.Tensor):
        self.eval()
        with torch.no_grad():
            if len(feat_tensor.shape) == 2:
                feat_tensor = feat_tensor.unsqueeze(0)  # → [1, L, 2593]
            feat_tensor = feat_tensor.to(self.device)
            logits = self(feat_tensor)
            prob = torch.sigmoid(logits).squeeze(0)
            pred = (prob > 0.5).int()
            return {"logits": logits.cpu(), "prob": prob.cpu(), "pred": pred.cpu()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-5,
            weight_decay=1e-5
        )
        return optimizer