"""
ssl_dinov2.py — DINOv2 Self-Distillation for Domain-Adaptive Continual Pre-training.

Implements the student-teacher self-distillation framework used by DINOv2,
adapted for continual pre-training on medical chest X-ray images.

Architecture:
    Teacher (EMA of student) ← receives global crops only
    Student                  ← receives global + local crops
    Both share the same backbone architecture but have separate weights.
    The student is trained to match the teacher's output distribution.

Key insight: By starting from the original DINOv2 weights (trained on natural
images) and continuing pre-training on chest X-rays, the backbone learns to
recognize medical-domain features (rib patterns, opacity textures, lung boundaries)
while retaining general visual understanding.
"""

import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """
    Projection + prototyping head for DINO self-distillation.

    Maps backbone features to a lower-dimensional space where the
    self-distillation loss is computed.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
    ):
        super().__init__()
        # 3-layer MLP projection head
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # Prototype layer (weight-normalized)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """
    Cross-entropy loss between teacher and student distributions.

    The teacher output is centered (mean-subtracted) to prevent mode collapse,
    and sharpened with a low temperature. The student uses a higher temperature.
    """

    def __init__(
        self,
        out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # Running mean of teacher outputs to prevent collapse
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: list[torch.Tensor],
        teacher_output: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute DINO loss.

        Args:
            student_output: List of student logits from all crops.
            teacher_output: List of teacher logits from global crops only.

        Returns:
            Scalar loss tensor.
        """
        # Teacher: sharpen + center
        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_output
        ]

        # Student: soften
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_output
        ]

        # Cross-entropy: each teacher view vs each student view (excluding same view)
        total_loss = 0.0
        n_loss_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                # Skip matching global indices (student should not just copy itself)
                if s_idx == t_idx:
                    continue
                loss = -torch.sum(t_prob * s_log_prob, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center with exponential moving average
        self._update_center(teacher_output)

        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output: list[torch.Tensor]) -> None:
        """Update the running mean (center) of teacher outputs."""
        batch_center = torch.cat(teacher_output, dim=0).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class _HFBackboneWrapper(nn.Module):
    """
    Wrapper to make a HuggingFace Dinov2 backbone behave like the Facebook
    DINOv2 backbone (return CLS token features as [B, embed_dim]).
    """

    def __init__(self, hf_backbone, embed_dim: int):
        super().__init__()
        self.backbone = hf_backbone
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        # HuggingFace returns (last_hidden_state, pooler_output) or similar
        if hasattr(outputs, "last_hidden_state"):
            # Take CLS token (first token)
            return outputs.last_hidden_state[:, 0]
        elif isinstance(outputs, (tuple, list)):
            return outputs[0][:, 0]
        else:
            return outputs[:, 0]


class SSLDINOv2(nn.Module):
    """
    Full DINOv2 self-distillation system for continual pre-training.

    Wraps a DINOv2 backbone with student-teacher architecture and
    projection heads.

    Args:
        backbone_name: Name of the DINOv2 backbone (e.g., 'dinov2_vits14_reg').
        projection_dim: Output dimension of the projection head bottleneck.
        hidden_dim: Hidden dimension of the projection head MLP.
        out_dim: Number of prototypes (output dimension of the final layer).
        teacher_momentum: EMA momentum for updating teacher from student.
        backbone_source: 'torchhub' (Facebook DINOv2, patch14) or
                         'rfdetr' (extract from RF-DETR, patch16 — recommended for compatibility).
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14_reg",
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        out_dim: int = 65536,
        teacher_momentum: float = 0.996,
        backbone_source: str = "torchhub",
    ):
        super().__init__()
        self.teacher_momentum = teacher_momentum
        self.backbone_source = backbone_source

        if backbone_source == "rfdetr":
            # ================================================================
            # RECOMMENDED for full RF-DETR compatibility (patch_size=16)
            # Extract the DINOv2 backbone directly from RF-DETR
            # ================================================================
            print("[SSL] Loading backbone from RF-DETR (patch_size=16, HuggingFace DINOv2)")
            import rfdetr
            _tmp = rfdetr.RFDETRSmall()
            # Locate backbone inside RF-DETR
            raw_backbone = None
            for attr_path in ["model.backbone", "model.model.backbone"]:
                obj = _tmp
                try:
                    for attr in attr_path.split("."):
                        obj = getattr(obj, attr)
                    raw_backbone = obj
                    break
                except AttributeError:
                    continue
            
            if raw_backbone is None:
                raise RuntimeError("Could not extract backbone from RF-DETR model")
            
            # Ambil bagian intinya (ViT encoder)
            inner = raw_backbone
            if isinstance(inner, nn.Sequential) or hasattr(inner, "__getitem__"):
                try: inner = inner[0]
                except: pass
            
            if hasattr(inner, "encoder"):
                inner = inner.encoder
            elif hasattr(inner, "body"):
                inner = inner.body

            self.student_backbone = copy.deepcopy(inner)

            # Detect embed_dim from the backbone's state dict
            state_keys = list(self.student_backbone.state_dict().keys())
            for k in state_keys:
                if "cls_token" in k:
                    embed_dim = self.student_backbone.state_dict()[k].shape[-1]
                    break
            else:
                embed_dim = 384  # Default for ViT-S

            del _tmp  # Free memory
            print(f"[SSL] Extracted pure HF transformer with embed_dim={embed_dim}")

        else:
            # ================================================================
            # LEGACY: Load from Facebook's torch.hub (patch_size=14)
            # Key mapping will handle conversion during RF-DETR injection
            # ================================================================
            print(f"[SSL] Loading DINOv2 backbone: {backbone_name}")
            self.student_backbone = torch.hub.load(
                "facebookresearch/dinov2", backbone_name
            )
            embed_dim = self.student_backbone.embed_dim

        # Enable gradient checkpointing for memory savings on T4 GPUs
        if hasattr(self.student_backbone, 'set_grad_checkpointing'):
            self.student_backbone.set_grad_checkpointing(True)
        elif hasattr(self.student_backbone, 'blocks'):
            for block in self.student_backbone.blocks:
                block.use_checkpoint = True

        # Create projection head
        self.student_head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=projection_dim,
            out_dim=out_dim,
        )

        # Teacher is a deep copy (separate weights, no gradient)
        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = copy.deepcopy(self.student_head)

        # Freeze teacher (updated only via EMA)
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        # Loss
        self.criterion = DINOLoss(
            out_dim=out_dim,
            teacher_temp=0.04,
            student_temp=0.1,
        )

        print(f"[SSL] Backbone embed_dim={embed_dim}, projection_dim={projection_dim}")
        print(f"[SSL] Backbone source: {backbone_source}")

    def _extract_cls(self, outputs) -> torch.Tensor:
        """
        Aman mengekstrak representasi global dari berbagai bentuk output.
        Mendukung Feature Map spasial (GAP) maupun Sequence Token (CLS).
        """
        # 1. Jika objek HuggingFace asli
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state[:, 0]
        
        # 2. Jika tuple/list (seperti kebanyakan output backbone DETR)
        # Ambil feature map dari level paling akhir (resolusi fitur terdalam)
        if isinstance(outputs, (tuple, list)):
            out = outputs[-1]
        else:
            out = outputs
            
        # 3. Kupas tipe data 'NestedTensor' bawaan RF-DETR jika ada
        if hasattr(out, "tensors"):
            out = out.tensors
            
        # 4. Standardisasi Bentuk ke [Batch, Channel]
        if out.dim() == 4:
            # Jika Spasial [B, C, H, W] -> Lakukan Global Average Pooling
            return out.mean(dim=[2, 3])
        elif out.dim() == 3:
            # Jika Sequence [B, N, C] -> Ambil CLS token
            return out[:, 0]
            
        return out

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through student backbone + head."""
        outputs = self.student_backbone(x)
        features = self._extract_cls(outputs)
        return self.student_head(features)

    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through teacher backbone + head (no grad)."""
        with torch.no_grad():
            outputs = self.teacher_backbone(x)
            features = self._extract_cls(outputs)
            return self.teacher_head(features)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Compute DINO self-distillation loss for a batch.

        Args:
            batch: Dict with keys 'global_1', 'global_2', 'local_crops'.

        Returns:
            Scalar loss tensor.
        """
        global_1 = batch["global_1"]     # [B, C, H, W]
        global_2 = batch["global_2"]     # [B, C, H, W]
        local_crops = batch["local_crops"]  # [B, N, C, h, w]

        # Teacher: process global crops only
        teacher_out_1 = self.forward_teacher(global_1)
        teacher_out_2 = self.forward_teacher(global_2)
        teacher_outputs = [teacher_out_1, teacher_out_2]

        # Student: process all crops
        student_out_1 = self.forward_student(global_1)
        student_out_2 = self.forward_student(global_2)
        student_outputs = [student_out_1, student_out_2]

        # Process local crops
        B, N, C, h, w = local_crops.shape
        local_flat = local_crops.reshape(B * N, C, h, w)
        local_out = self.forward_student(local_flat)
        # Split back into individual local crop outputs
        local_out_split = local_out.reshape(B, N, -1)
        for i in range(N):
            student_outputs.append(local_out_split[:, i, :])

        return self.criterion(student_outputs, teacher_outputs)

    @torch.no_grad()
    def update_teacher(self, momentum: float | None = None) -> None:
        """Update teacher weights via exponential moving average of student."""
        m = momentum if momentum is not None else self.teacher_momentum

        for param_s, param_t in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)

        for param_s, param_t in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_t.data.mul_(m).add_((1 - m) * param_s.data)

    def get_backbone_state_dict(self) -> dict:
        """
        Extract the student backbone state dict for downstream use.

        This is what gets saved as 'dinov2_medical_adapted.pth' and later
        loaded into RF-DETR's DINOv2 backbone.
        """
        return self.student_backbone.state_dict()
