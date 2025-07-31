
import torch
import torch.nn as nn

# region DinoV2Info

# INPUT_IMAGE_SIZE = (1280, 720) # rescale to for DINOv2[1, 3, 224, 224]
# INPUT_TOKEN_SIZE = 1 * (1+256) * 768 # output of DINOv2 [1, 1 + 256, 768] 
# NUM_PATCHES = (16, 16) # DINOv2 patch size
# # source for numbers: https://huggingface.co/docs/transformers/model_doc/dinov2
# DINO_MODEL_NAME = 'dinov2_vitl14_reg'  # DINOv2 model name
# # cls_token = last_hidden_states[:, 0, :]
# # patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))

# endregion

#region OLDCODE



# class MLPGaze(nn.Module):
#     """
#     Simple MLP with no language-input, just a image input.
#     """
#     def __init__(self, dropout):
#         super().__init__()
#         self.dinov2 = torch.hub.load('facebookresearch/dinov2', DINO_MODEL_NAME)
#         self.fc: nn.Sequential = nn.Sequential(
#             nn.Linear(INPUT_TOKEN_SIZE),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         with torch.no_grad():
#             last_hidden_states = self.dinov2(x)
#             # Extract only the patch features and ignore the CLS token
#             # last_hidden_states shape: [batch_size, 1 + num_patches_height * num_patches_width, feature_size]
#             patch_features = last_hidden_states[:, 1:, :]
#             # Reshape patch features to match the expected input size for the fully connected layer
#             # (batch_size * num_patches_height * num_patches_width, feature_size)
#             patch_features = patch_features.view(-1, INPUT_TOKEN_SIZE)
#         return self.fc(patch_features)
    
# endregion


class MLP(nn.Module):
    """
    A 4-layer MLP mapping a feature vector of dimension `in_dim` to a scalar logit.
    Architecture: in_dim -> hidden_dims[0] -> hidden_dims[1] -> hidden_dims[2] -> 1
    Activations: ReLU between layers.
    """
    def __init__(self, in_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        dims = [in_dim] + hidden_dims + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        # returns: (B, N)
        logits = self.net(x)            # (B, N, 1)
        return logits.squeeze(-1)      # (B, N)

class GazePredictor(nn.Module):
    """
    Gaze prediction module that:
      1. Extracts visual tokens via a frozen DINOv2 backbone.
      2. Applies a single MLP in parallel to every feature vector (patch token).
      3. Computes spatial softmax and argmax to predict gaze patch index.

    Args:
        backbone_name (str): timm model name for DINOv2 (e.g., 'dino_vitb14').
        hidden_dims (list): hidden layer sizes for the MLP.
    """
    def __init__(
        self,
        # backbone_name: str = 'dino_vitb14',
        hidden_dims: list, dropout: float, repo: str, dino_model_name: str
    ):
        super().__init__()
        # 1) Load and freeze the DINOv2 backbone
        self.backbone: nn.Module = torch.hub.load(repo, dino_model_name) # type: ignore
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Determine feature dimension from backbone
        dummy = torch.randn(1, 3, 224, 224)
        seq = self.backbone(dummy)[-1]  # (1, D, Hf, Wf), should be (1, 1+16 * 16, 768) 
        _, token_count, dim = seq.shape
        print(f"Feature shape from backbone: {seq.shape}, token_count={token_count}, dim={dim}")

        # 2) Create single MLP head
        self.mlp = MLP(in_dim=dim, hidden_dims=hidden_dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 3, 224, 224) input images.
        Returns:
            (B, 16 * 16) spatial softmax probabilities
        """
        # Extract last feature map: (B, D, Hf, Wf)
        seq = self.backbone(x)
        # Drop CLS token
        feats = seq[:, 1:, :]       # (B, P, D)

        # 2) Apply MLP in parallel across patches: (B, P)
        logits = self.mlp(feats)

        # Spatial softmax over patches
        probs = torch.softmax(logits, dim=-1)
        return probs