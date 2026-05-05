# -*- coding: utf-8 -*-
"""
CLIP adapters for classification tasks.

- CLIPSearchAdapter         : insertion / deletion scoring for greedy search
- CLIPGradientAdapter       : few-backward saliency for gradient-family methods
- CLIPXpliqueWrapper        : optional classification wrapper for D-HSIC
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from attribution_research.adapters.base import ForwardCounterMixin, SearchAdapter
from attribution_research.adapters.gradient import GradientAdapter


class CLIPSearchAdapter(SearchAdapter):
    """
    CLIP classifier wrapper for greedy / PhaseWin / map-based evaluation.

    The target is an integer class index into the zero-shot semantic feature
    matrix.  Scores follow the standard insertion / deletion formulation:

        gain = lambda1 * p(class | insertion_image)
             + lambda2 * (1 - p(class | deletion_image))
    """

    model_name = "clip"
    task_type = "classification"

    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        try:
            import clip as openai_clip
        except ImportError as e:
            raise ImportError(
                "openai-clip is required for CLIPSearchAdapter. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from e

        self._device = device
        self._model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=download_root,
        )
        self._model = self._model.float().eval()
        self.model_name = f"clip_{clip_type.replace('/', '_').lower()}"
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.batch_size = int(batch_size)

        self._semantic_features: Optional[torch.Tensor] = None
        if semantic_feature_path is not None:
            self.load_semantic_features(semantic_feature_path)

        self._source_image: Optional[np.ndarray] = None
        self._target_label: Optional[int] = None
        self._mean = torch.tensor(self._CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self._CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    def load_semantic_features(self, path: str) -> None:
        self._semantic_features = torch.load(path, map_location=self._device).float()

    def setup(
        self,
        image: np.ndarray,
        target: int,
        **kwargs,
    ) -> None:
        if self._semantic_features is None:
            raise RuntimeError(
                "Semantic features not loaded. "
                "Pass semantic_feature_path or call load_semantic_features(path)."
            )
        self._source_image = image.astype(np.uint8)
        self._target_label = int(target)

    def _preprocess_batch(self, images_bgr: np.ndarray) -> torch.Tensor:
        rgb = images_bgr[..., ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self._device)
        return (tensor - self._mean) / self._std

    def _compose_images(self, alpha: np.ndarray) -> np.ndarray:
        return (alpha * self._source_image[None].astype(np.float32)).astype(np.uint8)

    @torch.no_grad()
    def _class_scores(self, images_bgr: np.ndarray) -> torch.Tensor:
        batches = []
        for start in range(0, len(images_bgr), self.batch_size):
            batch = self._preprocess_batch(images_bgr[start : start + self.batch_size])
            feats = self._model.encode_image(batch).float()
            self.record_model_forward(int(batch.shape[0]))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ self._semantic_features.to(feats.device).T
            probs = torch.softmax(logits, dim=-1)[:, self._target_label]
            batches.append(probs)
        return torch.cat(batches, dim=0)

    def _build_masked_images(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        alpha = np.clip(masks.astype(np.float32) + baseline.astype(np.float32)[None], 0, 1)
        return self._compose_images(alpha)

    @torch.no_grad()
    def score_batch_detailed(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        alpha = np.clip(masks.astype(np.float32) + baseline.astype(np.float32)[None], 0, 1)
        img_ins = self._compose_images(alpha)
        img_del = self._compose_images(1.0 - alpha)
        ins = self._class_scores(img_ins)
        dele = self._class_scores(img_del)
        gain = self.lambda1 * ins + self.lambda2 * (1.0 - dele)
        return {
            "insertion_score": ins.detach().cpu(),
            "deletion_score": dele.detach().cpu(),
            "smdl_score": gain.detach().cpu(),
        }

    @torch.no_grad()
    def score_batch(
        self,
        masks: np.ndarray,
        baseline: np.ndarray,
    ) -> torch.Tensor:
        return self.score_batch_detailed(masks, baseline)["smdl_score"]

    @torch.no_grad()
    def score_single_detailed(
        self,
        mask: np.ndarray,
        baseline: np.ndarray,
    ) -> Dict[str, float]:
        details = self.score_batch_detailed(mask[None], baseline)
        return {
            "insertion_score": float(details["insertion_score"][0].item()),
            "deletion_score": float(details["deletion_score"][0].item()),
            "smdl_score": float(details["smdl_score"][0].item()),
        }


class CLIPGradientAdapter(GradientAdapter):
    """Input-gradient saliency for CLIP classification."""

    model_name = "clip"
    task_type = "classification"

    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        device: str = "cuda",
        score_mode: str = "prob",
    ):
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip is required for CLIPGradientAdapter. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        if score_mode not in {"prob", "logit"}:
            raise ValueError(f"Unsupported score_mode: {score_mode!r}")

        self._device = device
        self._model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=download_root,
        )
        self._model = self._model.float().eval()
        self._semantic_features: Optional[torch.Tensor] = None
        if semantic_feature_path is not None:
            self.load_semantic_features(semantic_feature_path)

        self._target_label: Optional[int] = None
        self._image: Optional[np.ndarray] = None
        self.score_mode = score_mode
        self._mean = torch.tensor(self._CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self._CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    def load_semantic_features(self, path: str) -> None:
        self._semantic_features = torch.load(path, map_location=self._device).float()

    def setup(self, image: np.ndarray, target: int, **kwargs) -> None:
        if self._semantic_features is None:
            raise RuntimeError(
                "Semantic features not loaded. "
                "Pass semantic_feature_path or call load_semantic_features(path)."
            )
        self._image = image.astype(np.uint8)
        self._target_label = int(target)

    def _preprocess_single(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return (tensor - self._mean) / self._std

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        image_tensor = self._preprocess_single(self._image).detach().clone().requires_grad_(True)
        self._model.zero_grad(set_to_none=True)
        feats = self._model.encode_image(image_tensor).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats @ self._semantic_features.to(feats.device).T
        if self.score_mode == "logit":
            score = logits[0, self._target_label]
        else:
            probs = torch.softmax(logits, dim=-1)
            score = probs[0, self._target_label]
        score.backward()

        grad = image_tensor.grad[0]
        saliency = (grad * image_tensor[0]).abs().mean(dim=0)
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class CLIPGradEClipAdapter(GradientAdapter):
    """Grad-ECLIP saliency for CLIP ViT backbones."""

    model_name = "clip"
    task_type = "classification"

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        device: str = "cuda",
        layer_span: int = 1,
    ):
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip is required for CLIPGradEClipAdapter. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        if int(layer_span) < 1:
            raise ValueError(f"layer_span must be >= 1, got {layer_span!r}")

        self._device = device
        self._model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=download_root,
        )
        self._model = self._model.float().eval()
        self._semantic_features: Optional[torch.Tensor] = None
        if semantic_feature_path is not None:
            self.load_semantic_features(semantic_feature_path)

        self._image: Optional[np.ndarray] = None
        self._target_label: Optional[int] = None
        self.layer_span = int(layer_span)
        self.model_name = f"clip_{clip_type.replace('/', '_').lower()}"

    @property
    def device(self) -> str:
        return self._device

    def load_semantic_features(self, path: str) -> None:
        self._semantic_features = torch.load(path, map_location=self._device).float()

    def setup(self, image: np.ndarray, target: int, **kwargs) -> None:
        if self._semantic_features is None:
            raise RuntimeError(
                "Semantic features not loaded. "
                "Pass semantic_feature_path or call load_semantic_features(path)."
            )
        self._image = image.astype(np.uint8)
        self._target_label = int(target)

    def _preprocess_single(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
        mean = torch.tensor(CLIPGradientAdapter._CLIP_MEAN, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
        std = torch.tensor(CLIPGradientAdapter._CLIP_STD, dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
        return (tensor - mean) / std

    @staticmethod
    def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
        tgt_len, batch, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        scale = float(head_dim) ** -0.5

        q = q * scale
        q = q.contiguous().view(tgt_len, batch * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch * num_heads, head_dim).transpose(0, 1)

        attn = torch.bmm(q, k.transpose(1, 2)).softmax(dim=-1)
        out = torch.bmm(attn, v)
        out = out.transpose(0, 1).contiguous().view(tgt_len, batch, embed_dim)
        return out

    @staticmethod
    def _normalize_01(array: np.ndarray) -> np.ndarray:
        array = array.astype(np.float32)
        array -= float(array.min())
        vmax = float(array.max())
        if vmax > 0:
            array /= vmax
        return array

    @staticmethod
    def _sim_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_cls = F.normalize(q[:1, 0, :].float(), dim=-1)
        k_patch = F.normalize(k[1:, 0, :].float(), dim=-1)
        cosine = (q_cls * k_patch).sum(dim=-1)
        cosine_min = cosine.min()
        cosine_max = cosine.max()
        if float(cosine_max - cosine_min) <= 1e-8:
            return torch.ones_like(cosine)
        return (cosine - cosine_min) / (cosine_max - cosine_min)

    def _clip_encode_dense(self, image_tensor: torch.Tensor):
        visual = self._model.visual
        vision_width = visual.transformer.width
        vision_heads = max(1, vision_width // 64)

        x = visual.conv1(image_tensor)
        feat_h, feat_w = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        class_embedding = visual.class_embedding.to(x.dtype)
        x = torch.cat(
            [class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype), x],
            dim=1,
        )

        pos_embedding = visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = visual.input_resolution // visual.conv1.kernel_size[0]
        pos_w = visual.input_resolution // visual.conv1.kernel_size[1]
        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        img_pos = F.interpolate(img_pos, size=(feat_h, feat_w), mode="bicubic", align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
        x = x + torch.cat((tok_pos[None, ...], img_pos), dim=1)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        blocks = list(visual.transformer.resblocks)
        prefix_blocks = blocks[:-self.layer_span]
        if prefix_blocks:
            x = torch.nn.Sequential(*prefix_blocks)(x)

        qs = []
        ks = []
        vs = []
        attn_outputs = []
        for block in blocks[-self.layer_span :]:
            x_in = x
            x_ln = block.ln_1(x_in)
            q, k, v = F.linear(x_ln, block.attn.in_proj_weight, block.attn.in_proj_bias).chunk(3, dim=-1)
            attn_output = self._manual_attention(q, k, v, num_heads=vision_heads)
            qs.append(q)
            ks.append(k)
            vs.append(v)
            attn_outputs.append(attn_output)

            x_after_attn = F.linear(attn_output, block.attn.out_proj.weight, block.attn.out_proj.bias)
            x = x_after_attn + x_in
            x = x + block.mlp(block.ln_2(x))

        x = x.permute(1, 0, 2)
        x = visual.ln_post(x)
        if visual.proj is not None:
            x = x @ visual.proj
        return x, qs, ks, vs, attn_outputs, (feat_h, feat_w)

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        image_tensor = self._preprocess_single(self._image).detach().clone().requires_grad_(True)
        self._model.zero_grad(set_to_none=True)

        outputs, qs, ks, vs, attn_outputs, map_size = self._clip_encode_dense(image_tensor)
        image_embedding = F.normalize(outputs[:, 0].float(), dim=-1)
        text_embedding = F.normalize(
            self._semantic_features[self._target_label : self._target_label + 1].to(image_embedding.device).float(),
            dim=-1,
        )
        score = (image_embedding * text_embedding).sum(dim=-1)[0]

        tmp_maps = []
        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            grad = torch.autograd.grad(score, attn_output, retain_graph=True)[0]
            grad_cls = grad[:1, 0, :].float()
            v_patch = v[1:, 0, :].float()
            cosine_qk = self._sim_qk(q, k).reshape(-1, 1)
            tmp_maps.append((grad_cls * v_patch * cosine_qk).sum(dim=-1))

        saliency = torch.relu(torch.stack(tmp_maps, dim=0)).sum(dim=0).reshape(*map_size)
        saliency = saliency.detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return self._normalize_01(saliency)


class CLIPIG2Adapter(GradientAdapter):
    """IG^2 saliency for CLIP classification using blur-based references."""

    model_name = "clip"
    task_type = "classification"

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        device: str = "cuda",
        steps: int = 32,
        step_size: float = 8.0,
        blur_sigmas: Optional[List[float]] = None,
        score_mode: str = "prob",
    ):
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip is required for CLIPIG2Adapter. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        if score_mode not in {"prob", "logit"}:
            raise ValueError(f"Unsupported score_mode: {score_mode!r}")
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")
        if float(step_size) <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size!r}")

        self._device = device
        self._model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=download_root,
        )
        self._model = self._model.float().eval()
        self._semantic_features: Optional[torch.Tensor] = None
        if semantic_feature_path is not None:
            self.load_semantic_features(semantic_feature_path)

        self._image: Optional[np.ndarray] = None
        self._target_label: Optional[int] = None
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.blur_sigmas = [float(s) for s in (blur_sigmas or [3.0, 7.0, 15.0, 31.0])]
        self.score_mode = score_mode
        self._rep_output: Optional[torch.Tensor] = None
        self._rep_handle = None
        ln_post = getattr(self._model.visual, "ln_post", None)
        if ln_post is not None:
            self._rep_handle = ln_post.register_forward_hook(self._capture_representation)
        self._mean = torch.tensor(CLIPGradientAdapter._CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIPGradientAdapter._CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    @property
    def device(self) -> str:
        return self._device

    def teardown(self) -> None:
        self._image = None
        self._target_label = None

    def load_semantic_features(self, path: str) -> None:
        self._semantic_features = torch.load(path, map_location=self._device).float()

    def setup(self, image: np.ndarray, target: int, **kwargs) -> None:
        if self._semantic_features is None:
            raise RuntimeError(
                "Semantic features not loaded. "
                "Pass semantic_feature_path or call load_semantic_features(path)."
            )
        self._image = image.astype(np.uint8)
        self._target_label = int(target)

    def _capture_representation(self, module, inputs, output) -> None:
        self._rep_output = inputs[0]

    def _raw_rgb_batch(self, images_bgr: np.ndarray) -> torch.Tensor:
        batch = np.asarray(images_bgr, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None]
        rgb = batch[..., ::-1].copy()
        return torch.from_numpy(rgb).permute(0, 3, 1, 2).to(self._device)

    def _forward_scores_and_rep(self, raw_rgb_255: torch.Tensor):
        self._rep_output = None
        normalized = (raw_rgb_255 / 255.0 - self._mean) / self._std
        features = self._model.encode_image(normalized).float()
        rep = self._rep_output
        if rep is None:
            # ModifiedResNet backbones do not expose ViT's ln_post hook point.
            # Fall back to the pre-normalization image embedding so IG2 remains
            # available for RN backbones under the same task interface.
            rep = features
        features = features / features.norm(dim=-1, keepdim=True)
        logits = features @ self._semantic_features.to(features.device).T
        if self.score_mode == "logit":
            scores = logits[:, self._target_label]
        else:
            scores = torch.softmax(logits, dim=-1)[:, self._target_label]
        return scores, rep

    def _normalize_batch_l2(self, grads: torch.Tensor) -> torch.Tensor:
        flat = grads.view(grads.shape[0], -1)
        norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return grads / norm.view(-1, 1, 1, 1)

    def _build_blur_references(self) -> np.ndarray:
        assert self._image is not None
        refs = []
        for sigma in self.blur_sigmas:
            kernel = max(3, int(round(sigma * 4)) | 1)
            refs.append(cv2.GaussianBlur(self._image, (kernel, kernel), sigmaX=sigma))
        return np.stack(refs, axis=0)

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        input_raw = self._raw_rgb_batch(self._image)
        references = self._raw_rgb_batch(self._build_blur_references())

        with torch.no_grad():
            _, baseline_rep = self._forward_scores_and_rep(references)
            baseline_rep = baseline_rep.detach()

        current = input_raw.repeat(references.shape[0], 1, 1, 1).detach()
        path = [current.detach().clone()]
        for _ in range(self.steps):
            current = current.detach().clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            _, rep = self._forward_scores_and_rep(current)
            loss = -torch.nn.functional.mse_loss(rep, baseline_rep, reduction="mean")
            loss.backward()
            grad = current.grad.detach()
            current = torch.clamp(current + self._normalize_batch_l2(grad) * self.step_size, 0.0, 255.0)
            path.append(current.detach().clone())

        attr = torch.zeros_like(current)
        previous = path[0]
        for step_tensor in path[1:]:
            prev = previous.detach().clone().requires_grad_(True)
            self._model.zero_grad(set_to_none=True)
            scores, _ = self._forward_scores_and_rep(prev)
            scores.sum().backward()
            grad = prev.grad.detach()
            attr += (prev.detach() - step_tensor.detach()) * grad
            previous = step_tensor

        saliency = attr.mean(dim=0).abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class CLIPIGOSPPAdapter(CLIPGradientAdapter):
    """IGOS++-style optimized mask saliency for CLIP classification."""

    method_name = "igos_pp"

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        device: str = "cuda",
        score_mode: str = "prob",
        mask_size: int = 28,
        steps: int = 24,
        lr: float = 0.1,
        blur_sigma: float = 15.0,
        preserve_coeff: float = 2.0,
        delete_coeff: float = 1.0,
        area_coeff: float = 0.01,
        tv_coeff: float = 0.2,
        binary_coeff: float = 0.01,
    ):
        if int(mask_size) < 2:
            raise ValueError(f"mask_size must be >= 2, got {mask_size!r}")
        if int(steps) < 1:
            raise ValueError(f"steps must be >= 1, got {steps!r}")
        if float(lr) <= 0:
            raise ValueError(f"lr must be > 0, got {lr!r}")
        if float(blur_sigma) <= 0:
            raise ValueError(f"blur_sigma must be > 0, got {blur_sigma!r}")

        super().__init__(
            clip_type=clip_type,
            semantic_feature_path=semantic_feature_path,
            download_root=download_root,
            device=device,
            score_mode=score_mode,
        )
        self.mask_size = int(mask_size)
        self.steps = int(steps)
        self.lr = float(lr)
        self.blur_sigma = float(blur_sigma)
        self.preserve_coeff = float(preserve_coeff)
        self.delete_coeff = float(delete_coeff)
        self.area_coeff = float(area_coeff)
        self.tv_coeff = float(tv_coeff)
        self.binary_coeff = float(binary_coeff)

    def _build_blur_reference(self, image: np.ndarray) -> np.ndarray:
        kernel = max(3, int(round(self.blur_sigma * 4)) | 1)
        return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=self.blur_sigma)

    def _target_score(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self._semantic_features is None:
            raise RuntimeError("Semantic features are not loaded")
        if self._target_label is None:
            raise RuntimeError("target label is not configured")
        feats = self._model.encode_image(pixel_values).float()
        self.record_model_forward(int(pixel_values.shape[0]))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = feats @ self._semantic_features.to(feats.device).T
        if self.score_mode == "logit":
            return logits[:, self._target_label].mean()
        probs = torch.softmax(logits, dim=-1)
        return probs[:, self._target_label].mean()

    @staticmethod
    def _tv_loss(mask: torch.Tensor) -> torch.Tensor:
        tv_h = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean()
        tv_w = (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    def saliency_map(self, **kwargs) -> np.ndarray:
        if self._image is None or self._target_label is None:
            raise RuntimeError("setup(image, target) must be called before saliency_map()")

        self.reset_forward_counter()
        orig_pixels = self._preprocess_single(self._image).detach()
        blur_pixels = self._preprocess_single(self._build_blur_reference(self._image)).detach()

        with torch.no_grad():
            orig_score = self._target_score(orig_pixels).detach()

        mask_logits = torch.zeros(
            (1, 1, self.mask_size, self.mask_size),
            dtype=torch.float32,
            device=orig_pixels.device,
            requires_grad=True,
        )
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
        eps = 1e-6

        for _ in range(self.steps):
            mask = torch.sigmoid(mask_logits)
            mask_up = F.interpolate(
                mask,
                size=orig_pixels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(orig_pixels.dtype)

            keep_pixels = mask_up * orig_pixels + (1.0 - mask_up) * blur_pixels
            drop_pixels = (1.0 - mask_up) * orig_pixels + mask_up * blur_pixels

            self._model.zero_grad(set_to_none=True)
            keep_score = self._target_score(keep_pixels)
            drop_score = self._target_score(drop_pixels)

            preserve_loss = torch.relu(orig_score - keep_score)
            area_loss = mask.mean()
            tv_loss = self._tv_loss(mask)
            entropy = -(mask * torch.log(mask + eps) + (1.0 - mask) * torch.log(1.0 - mask + eps)).mean()
            loss = (
                self.preserve_coeff * preserve_loss
                + self.delete_coeff * drop_score
                + self.area_coeff * area_loss
                + self.tv_coeff * tv_loss
                + self.binary_coeff * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_mask = torch.sigmoid(mask_logits.detach())
        final_mask = F.interpolate(
            final_mask,
            size=orig_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        saliency = final_mask.float().detach().cpu().numpy().astype(np.float32)
        saliency = cv2.resize(
            saliency,
            (self._image.shape[1], self._image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        saliency -= float(saliency.min())
        vmax = float(saliency.max())
        if vmax > 0:
            saliency /= vmax
        return saliency


class CLIPXpliqueWrapper(ForwardCounterMixin, torch.nn.Module):
    """
    Classification wrapper compatible with `xplique.wrappers.TorchWrapper`.

    It accepts either NHWC numpy arrays or NCHW/NHWC torch tensors and returns
    class probabilities.
    """

    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def __init__(
        self,
        clip_type: str = "ViT-L/14",
        semantic_feature_path: Optional[str] = None,
        download_root: Optional[str] = None,
        device: str = "cuda",
    ):
        super().__init__()
        try:
            import clip as openai_clip
        except ImportError as exc:
            raise ImportError(
                "openai-clip is required for CLIPXpliqueWrapper. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self.device = device
        self.model, _ = openai_clip.load(
            clip_type,
            device=device,
            download_root=download_root,
        )
        self.model = self.model.float().eval()
        if semantic_feature_path is None:
            raise ValueError("semantic_feature_path is required for CLIPXpliqueWrapper")
        self.semantic_features = torch.load(semantic_feature_path, map_location=device).float()
        self.num_classes = int(self.semantic_features.shape[0])
        self._mean = torch.tensor(self._CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self._CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def _prepare_inputs(self, inputs) -> torch.Tensor:
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs).to(self.device)
        else:
            tensor = inputs.to(self.device)

        if tensor.ndim != 4:
            raise ValueError(f"Expected 4-D inputs, got {tuple(tensor.shape)}")
        if tensor.shape[1] == 3:
            nchw = tensor.float()
        else:
            nchw = tensor.permute(0, 3, 1, 2).float()
        if float(nchw.detach().amax().item()) > 1.0:
            nchw = nchw / 255.0
        return (nchw - self._mean) / self._std

    def forward(self, vision_inputs):
        batch = self._prepare_inputs(vision_inputs)
        image_features = self.model.encode_image(batch).float()
        self.record_model_forward(int(batch.shape[0]))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.semantic_features.to(image_features.device).T
        return torch.softmax(logits, dim=-1).float()
