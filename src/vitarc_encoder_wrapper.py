"""Tensor interface between prepared ARC samples and future ViTARC-style encoders.

The wrapper accepts samples produced by ``VitarcGridAdapter.prepare_dttt_sample``
and converts them into a small, explicit tensor batch. It intentionally does not
import or instantiate ViTARC by default, so tests and demos can run with only
PyTorch available.
"""

from __future__ import annotations

from typing import Any

import torch


PreparedGrid = dict[str, Any]
PreparedSample = dict[str, Any]
TensorBatch = dict[str, Any]


class VitarcEncoderWrapper:
    """Prepare padded ARC grids as tensors and optionally call an encoder model.

    Args:
        encoder_cls: Optional model class hook for future ViTARC experiments.
            It is stored but not instantiated unless ``model_kwargs`` is given.
        model: Optional already-created model object. If provided, ``forward``
            will call it with safe keyword inputs.
        device: Torch device for created tensors.
        grid_dtype: dtype for grid/color tensors. Defaults to ``torch.long``.
        mask_dtype: dtype for mask tensors. Defaults to ``torch.bool``.
        model_kwargs: Optional kwargs used to instantiate ``encoder_cls``.
    """

    def __init__(
        self,
        encoder_cls: type[Any] | None = None,
        model: Any | None = None,
        device: str | torch.device = "cpu",
        grid_dtype: torch.dtype = torch.long,
        mask_dtype: torch.dtype = torch.bool,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.encoder_cls = encoder_cls
        self.device = torch.device(device)
        self.grid_dtype = grid_dtype
        self.mask_dtype = mask_dtype

        if model is not None:
            self.model = model
        elif encoder_cls is not None and model_kwargs is not None:
            self.model = encoder_cls(**model_kwargs)
        else:
            self.model = None

        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def _grid_tensor(self, prepared_grid: PreparedGrid) -> torch.Tensor:
        return torch.as_tensor(
            prepared_grid["padded_grid"],
            dtype=self.grid_dtype,
            device=self.device,
        )

    def _mask_tensor(self, prepared_grid: PreparedGrid) -> torch.Tensor:
        return torch.as_tensor(
            prepared_grid["mask"],
            dtype=self.mask_dtype,
            device=self.device,
        )

    def _size_tensor(self, prepared_grid: PreparedGrid) -> torch.Tensor:
        return torch.as_tensor(
            prepared_grid["original_size"],
            dtype=torch.long,
            device=self.device,
        )

    def prepare_batch(self, prepared_sample: PreparedSample) -> TensorBatch:
        """Convert one prepared D_TTT sample into a batch of torch tensors.

        The returned tensors include a batch dimension of size 1. Context grids
        use shape ``[1, num_context, height, width]``. Query and target grids use
        shape ``[1, height, width]``.
        """
        context_pairs = prepared_sample["context_pairs"]
        context_inputs = torch.stack([self._grid_tensor(pair["input"]) for pair in context_pairs])
        context_outputs = torch.stack([self._grid_tensor(pair["output"]) for pair in context_pairs])
        context_input_masks = torch.stack([self._mask_tensor(pair["input"]) for pair in context_pairs])
        context_output_masks = torch.stack([self._mask_tensor(pair["output"]) for pair in context_pairs])
        context_input_sizes = torch.stack([self._size_tensor(pair["input"]) for pair in context_pairs])
        context_output_sizes = torch.stack([self._size_tensor(pair["output"]) for pair in context_pairs])

        batch: TensorBatch = {
            "context_inputs": context_inputs.unsqueeze(0),
            "context_outputs": context_outputs.unsqueeze(0),
            "context_input_masks": context_input_masks.unsqueeze(0),
            "context_output_masks": context_output_masks.unsqueeze(0),
            "context_input_original_sizes": context_input_sizes.unsqueeze(0),
            "context_output_original_sizes": context_output_sizes.unsqueeze(0),
            "query": self._grid_tensor(prepared_sample["query"]).unsqueeze(0),
            "target": self._grid_tensor(prepared_sample["target"]).unsqueeze(0),
            "query_mask": self._mask_tensor(prepared_sample["query"]).unsqueeze(0),
            "target_mask": self._mask_tensor(prepared_sample["target"]).unsqueeze(0),
            "query_original_size": self._size_tensor(prepared_sample["query"]).unsqueeze(0),
            "target_original_size": self._size_tensor(prepared_sample["target"]).unsqueeze(0),
            "metadata": {
                "heldout_index": prepared_sample.get("heldout_index"),
                "transform": prepared_sample.get("transform"),
            },
        }
        return batch

    def _placeholder_output(self, batch: TensorBatch) -> dict[str, Any]:
        tensor_shapes = {
            key: tuple(value.shape)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        return {
            "encoder_available": False,
            "shape_summary": tensor_shapes,
            "metadata": batch["metadata"],
        }

    def _model_inputs(self, batch: TensorBatch) -> dict[str, torch.Tensor]:
        return {
            "context_inputs": batch["context_inputs"],
            "context_outputs": batch["context_outputs"],
            "context_input_masks": batch["context_input_masks"],
            "context_output_masks": batch["context_output_masks"],
            "query": batch["query"],
            "query_mask": batch["query_mask"],
            "target": batch["target"],
            "target_mask": batch["target_mask"],
        }

    def forward(self, batch: TensorBatch) -> Any:
        """Run the optional encoder model or return a lightweight placeholder.

        If no model is attached, this method returns tensor shape metadata. If a
        model is attached, it calls ``self.model(**model_inputs)`` with explicit
        grid/mask tensors. Exceptions are wrapped with context so future adapter
        failures are easier to debug.
        """
        if self.model is None:
            return self._placeholder_output(batch)

        model_inputs = self._model_inputs(batch)
        try:
            return self.model(**model_inputs)
        except TypeError as exc:
            raise RuntimeError(
                "Attached encoder model did not accept the lightweight ARC tensor inputs."
            ) from exc

    def __call__(self, batch: TensorBatch) -> Any:
        """Alias for ``forward``."""
        return self.forward(batch)

    def try_load_vitarc_encoder(self) -> type[Any] | None:
        """Optionally import the ViTARC model class without making it required."""
        try:
            from vitarc.models.model import ViTARCForConditionalGeneration
        except Exception:
            return None

        self.encoder_cls = ViTARCForConditionalGeneration
        return self.encoder_cls
