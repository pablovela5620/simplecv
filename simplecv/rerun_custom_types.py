from typing import Any, Self

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float, Int, UInt8
from numpy import ndarray


def confidence_scores_to_rgb(
    confidence_scores: Float[ndarray, "n_frames n_kpts 1"],
) -> UInt8[ndarray, "n_frames n_kpts 3"]:
    """Converts confidence scores to RGB colors using a Red-Yellow-Green gradient.

    The color mapping is as follows:
    - A confidence score of 0.0 is mapped to Red (255, 0, 0).
    - A confidence score of 0.5 is mapped to Yellow (255, 255, 0).
    - A confidence score of 1.0 is mapped to Green (0, 255, 0).
    Scores are linearly interpolated between these points. Values outside the
    [0.0, 1.0] range will be clipped by the function.

        confidence_scores (Float32[ndarray, "n_frames n_kpts 1"]):
            A NumPy array of shape (n_frames, n_kpts, 1) containing
            confidence values. Values are typically between 0.0 and 1.0.

        UInt8[ndarray, "n_frames n_kpts 3"]:
            A NumPy array of shape (n_frames, n_kpts, 3) containing
            the corresponding RGB colors as uint8 values. Each color is
            represented as an array of three integers [R, G, B]."""
    n_frames, n_kpts, _ = confidence_scores.shape
    clipped_confidences: Float[ndarray, "n_frames n_kpts 1"] = np.clip(confidence_scores, a_min=0.0, a_max=1.0)
    clipped_confidences: Float[ndarray, "n_frames n_kpts"] = np.squeeze(clipped_confidences, axis=-1)
    safe_confidences: Float[ndarray, "n_frames n_kpts"] = np.nan_to_num(clipped_confidences, nan=0.0)

    colors: UInt8[ndarray, "n_frames n_kpts 3"] = np.zeros((n_frames, n_kpts, 3), dtype=np.uint8)
    # Segment A: red → yellow for conf ≤ 0.5
    mask_low = safe_confidences <= 0.5
    if mask_low.any():
        t_low = safe_confidences[mask_low] * 2.0  # 0‥1
        colors[..., 0][mask_low] = 255  # red fixed
        colors[..., 1][mask_low] = (t_low * 255).astype(np.uint8)

    # Segment B: yellow → green for conf > 0.5
    mask_high = ~mask_low
    if mask_high.any():
        t_high = (safe_confidences[mask_high] - 0.5) * 2.0
        colors[..., 0][mask_high] = ((1.0 - t_high) * 255).astype(np.uint8)
        colors[..., 1][mask_high] = 255  # green fixed

    # blue channel remains 0
    return colors


def _confidence_component_descriptor(
    archetype_name: str,
    field_name: str = "confidences",
    *,
    component_type: str = "simplecv.components.KeypointConfidence",
) -> rr.ComponentDescriptor:
    component: str = f"{archetype_name}:{field_name}"
    return rr.ComponentDescriptor(
        component=component,
        archetype=archetype_name,
        component_type=component_type,
    )


class ConfidenceBatch(rr.ComponentBatchMixin):
    """A batch of confidence data."""

    def __init__(self, confidence: Float[ndarray, "..."]) -> None:
        self.confidence = confidence

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """The descriptor of the custom component."""
        return rr.ComponentDescriptor(
            "simplecv.components.KeypointConfidence",
            component_type="simplecv.components.KeypointConfidence",
        )

    def as_arrow_array(self) -> pa.Array:
        """The arrow batch representing the custom component."""
        return pa.array(self.confidence, type=pa.float32())


class AverageConfidenceBatch(rr.ComponentBatchMixin):
    """A batch containing one or more average confidence values."""

    def __init__(
        self,
        average_confidence: float | Float[ndarray, "n"],
    ) -> None:
        average_conf_array: Float[ndarray, "n"] = np.atleast_1d(np.asarray(average_confidence, dtype=np.float32))
        self.average_confidence: Float[ndarray, "n"] = average_conf_array

    def component_descriptor(self) -> rr.ComponentDescriptor:
        """Descriptor for the average confidence component."""
        return rr.ComponentDescriptor(
            "simplecv.components.KeypointConfidenceMean",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

    def as_arrow_array(self) -> pa.Array:
        """Arrow representation of the average confidence value."""
        return pa.array(self.average_confidence, type=pa.float32())


def _flatten_confidences(confidences: Float[ndarray, "..."]) -> Float[ndarray, "n"]:
    conf_array: Float[ndarray, "..."] = np.asarray(confidences, dtype=np.float32)
    if conf_array.ndim > 1:
        conf_array = conf_array.reshape(-1)
    return conf_array.astype(np.float32, copy=False)


class _ConfidenceAwareColumnList(rr.ComponentColumnList):
    """Extend a column list with confidence and per-frame averages for send_columns.

    The helper mirrors the built-in archetype column helpers while quietly
    appending the extra confidence-related components before partitioning.
    """

    def __init__(
        self,
        base_columns: rr.ComponentColumnList,
        confidences: Float[ndarray, "n"] | None,
        confidence_descriptor: rr.ComponentDescriptor,
        average_descriptor: rr.ComponentDescriptor,
        *,
        average_confidences: Float[ndarray, "m"] | None = None,
    ) -> None:
        base_list: list[rr.ComponentColumn] = list(base_columns)
        self._average_descriptor: rr.ComponentDescriptor = average_descriptor
        self._provided_average: Float[ndarray, "m"] | None = (
            None if average_confidences is None else np.asarray(average_confidences, dtype=np.float32).reshape(-1)
        )
        if confidences is None:
            self._raw_confidences: Float[ndarray, "0"] | None = None
            super().__init__(base_list)
            return

        raw_conf: Float[ndarray, "n"] = _flatten_confidences(confidences)
        self._raw_confidences = raw_conf
        base_list.append(rr.ComponentColumn(confidence_descriptor, ConfidenceBatch(raw_conf)))
        super().__init__(base_list)

    def partition(self, lengths: Int[ndarray, "m"]) -> rr.ComponentColumnList:
        partitioned: rr.ComponentColumnList = super().partition(lengths)
        if self._raw_confidences is None:
            return partitioned

        lengths_arr: Int[ndarray, "m"] = np.asarray(lengths, dtype=np.int64)
        total_length: int = int(lengths_arr.sum())
        if total_length != int(self._raw_confidences.size):
            raise ValueError("Sum of partition lengths does not match number of confidences.")

        if self._provided_average is not None:
            if self._provided_average.shape[0] != lengths_arr.shape[0]:
                raise ValueError("Provided average confidences must match number of partitions.")
            averages = self._provided_average.astype(np.float32, copy=False)
        else:
            offsets: Int[ndarray, "m_plus_one"] = np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(lengths_arr, dtype=np.int64))
            )
            averages = np.empty(lengths_arr.shape[0], dtype=np.float32)
            for idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=False)):
                if end <= start:
                    averages[idx] = np.nan
                    continue
                segment: Float[ndarray, "k"] = self._raw_confidences[start:end]
                valid = segment[~np.isnan(segment)]
                averages[idx] = float(np.nanmean(valid)) if valid.size > 0 else np.nan

        avg_column = rr.ComponentColumn(
            self._average_descriptor,
            AverageConfidenceBatch(averages),
            lengths=np.ones_like(averages, dtype=np.int32),
        )

        columns_with_average: list[rr.ComponentColumn] = list(partitioned)
        columns_with_average.append(avg_column)
        return rr.ComponentColumnList(columns_with_average)


class Points2DWithConfidence(rr.AsComponents):
    """Custom Points2D archetype with per-keypoint and average confidences."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 2"],
        confidences: Float[ndarray, "n_kpts"],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
        colors: UInt8[ndarray, "n_kpts 3"] | None = None,
        radii: float | None = None,
    ) -> None:
        self.points2d = rr.Points2D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
            colors=colors,
            radii=radii,
        )
        self._include_confidence: bool = True
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        confidences_arr: Float[ndarray, "n_kpts"] = np.asarray(confidences, dtype=np.float32)
        mean_confidence: float = float(np.nanmean(confidences_arr)) if confidences_arr.size else float("nan")
        self.confidences = ConfidenceBatch(confidences_arr).described(confidence_descriptor)
        self.average_confidence = AverageConfidenceBatch(mean_confidence).described(average_descriptor)

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        batches: list[rr.DescribedComponentBatch] = list(self.points2d.as_component_batches())
        if self._include_confidence:
            batches.extend([self.confidences, self.average_confidence])
        return batches

    @classmethod
    def columns(
        cls,
        *,
        positions: Float[ndarray, "n 2"] | None = None,
        confidences: Float[ndarray, "n"] | None = None,
        average_confidences: Float[ndarray, "m"] | None = None,
        class_ids: Int[ndarray, "n"] | int | None = None,
        keypoint_ids: Int[ndarray, "n"] | list[int] | None = None,
        show_labels: bool | ndarray | None = None,
        colors: UInt8[ndarray, "n 3"] | None = None,
        radii: float | Float[ndarray, "n"] | None = None,
    ) -> rr.ComponentColumnList:
        """Return column components mirroring `rr.Points2D.columns` plus confidences.

        Inputs are already flattened to `(n_frames * n_kpts)` by the caller so the
        returned list can be partitioned with per-frame lengths before handing it
        to `rr.send_columns`.
        """
        base_columns: rr.ComponentColumnList = rr.Points2D.columns(
            positions=positions,
            radii=radii,
            colors=colors,
            show_labels=show_labels,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
        )

        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

        if confidences is None and average_confidences is None:
            return base_columns

        return _ConfidenceAwareColumnList(
            base_columns,
            confidences,
            confidence_descriptor,
            average_descriptor,
            average_confidences=average_confidences,
        )

    @classmethod
    def from_fields(cls, **fields: Any) -> Self:
        """Create a static placeholder matching `rr.Points2D.from_fields`.

        The returned instance excludes confidence data so it can be safely logged
        as a static archetype while columnar updates provide the dynamic values.
        """
        instance = cls.__new__(cls)
        instance.points2d = rr.Points2D.from_fields(**fields)
        instance._include_confidence = False
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence2D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        empty_conf: Float[ndarray, "0"] = np.empty(0, dtype=np.float32)
        instance.confidences = ConfidenceBatch(empty_conf).described(confidence_descriptor)
        instance.average_confidence = AverageConfidenceBatch(empty_conf).described(average_descriptor)
        return instance


class Points3DWithConfidence(rr.ComponentColumn):
    """Custom Points3D archetype with per-keypoint and average confidences."""

    def __init__(
        self: Any,
        positions: Float[ndarray, "n_kpts 3"],
        confidences: Float[ndarray, "n_kpts"],
        class_ids: int,
        keypoint_ids: list[int],
        show_labels: bool = False,
        colors: UInt8[ndarray, "n_kpts 3"] | None = None,
        radii: float | None = None,
    ) -> None:
        self.points3d = rr.Points3D(
            positions=positions,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
            show_labels=show_labels,
            colors=colors,
            radii=radii,
        )
        self._include_confidence: bool = True
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        confidences_arr: Float[ndarray, "n_kpts"] = np.asarray(confidences, dtype=np.float32)
        mean_confidence: float = float(np.nanmean(confidences_arr)) if confidences_arr.size else float("nan")
        self.confidences = ConfidenceBatch(confidences_arr).described(confidence_descriptor)
        self.average_confidence = AverageConfidenceBatch(mean_confidence).described(average_descriptor)

    def as_component_batches(self) -> list[rr.DescribedComponentBatch]:
        batches: list[rr.DescribedComponentBatch] = list(self.points3d.as_component_batches())
        if self._include_confidence:
            batches.extend([self.confidences, self.average_confidence])
        return batches

    @classmethod
    def columns(
        cls,
        *,
        positions: Float[ndarray, "n 3"] | None = None,
        confidences: Float[ndarray, "n"] | None = None,
        average_confidences: Float[ndarray, "m"] | None = None,
        class_ids: Int[ndarray, "n"] | int | None = None,
        keypoint_ids: Int[ndarray, "n"] | list[int] | None = None,
        show_labels: bool | ndarray | None = None,
        colors: UInt8[ndarray, "n 3"] | None = None,
        radii: float | Float[ndarray, "n"] | None = None,
    ) -> rr.ComponentColumnList:
        """Return column components mirroring `rr.Points3D.columns` plus confidences."""
        base_columns: rr.ComponentColumnList = rr.Points3D.columns(
            positions=positions,
            colors=colors,
            radii=radii,
            show_labels=show_labels,
            class_ids=class_ids,
            keypoint_ids=keypoint_ids,
        )

        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )

        if confidences is None and average_confidences is None:
            return base_columns

        return _ConfidenceAwareColumnList(
            base_columns,
            confidences,
            confidence_descriptor,
            average_descriptor,
            average_confidences=average_confidences,
        )

    @classmethod
    def from_fields(cls, **fields: Any) -> Self:
        """Create a static placeholder for `Points3DWithConfidence` headers."""
        instance = cls.__new__(cls)
        instance.points3d = rr.Points3D.from_fields(**fields)
        instance._include_confidence = False
        confidence_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D"
        )
        average_descriptor: rr.ComponentDescriptor = _confidence_component_descriptor(
            "simplecv.KeypointConfidence3D",
            field_name="average_confidence",
            component_type="simplecv.components.KeypointConfidenceMean",
        )
        empty_conf: Float[ndarray, "0"] = np.empty(0, dtype=np.float32)
        instance.confidences = ConfidenceBatch(empty_conf).described(confidence_descriptor)
        instance.average_confidence = AverageConfidenceBatch(empty_conf).described(average_descriptor)
        return instance
