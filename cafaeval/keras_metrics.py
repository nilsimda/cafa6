import numpy as np
import tensorflow as tf

from cafaeval.evaluation import evaluate_prediction
from cafaeval.graph import Prediction, propagate


class CAFAMetric(tf.keras.metrics.Metric):
    """
    Stateful metric that buffers per-protein predictions for one namespace and
    computes the CAFA weighted F score (or another CAFA metric column) once the
    full validation fold has been seen.

    Notes
    -----
    * The order of `protein_ids_in_feed` must exactly match the order in which
      the validation dataset feeds proteins to `model.fit` (disable shuffling).
    * The metric collects logits/probabilities batch by batch on the host and
      runs the official CAFA evaluation code on CPU via `tf.py_function` when
      the epoch finishes. As a consequence the reported value is only updated
      at epoch end; during the epoch it returns NaN to avoid pretending that
      partial buffers represent the final score.
    * Only single-namespace use is supported per metric instance. Train one
      model per namespace (as requested) and instantiate the metric separately
      for each model.
    """

    def __init__(
        self,
        namespace,
        ontology,
        gt,
        protein_ids_in_feed,
        tau_arr=None,
        score_column="f_w",
        propagate_mode="max",
        name=None,
        **kwargs,
    ):
        name = name or f"{namespace}_{score_column}"
        super().__init__(name=name, **kwargs)
        self.namespace = namespace
        self.ontology = ontology
        self.gt = gt
        self.score_column = score_column
        self.propagate_mode = propagate_mode
        if tau_arr is None:
            tau_arr = np.arange(0.01, 1.0, 0.01, dtype=np.float32)
        self.tau_arr = np.asarray(tau_arr, dtype=np.float32)

        # Map each protein id in the validation feed order to its GT row index
        # (-1 marks proteins that are not part of the CAFA ground truth).
        self._row_lookup = np.array(
            [gt.ids.get(pid, -1) for pid in protein_ids_in_feed], dtype=np.int32
        )
        self._num_feed_examples = len(self._row_lookup)
        self._num_feed_examples_tensor = tf.constant(self._num_feed_examples, dtype=tf.int32)
        self._num_terms = gt.matrix.shape[1]

        # Stateful tensors.
        self._pred_store = self.add_weight(
            name="pred_store",
            shape=(self._num_feed_examples, self._num_terms),
            initializer="zeros",
            dtype=tf.float32,
            trainable=False,
            aggregation=tf.VariableAggregation.NONE,
        )
        self._cursor = self.add_weight(
            name="cursor",
            shape=(),
            initializer="zeros",
            dtype=tf.int32,
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )
        self._ready = self.add_weight(
            name="ready",
            shape=(),
            initializer="zeros",
            dtype=tf.int32,
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Only predictions are needed; y_true/sample_weight are ignored on purpose.
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        batch_size = tf.shape(y_pred)[0]
        start = self._cursor
        end = start + batch_size

        # Guard against mismatched epoch lengths early.
        with tf.control_dependencies(
            [
                tf.debugging.assert_less_equal(
                    end,
                    self._num_feed_examples_tensor,
                    message="CAFA metric received more batches than protein IDs supplied.",
                )
            ]
        ):
            batch_indices = tf.range(start, end)

        current_store = self._pred_store.read_value()
        updated = tf.tensor_scatter_nd_update(
            current_store,
            tf.expand_dims(batch_indices, axis=1),
            y_pred,
        )
        self._pred_store.assign(updated)
        self._cursor.assign(end)
        # Mark metric "ready" once the full validation fold has been seen.
        self._ready.assign(
            tf.where(
                tf.equal(end, self._num_feed_examples),
                tf.constant(1, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
            )
        )

    def result(self):
        def _compute():
            # Build prediction matrix aligned with GT rows and propagate scores upward.
            pred_seq = self._pred_store.numpy()
            pred_mat = np.zeros_like(self.gt.matrix, dtype=np.float32)
            for row_idx, scores in zip(self._row_lookup, pred_seq):
                if row_idx >= 0:
                    pred_mat[row_idx] = scores
            propagate(pred_mat, self.ontology, self.ontology.order, mode=self.propagate_mode)
            predictions = {self.namespace: Prediction(self.gt.ids, pred_mat, self.namespace)}
            eval_df = evaluate_prediction(
                predictions,
                {self.namespace: self.gt},
                {self.namespace: self.ontology},
                self.tau_arr,
                normalization="cafa",
                n_cpu=1,
                progress=False,
            )
            if self.score_column not in eval_df.columns:
                raise ValueError(f"Metric column '{self.score_column}' not found in evaluation frame.")
            best = eval_df[eval_df["ns"] == self.namespace][self.score_column].max()
            if np.isnan(best):
                best = 0.0
            return np.array(best, dtype=np.float32)

        return tf.cond(
            tf.equal(self._ready, 1),
            lambda: tf.py_function(_compute, [], Tout=tf.float32),
            lambda: tf.constant(np.nan, dtype=tf.float32),
        )

    def reset_states(self):
        self._pred_store.assign(tf.zeros_like(self._pred_store))
        self._cursor.assign(0)
        self._ready.assign(0)
