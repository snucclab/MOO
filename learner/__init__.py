from common.model.types import Description, Text, Label
from learner.common import *


class SupervisedTrainer(TrainerBase):
    def _evaluate_a_batch(self, batch: Example, **kwargs) -> dict:
        # Compute output first.
        # ---- Output ----
        # expression: Expression [B, T]
        # description?: B-List of Description ([N, D] & [V, D])
        output = self._model(text=batch.text, **kwargs)

        # Filter only the interested output
        generated_desc = output.get('description', None)
        output = {
            'batch': batch,
            'joint': {
                'expression': output['expression'],
                'description': generated_desc
            }
        }

        return output

    def _evaluate(self, name: str, configuration: dict) -> dict:
        self._reset_test_random_generator()
        self._dataset.select_items_with_file(configuration[KEY_SPLIT_FILE])
        self._model.eval()

        batch_output_pairs = []
        with torch.no_grad():
            batches = self._dataset.get_minibatches(self._batch_size)

            for batch in batches:
                # ---- Input ----
                # text: Text [B, S]
                # beam: int
                # beam_desc: int
                batch_output_pairs.append(self._evaluate_a_batch(batch, beam=configuration[KEY_BEAM],
                                                                 beam_desc=configuration[KEY_BEAM_DESC]))

            results = self._tester.check(batch_output_pairs)
            self._record_evaluation_output(name, results)

        # Remove 'dump' key before returning
        results.pop('dump')
        return results

    def _before_update(self) -> dict:
        self._dataset.select_items_with_file(self._train_config[KEY_SPLIT_FILE])
        self._model.train()
        return {}

    def _update_module(self, pretrain) -> dict:
        reports = []
        batch_gen = list(self._dataset.get_minibatches(self._batch_size))
        for batch in batch_gen:
            # ---- Input ----
            # text: Text [B, S]
            # expression: Expression [B, T]
            # description: B-List of Description [N/V, D]
            # ---- Output ----
            # expression: ExpressionPrediction [B, T]
            # num_desc?: B-List of Prediction [N, D]
            # var_desc?: B-List of Prediction [V, D] or Prediction [B, VD]
            # var_target?: Label [B, VD]
            out_dict = self._model(**batch.as_dict(), is_training=True)

            # Compute accuracy of tokens
            with torch.no_grad():
                report = batch.accuracy_of(**out_dict)

            # Compute loss
            losses = batch.smoothed_cross_entropy(**out_dict)

            # If pretraining, select the loss values specified for pre-training
            if pretrain:
                losses = {key: value
                          for key, value in losses.items()
                          if key in self._pretrain}

                if not losses:
                    # If there's no pretraining signal, flag it and pass the pretraining phase.
                    self._no_pretrain_signal = True
                    return {TIMESTEPS_THIS_ITER: 1}

            # Build sum of losses
            total_loss = sum(losses.values())
            losses['total'] = total_loss
            report.update({'loss_' + key: value for key, value in losses.items()})

            # Add to report (cast to float to avoid memory leak)
            reports.append({key: float(value) for key, value in report.items()})

            # Run Backward prop.
            total_loss.backward()
            self._model.after_backprop()

        report = merge_reports(reports)
        report[TIMESTEPS_THIS_ITER] = len(reports)
        return report
