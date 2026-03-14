import torch
from transformers import Trainer

class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop('sample_idxs')
            evaluation_kwargs = inputs.pop('evaluation_kwargs')
            evaluator = evaluation_kwargs.pop('evaluator')
            
            
            # # DEBUG: print batch info before the failing call
            # print(f"[DEBUG] input_ids shape: {inputs.get('input_ids', 'MISSING')}")
            # if 'input_ids' in inputs and inputs['input_ids'] is not None:
            #     print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
            # if 'frames' in inputs and inputs['frames'] is not None:
            #     print(f"[DEBUG] frames shape: {inputs['frames'].shape}")

            try:
                output_ids = getattr(model, evaluator)(
                    **inputs, **evaluation_kwargs,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            except ValueError as e:
                # Catch the error and dump the problematic batch
                print(f"\n{'='*60}")
                print(f"[ERROR] Caught ValueError in prediction_step!")
                print(f"[ERROR] Message: {e}")
                print(f"[ERROR] Input keys: {list(inputs.keys())}")
                for k, v in inputs.items():
                    if hasattr(v, 'shape'):
                        print(f"[ERROR]   {k}: shape={v.shape}, dtype={v.dtype}")
                    elif isinstance(v, list):
                        print(f"[ERROR]   {k}: list len={len(v)}")
                    else:
                        print(f"[ERROR]   {k}: {type(v)} = {v}")
                print(f"{'='*60}")
                breakpoint()  # Enter the debugger to inspect the issue
                raise  # re-raise so you still see the full traceback
            
            # output_ids = getattr(model, evaluator)(**inputs, **evaluation_kwargs, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            return (None, output_ids.reshape(1, -1), sample_idxs)