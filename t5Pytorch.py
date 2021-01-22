from t5.models import HfPyTorchModel
from tqdm import tqdm
import time
import torch
from absl import logging
from utils import *
import tensorflow.compat.v1 as tf
import functools
import os
import torch.nn as nn
from T5mini import T5mini


class t5Pytorch(HfPyTorchModel):
    def __init__(self, model_spec, model_dir, device, student=None):
        """Constructor for HfModel class.

        Args:
          model_spec: A str to pass into the `pretrained_model_name_or_path`
            argument of `transformers.T5ForConditionalGeneration.from_pretrained`
            (e.g. `"t5-base"` or a path to a previously trained model) or an
            instance of the `transformers.configuration_t5.T5Config` class to use
            to directly construct the `transformers.T5ForConditionalGeneration`
            object.
          model_dir: str, directory to save and load model checkpoints.
          device: `torch.device` on which the model should be run.
        """
        super(HfPyTorchModel, self).__init__()
        # We have to import transformers here because it has a side effect of
        # creating a TensorFlow graph, which prevents eager execution from being
        # enabled in files that import hf_model.py
        import transformers  # pylint: disable=import-outside-toplevel,g-import-not-at-top
        if isinstance(model_spec, str):
            self._model = transformers.T5ForConditionalGeneration.from_pretrained(
                model_spec
            )
        elif isinstance(model_spec, transformers.T5Config):
            self._model = transformers.T5ForConditionalGeneration(model_spec)
        else:
            raise ValueError("model_spec should be a string or T5Config.")

        if student is not None:
            self.student = T5mini(student, teacher_size=self._model.shared.embedding_dim)
            self.student.to(device)
        else:
            self.student = None


        tf.io.gfile.makedirs(model_dir)
        self._writer = torch.utils.tensorboard.writer.SummaryWriter(model_dir)
        self._model_dir = model_dir
        self._device = device
        self._model.to(device)

        self._step = 0
        self.load_latest_checkpoint()
        if self.student is not None:
            self._step = 0
            self.student.to(device)
            self._model.eval()
        self.to_tensor = functools.partial(torch.as_tensor, device=self._device)

    def train(
            self,
            data_loader,
            eval_loaders,
            steps,
            save_steps,
            sequence_length,
            split,
            batch_size,
            optimizer,
            tokenizer,
            learning_rate_scheduler=None,
    ):
        """Train the model on the given Mixture or Task.

    Args:
      mixture_or_task_name: str, the name of the Mixture or Task to train on.
        Must be pre-registered in the global `t5.data.TaskRegistry` or
        `t5.data.MixtureRegistry.`
      steps: int, the total number of steps to train for.
      save_steps: int, the number of steps between checkpoint saves.
      sequence_length: dict of int, a dict mapping feature name to length.
      split: str or `tensorflow_datasets.Split`, the data split to load.
      batch_size: int, the number of padded sequences in each batch.
      optimizer: function that takes the model parameters as its sole argument.
        For example, to use an AdamW optimizer with a learning rate of 1e-4,
        you could pass in `functools.partial(transformers.AdamW, lr=1e-4)`.
      learning_rate_scheduler: optional function that takes in an optimizer as
        its sole argument. For example, to use a schedule that warms up the
        optimizer's learning rate after 100 steps, you could pass in
        `functools.partial(transformers.get_constant_schedule_with_warmup,
       num_warmup_steps=100)`.
    """
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        self._model.train()
        ds = data_loader
        if self.student is not None:
            self._model.eval()
            self.student.train()
            optimizer = optimizer(self.student.parameters())
        else:
            optimizer = optimizer(self._model.parameters())
        if learning_rate_scheduler:
            learning_rate_scheduler = learning_rate_scheduler(optimizer)
        now = time.time()
        tran_loss = 0
        train_steps = 0
        while self._step < steps:
            pbar = tqdm(enumerate(ds), "training...", len(ds))
            for train_step, batch in pbar:
                if not self._step % save_steps:
                    if train_steps != 0:
                        self._writer.add_scalar(
                            f"train_mean_loss", tran_loss/train_steps, global_step=self._step
                        )
                    tran_loss = 0
                    train_steps = 0

                    self._model.eval()

                    if self.student is not None:
                        self.student.eval()

                    for task in eval_loaders:
                        loader = eval_loaders[task]
                        evalbar = tqdm(enumerate(loader), "evaling...", len(loader))
                        scores = 0
                        loss = 0
                        task_loss = 0
                        emb_loss = 0

                        for _, eval_batch in evalbar:
                            y = eval_batch['target_ids'].to("cuda:0", dtype=torch.long)
                            attention_mask = eval_batch['source_mask'].to("cuda:0", dtype=torch.long)
                            ids = eval_batch['source_ids'].to("cuda:0", dtype=torch.long)
                            lm_labels = y.clone()

                            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                            if task == "bea":
                                if self.student is not None:
                                    out = self.student(input_ids=ids, labels=lm_labels, attention_mask=attention_mask)
                                else:
                                    out = self._model(input_ids=ids, labels=lm_labels, attention_mask=attention_mask)
                                scores += out[0].cpu().tolist()
                            else:
                                if self.student is not None:
                                    with torch.no_grad():
                                        generated_ids = self.student.T5mini.generate(
                                            input_ids=ids,
                                            attention_mask=attention_mask,
                                            max_length=256,
                                        )

                                else:
                                    generated_ids = self._model.generate(
                                        input_ids=ids,
                                        attention_mask=attention_mask,
                                        max_length=256,
                                    )

                                if task == "corr":
                                        preds, target, score = get_bleu(generated_ids, y)
                                elif task == "jfleg":
                                    preds, target, score = get_gleu(generated_ids, y)
                                scores += score

                        self._writer.add_scalar(
                            f"eval_{task}", scores/len(loader), global_step=self._step
                        )
                        self._writer.add_text(f"val_text_example_{task}", ' '.join(preds) + " target --->" + ' '.join(target),
                                                global_step=self._step)
                        self._writer.add_scalar(
                            f"eval_loss_{task}", loss / len(loader), global_step=self._step
                        )
                        self._writer.add_scalar(
                            f"eval_task_loss_{task}", task_loss / len(loader), global_step=self._step
                        )
                        self._writer.add_scalar(
                            f"eval_emb_loss_{task}", emb_loss / len(loader), global_step=self._step
                        )

                    if self.student is None:
                        logging.info("Saving checkpoint for step %s", self._step)
                        self.save_checkpoint(self._step)
                    else:
                        logging.info("Saving checkpoint for step %s", self._step)
                        self.save_checkpoint_student(self._step)

                    if self.student is None:
                        self._model.train()
                    else:
                        self.student.train()

                train_steps += 1
                if self.student is not None:
                    self.student.zero_grad()
                else:
                    self._model.zero_grad()

                y = batch['target_ids'].to("cuda:0", dtype=torch.long)
                attention_mask = batch['source_mask'].to("cuda:0", dtype=torch.long)
                ids = batch['source_ids'].to("cuda:0", dtype=torch.long)
                lm_labels = y.clone()
                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

                if self.student is not None:
                    with torch.no_grad():
                        outputs = self._model(
                            input_ids=ids, labels=lm_labels, attention_mask=attention_mask
                        )
                    student_outputs = self.student(
                        input_ids=ids, labels=lm_labels, attention_mask=attention_mask, teacher=outputs
                    )

                    loss = student_outputs[1]
                    task_loss = student_outputs[0]
                    emb_loss = student_outputs[2]
                    distill_loss = student_outputs[3]
                    self._writer.add_scalar(
                        "task_loss", task_loss.detach().cpu().numpy(), self._step
                    )
                    self._writer.add_scalar(
                        "emb_loss", emb_loss.detach().cpu().numpy(), self._step
                    )
                    self._writer.add_scalar(
                        "distill_loss", distill_loss.detach().cpu().numpy(), self._step
                    )
                else:
                    outputs = self._model(
                        input_ids=ids, labels=lm_labels, attention_mask=attention_mask
                    )
                    loss = outputs[0]
                tran_loss += loss
                pbar.set_description(f"loss: {loss.tolist()}")
                loss.backward()
                optimizer.step()
                if learning_rate_scheduler:
                    learning_rate_scheduler.step()
                self._writer.add_scalar(
                    "learning_rate_scheduler", get_lr(optimizer), self._step
                )

                self._writer.add_scalar(
                    "loss", loss.detach().cpu().numpy(), self._step
                )

                self._writer.add_scalar("step/s", 1 / (time.time() - now), self._step)
                now = time.time()
                self._step += 1

            if self.student is None:
                logging.info("Saving checkpoint for step %s", self._step)
                self.save_checkpoint(self._step)
            else:
                logging.info("Saving checkpoint for step %s", self._step)
                self.save_checkpoint_student(self._step)

    def eval(self, tokenizer, device, loader, max_length=64, decode=True):
        """

    :param tokenizer:
    :param device:
    :param loader:
    :return:
    """
        self._model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for _, batch in tqdm(enumerate(loader, 0), "eval...", len(loader)):
                y = batch['target_ids'].to(device, dtype=torch.long)
                ids = batch['source_ids'].to(device, dtype=torch.long)
                mask = batch['source_mask'].to(device, dtype=torch.long)

                generated_ids = self._model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=max_length,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                if decode:
                    preds = [tokenizer.decode(g[:max_length], skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                             generated_ids]
                    target = [tokenizer.decode(t[:max_length], skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in
                              y]
                    predictions.extend(preds)
                    actuals.extend(target)
                else:
                    predictions.extend(generated_ids)
                    actuals.extend(y)

        return predictions, actuals

    def save_checkpoint_student(self, step):
        """Save the current model parameters to the `model_dir`.

        Args:
          step: int, the current training step.
        """
        path = os.path.join(self._model_dir, f"student_{step}")
        path_w = os.path.join(self._model_dir, f"student_w_{step}")
        torch.save(self.student.W.state_dict(), path_w)
        torch.save(self.student.T5mini.state_dict(), path)

    def load_latest_checkpoint_student(self):
        """Load the most recent checkpoint and update the model's current step."""
        latest_step = self.get_latest_checkpoint_step()
        if latest_step is not None:
            self.load_checkpoint(latest_step)
