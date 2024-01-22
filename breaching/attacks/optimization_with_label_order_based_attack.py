"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import numpy as np

import torch
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, MaskedGradientLoss, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)


class OptimizationLabelOrderAttacker(OptimizationBasedAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        self.num_classes = server_payload[0]["metadata"].classes
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions, candidate_labels = [], []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                data, label_order = self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                candidate_solutions += [data]
                candidate_labels += [label_order.argmax(dim=-1)]

                scores[trial] = self._score_trial(
                    candidate_solutions[trial], candidate_labels[trial], labels, rec_models, shared_data
                )
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution, optimal_labels = self._select_optimal_reconstruction(
            candidate_solutions, candidate_labels, scores, stats
        )
        reconstructed_data = dict(data=optimal_solution, labels=optimal_labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(
            self.loss_fn,
            self.cfg.impl,
            shared_data[0]["metadata"]["local_hyperparams"],
        )

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        # candidate_label = self._initialize_data(labels.shape)
        candidate_label = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], self.num_classes])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        best_candidate_label = candidate_label.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        optimizer, scheduler = self._init_optimizer([candidate, candidate_label])
        current_wallclock = time.time()
        objective_values = np.zeros(self.cfg.optim.max_iterations)
        minimal_iteration_so_far = 0
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(
                    candidate, candidate_label, labels, rec_model, optimizer, shared_data, iteration
                )
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    objective_values[iteration] = objective_value.detach()
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()
                        best_candidate_label = candidate_label.detach().clone()
                        minimal_iteration_so_far = iteration

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1}/{self.cfg.optim.max_iterations}"
                        f" | Rec. loss: {objective_value.item():2.4e}"
                        f" | Task loss: {task_loss.item():2.4e}"
                        f" | Label Loss: {self._label_regularizer(candidate_label.softmax(dim=-1), labels).item():2.4e}"
                        f" | T: {timestamp - current_wallclock:4.2f}s"
                        + (" | Checking Early Stopping " if self.cfg.optim.get("stopping", False) else "")
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())
                stats[f"Trial_{trial}_Task"].append(task_loss.item())

                if self.cfg.optim.get("stopping", False) and self._early_stopping_criterion(
                    objective_values,
                    iteration,
                    minimal_value_so_far,
                    minimal_iteration_so_far,
                ):
                    log.info(f"Early stopping criterion triggered in iteration {iteration}.")
                    break

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach(), best_candidate_label.detach()

    def _label_regularizer(self, candidate_labels, labels):
        sorted_candidate_labels, _ = candidate_labels.argmax(-1).sort(dim=-1, descending=True)
        labels_sorted, _ = labels.sort(dim=-1, descending=True)
        mean_square_error = (sorted_candidate_labels - labels.float()).pow(2).mean()
        return mean_square_error * self.cfg.optim.get("label_regularizer_weight", 0.01)

    def _compute_objective(self, candidate, candidate_labels, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(
                    model, data["gradients"], candidate_augmented, candidate_labels.softmax(dim=-1)
                )
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented)

            total_objective += self._label_regularizer(candidate_labels.softmax(dim=-1), labels)

            if total_objective.requires_grad:
                total_objective.backward(inputs=[candidate, candidate_labels], create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _score_trial(self, candidate, candidate_label, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, candidate_label)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, candidate_labels, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        optimal_solution_label = candidate_labels[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution, optimal_solution_label
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution), torch.zeros_like(optimal_solution_label)
