"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
import os
import datetime
import time
import logging

from omegaconf import OmegaConf, open_dict

import torch

import breaching


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval)


def main_process(process_idx, local_group_size, cfg, cfg_flprocess=None, num_trials=100):
    """
    This function will use FLPrivacy trained model and attack them through several configurations.
    Things that can be done:
        1. Attack FLModel at different time of the training process (made through cfg file)
        2. Attack FLModel with different training strategy (e.g. batch size, local epoch, etc.)
    """
    total_time = time.time()  # Rough time measurements here
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)
    if cfg_flprocess is not None:
        assert cfg_flprocess.server.model_name in cfg.case.model  # `in` because "custom" keyword in Breaching models
    model, loss_fn = breaching.cases.construct_model(cfg.case.model, cfg.case.data, cfg.case.server.pretrained)
    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    if cfg.case.user.user_idx is not None:
        print("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")
    log.info(
        f"Partitioning is set to {cfg.case.data.partition}. Make sure there exist {num_trials} users in this scheme."
    )

    cfg.case.user.user_idx = -1
    run = 0
    overall_metrics = []
    once_print_overview = True
    while run < num_trials:
        local_time = time.time()
        # Select data that has not been seen before:
        cfg.case.user.user_idx += 1

        # Load Model from specific location
        if cfg.get("load_model_path", None) is not None:
            log.info(f"Loading model from {cfg.load_model_path}")
            model.model.load_state_dict(torch.load(cfg.load_model_path))
        try:
            user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
        except ValueError:
            log.info("Cannot find other valid users. Finishing benchmark.")
            break
        if cfg.case.data.modality == "text":
            dshape = user.dataloader.dataset[0]["input_ids"].shape
            data_shape_mismatch = any([d != d_ref for d, d_ref in zip(dshape, cfg.case.data.shape)])
        else:
            data_shape_mismatch = False  # Handled by preprocessing for images
        if len(user.dataloader.dataset) < user.num_data_points or data_shape_mismatch:
            log.info(f"Skipping user {user.user_idx} (has not enough data or data shape mismatch).")
        else:
            if once_print_overview:
                # Summarize startup:
                overview = breaching.utils.return_overview(server, user, attacker)
                print(overview)
                log.info(overview)
                once_print_overview = False

            log.info(f"Now evaluating user {user.user_idx} in trial {run}.")
            run += 1
            # Run exchange
            shared_user_data, payloads, true_user_data = server.run_protocol(user)
            # Evaluate attack:
            with open_dict(cfg):
                cfg.case.data.vocab_size = 100  # otherwise there is an error
            try:
                reconstruction, stats = attacker.reconstruct(
                    payloads, shared_user_data, server.secrets, dryrun=cfg.dryrun
                )

                # Run the full set of metrics:
                metrics = breaching.analysis.report(
                    reconstruction,
                    true_user_data,
                    payloads,
                    server.model,
                    order_batch=True,
                    compute_full_iip=True,
                    compute_rpsnr=True,
                    compute_ssim=True,
                    cfg_case=cfg.case,
                    setup=setup,
                )
                # Add query metrics
                metrics["queries"] = user.counted_queries

                # Save local summary:
                breaching.utils.save_summary(cfg, metrics, stats, time.time() - local_time, original_cwd=False)
                overall_metrics.append(metrics)
                # Save recovered data:
                if cfg.save_reconstruction:
                    breaching.utils.save_reconstruction(reconstruction, payloads, true_user_data, cfg)
                if cfg.dryrun:
                    break
            except Exception as e:  # noqa # yeah we're that close to the deadlines
                log.info(f"Trial {run} broke down with error {e}.")

    # Compute average statistics:
    average_metrics = breaching.utils.avg_n_dicts(overall_metrics)

    # Save global summary:
    breaching.utils.save_summary(
        cfg, average_metrics, stats, time.time() - total_time, original_cwd=True, table_name="BENCHMARK_breach"
    )


@hydra.main(config_path="breaching/config", config_name="Adam_cfg", version_base="1.1")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")
    print(cfg.case.user.num_data_points)
    print(cfg.case.user.num_data_per_local_update_step)

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = 233  # The benchmark seed is fixed by default!

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations {cfg.name} with total train time: "
        f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


if __name__ == "__main__":
    main_launcher()
