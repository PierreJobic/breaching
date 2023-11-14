import re
import copy
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

from matplotlib import cm
from omegaconf import OmegaConf


def get_df_from_config(df, real_config, path_to_config="/breaching/config/experiment"):
    # Sachant un df: pandas.DataFrame df, et un config: DictConfig, extrait les lignes du df qui correspondent au config
    pass


def plot_3d_df(
    filtered_df,
    x_axis="mask_type",
    y_axis="sparsity",
    loss_list=["mse", "ssim", "psnr"],
    metric_list=["mean", "var"],
    squeeze_dict={"mask_type": [("no_mask", "mask_sparsity_quantile")]},
):
    df = copy.deepcopy(filtered_df)
    # Extraire les valeurs de b et c de la colonne 'col3'
    df["atk_type"] = df["ATK_objective"].str.extract(r"{'type': '(.+?)'")
    df["mask_type"] = df["ATK_objective"].str.extract(r": {'type': '(.+?)'")
    df["mask_type"] = df["mask_type"].fillna("no_mask")
    df["sparsity"] = df["ATK_objective"].str.extract(r"'sparsity': (\d+\.\d+)")
    df["sparsity"] = df["sparsity"].fillna("0.0")
    df["batch_size"] = df["datapoints"]
    for key, value_tab in squeeze_dict.items():
        for val_1, val_2 in value_tab:
            df[key] = df[key].replace(val_1, val_2)
    # Convertir les colonnes 'type' et 'sparsity' en type numérique
    # df['type'] = pd.to_numeric(df['type'])
    df["sparsity"] = pd.to_numeric(df["sparsity"])
    nb_experiment = len(df)

    model_list = df["model"].unique()
    # nb_model = len(model_list)
    x_list = df[x_axis].unique()
    nb_x = len(x_list)
    y_list = df[y_axis].unique()
    nb_y = len(y_list)

    for loss in loss_list:
        print(f"model: {model_list}")
        print(f"number of total experiment: {nb_experiment}")
        print(f"{x_axis}({nb_x}): {x_list}")
        print(f"{y_axis}({nb_y}): {y_list}")
        print(f"number of experiment for each (mask type&sparsity level): {nb_experiment / (nb_x * nb_y)}")
        grouped_df = df.groupby([x_axis, y_axis])[loss].agg(metric_list)

        # Tracer un graphique avec la valeur de "type"" en abscisse et la colonne 'ssim' en ordonnée
        fig = plt.figure(figsize=(24, 12), dpi=80)
        ax = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
        for i, metric in enumerate(metric_list):
            y = grouped_df.index.get_level_values(1).values
            z = np.array(grouped_df[metric].values)  # metric is `mean` or `var`

            if loss == "mse":
                result = z.reshape(nb_x, -1).transpose()
            elif loss == "ssim" or loss == "psnr":
                result = z.reshape(nb_x, -1).transpose()[::-1]
            colors = ["r", "b", "g"]

            ax[i].set_xlabel(x_axis, labelpad=10)
            ax[i].set_ylabel(y_axis, labelpad=10)
            ax[i].set_zlabel(loss.upper())
            xlabels = np.array(x_list)
            xpos = np.arange(xlabels.shape[0])
            if "mse" in loss:
                ylabels = y[:nb_y]
            elif ("ssim" in loss) or ("psnr" in loss):
                ylabels = y[:nb_y][::-1]
            ypos = np.arange(ylabels.shape[0])

            xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

            zpos = result
            zpos = zpos.ravel()

            dx = 0.5
            dy = 0.5
            dz = zpos

            ax[i].xaxis.set_ticks(xpos + dx / 2.0)
            ax[i].xaxis.set_ticklabels(xlabels)

            ax[i].yaxis.set_ticks(ypos + dy / 2.0)
            ax[i].yaxis.set_ticklabels(ylabels)

            values = np.linspace(0.2, 1.0, xposM.ravel().shape[0])
            colors = cm.rainbow(values)
            ax[i].bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
            ax[i].set_zlabel(metric)
            ax[i].set_title(f"{metric} {loss.upper()} en fonction {x_axis} et {y_axis}", fontsize=16)
        plt.show()


def plot_2d_df(
    filtered_df,
    x_axis="mask_type",
    y_axis="sparsity",
    loss_list=["mse", "ssim", "psnr"],
    metric_list=["mean", "var"],
    squeeze_dict={"mask_type": [("no_mask", "mask_sparsity_quantile")]},
):
    df = copy.deepcopy(filtered_df)
    # Extraire les valeurs de b et c de la colonne 'col3'
    df["atk_type"] = df["ATK_objective"].str.extract(r"{'type': '(.+?)'")
    df["mask_type"] = df["ATK_objective"].str.extract(r": {'type': '(.+?)'")
    df["mask_type"] = df["mask_type"].fillna("no_mask")
    df["sparsity"] = df["ATK_objective"].str.extract(r"'sparsity': (\d+\.\d+)")
    df["sparsity"] = df["sparsity"].fillna("0.0")
    df["batch_size"] = df["datapoints"]
    df["optimizer_type"] = df["ATK_optim"].str.extract(r"{'optimizer': '(.+?)'")
    for key, value_tab in squeeze_dict.items():
        for val_1, val_2 in value_tab:
            df[key] = df[key].replace(val_1, val_2)
    # Convertir les colonnes 'type' et 'sparsity' en type numérique
    # df['type'] = pd.to_numeric(df['type'])
    df["sparsity"] = pd.to_numeric(df["sparsity"])
    nb_experiment = len(df)

    model_list = df["model"].unique()
    # nb_model = len(model_list)
    x_list = df[x_axis].unique()
    nb_x = len(x_list)
    y_list = np.unique(df[y_axis])
    nb_y = len(y_list)

    # all_same_loss = np.array([("ssim" in loss) for loss in loss_list]).all()
    all_same_loss = False
    keyword_values_dict = {}
    keyword_values_total = 0
    for loss in loss_list:
        print(f"model: {model_list}")
        print(f"number of total experiment: {nb_experiment}")
        print(f"{x_axis}({nb_x}): {x_list}")
        print(f"{y_axis}({nb_y}): {y_list}")
        print(f"number of experiment for each ({x_axis}&{y_axis}): {nb_experiment / (nb_x * nb_y)}")
        if isinstance(y_axis, str):
            y_axis = [y_axis]

        # Group by keywords dynamically
        grouped_df = df.groupby([x_axis, *y_axis])[loss].agg(metric_list)

        # Extract the unique values for each keyword
        keyword_values_dict[loss] = np.stack(np.meshgrid(*[np.unique(df[y_ax]) for y_ax in y_axis]), -1).reshape(
            -1, len(y_axis)
        )
        keyword_values_total += len(keyword_values_dict[loss])
    if all_same_loss:
        fig = plt.figure(figsize=(24, 12), dpi=80)
        if len(metric_list) == 1:
            ax = [fig.add_subplot(111)]
        elif len(metric_list) == 2:
            ax = [fig.add_subplot(121), fig.add_subplot(122)]
        j = 0
        color = cm.viridis(np.linspace(0, 1, 2 * keyword_values_total))
        print(f"keyword_values_total: {keyword_values_total}")
    for loss, keyword_values in zip(loss_list, keyword_values_dict.values()):
        if not all_same_loss:
            fig = plt.figure(figsize=(24, 12), dpi=80)
            if len(metric_list) == 1:
                ax = [fig.add_subplot(111)]
            elif len(metric_list) == 2:
                ax = [fig.add_subplot(121), fig.add_subplot(122)]
        # Group by keywords dynamically
        grouped_df = df.groupby([x_axis, *y_axis])[loss].agg(metric_list)
        for i, metric in enumerate(metric_list):
            if not all_same_loss:
                color = cm.viridis(np.linspace(0, 1, len(keyword_values)))
                j = 0
            # Create a separate plot for each keyword combination
            for keyword_value_pairs in keyword_values:
                # Filter the grouped dataframe for each keyword combination
                df_kw = grouped_df[
                    np.logical_and.reduce(
                        [
                            (grouped_df.index.get_level_values(y_ax) == kw)
                            for y_ax, kw in zip(y_axis, keyword_value_pairs)
                        ]
                    )
                ]
                if len(df_kw[metric].values) > 0:
                    legend = " and ".join(f"{loss}: {y_ax}={kw}" for y_ax, kw in zip(y_axis, keyword_value_pairs))
                    x_values = df_kw[metric].index.get_level_values(x_axis)
                    ax[i].plot(
                        x_values,
                        df_kw[metric].values,
                        marker="o",
                        # linestyle="dashed",
                        color=color[j],
                        label=legend,
                    )
                    j += 1
            # if loss == "ssim":
            #     ax[i].hlines(0.3, min(x_values), max(x_values), colors="black", linestyles="dotted")
            if not all_same_loss:
                ax[i].set_xlabel(x_axis)
                ax[i].set_ylabel(loss)
                ax[i].set_title(f"{metric}: {loss} vs. {x_axis}")
                ax[i].legend()
        if not all_same_loss:
            plt.show()
    if all_same_loss:
        for i in range(len(metric_list)):
            ax[i].set_xlabel(x_axis)
            ax[i].set_ylabel("ssim")
            ax[i].set_title(f"{metric_list[i]}: ssim vs. {x_axis}")
            ax[i].legend()
        plt.show()

        # # Create a figure and axis for the plot
        # fig = plt.figure(figsize=(24, 12), dpi=80)
        # ax = [fig.add_subplot(121), fig.add_subplot(122)]
        # for i, metric in enumerate(metric_list):
        #     # Plot a line for each 'y' value
        #         # Filter the grouped dataframe for each 'y' value
        #         filters = {}
        #         for y_ax in y_axis:
        #             filters[y_ax] = row[y_ax]
        #         big_filters = np.logical_and([(grouped_df[y_ax] == filters[y_ax]) for y_ax in y_axis])
        #         df_bc = grouped_df[big_filters]
        #         ax[i].plot(
        #             df_bc.index.get_level_values(x_axis),
        #             df_bc.values,
        #             marker="o",
        #             label=",".join([f"{y_ax} = {filters[y_ax]}" for y_ax in y_axis]),
        #         )

        #     # Set the x-axis label and title
        #     ax[i].set_xlabel(x_axis)
        #     ax[i].set_ylabel(loss)
        #     ax[i].set_title(f"{metric} {loss.upper()} en fonction {x_axis} et {y_axis}", fontsize=16)

        #     # Add a legend
        #     ax[i].legend()

        # # Show the plot
        # plt.show()


def get_losses_from_log(file_path_log):
    log_file = open(file_path_log, "r")
    iterations, rec_loss, task_loss, time, metrics = [], [], [], [], {}
    for line in log_file.readlines():
        if "] | It: " in line:
            tab = line.split(" | ")
            for block in tab:
                if "It:" in block:
                    matched_pattern = re.findall(r"[\d]+", block)
                    if len(matched_pattern) == 1:
                        iterations.append(int(matched_pattern[0]))
                if "Rec. loss:" in block:
                    matched_pattern = re.findall(r"[\d]*[.][\d]+", block)
                    if len(matched_pattern) == 1:
                        rec_loss.append(float(matched_pattern[0]))
                if "Task loss:" in block:
                    matched_pattern = re.findall(r"[\d]*[.][\d]+", block)
                    if len(matched_pattern) == 1:
                        task_loss.append(float(matched_pattern[0]))
                if "T:" in block:
                    matched_pattern = re.findall(r"[\d]*[.][\d]+", block)
                    if len(matched_pattern) == 1:
                        time.append(float(matched_pattern[0]))
        if ("] METRICS: " in line) or ("SSIM" in line):
            tab = line.split(" | ")
            for block in tab:
                to_check_metrics = [
                    "MSE",
                    "PSNR",
                    "FMSE",
                    "LPIPS",
                    "R-PSNR",
                    "IIP-pixel",
                    "IIP-lpips",
                    "IIP-self",
                    "SSIM",
                    "max R-PSNR",
                    "max SSIM",
                    "Label Acc",
                ]
                for metric in to_check_metrics:
                    if metric in block:
                        matched_pattern = re.findall(r"[\d]*[.][\d]+", block)
                        if len(matched_pattern) == 1:
                            metrics[metric] = float(matched_pattern[0])
                        else:
                            raise ValueError(f"Error in log file: {file_path_log}")
        if "Computing user update on user" in line:
            pass

    return iterations, rec_loss, task_loss, time


def get_paired_from_log(file_path_log):
    log_file = open(file_path_log, "r")
    a, b, c = 0, 0, 0
    found = False
    line_number = 0
    for line in log_file.readlines():
        if line_number >= 3:
            break
        elif found:
            if "a:" in line:
                tab = line.split(":")
                a = int(tab[-1])
            elif "b:" in line:
                tab = line.split(":")
                b = int(tab[-1])
            elif "c:" in line:
                tab = line.split(":")
                c = int(tab[-1])
            line_number += 1
        if "paired:" in line:
            found = True
    return a, b, c


def get_load_round_from_log(file_path_log):
    log_file = open(file_path_log, "r")
    load_round = 0
    for line in log_file.readlines():
        if "load_round:" in line:
            load_round = int(line.split(":")[-1])
            break
    return load_round


def get_rec_loss_from_log(file_path_log):
    log_file = open(file_path_log, "r")
    rec_loss = {}
    for line in log_file.readlines():
        if "Now evaluating user" in line:
            rec_loss
        if "Rec. loss:" in line:
            rec_loss = float(line.split(":")[-1])
            break
    return rec_loss


def get_iteration_and_rec_loss_from_log(file_path_log):
    """
    Extracts the iteration and the reconstruction loss from a log file using regex expressions.
    """
    # Regex patterns
    user_pattern = r"user (\d+)"
    trial_pattern = r"trial (\d+)"
    it_pattern = r"It: (\d+)/\d+"
    rec_loss_pattern = r"Rec\. loss: ([\d.]+)"

    # Extracting information
    with open(file_path_log, "r") as f:
        logs = f.read()
    user = list(map(int, re.search(user_pattern, logs).group(1)))
    trial = list(map(int, re.search(trial_pattern, logs).group(1)))
    it_matches = list(map(int, re.findall(it_pattern, logs)))
    rec_loss_matches = list(map(float, re.findall(rec_loss_pattern, logs)))
    it_rec_pairs_matches = list(zip(it_matches, rec_loss_matches))

    return user, trial, it_rec_pairs_matches


# def get_iteration_and_rec_loss_from_df(df):
#     it_rec_pairs = []
#     for index, row in df.iterrows():
#         it_rec_pairs += get_iteration_and_rec_loss_from(row["folder"])
#     return it_rec_pairs


def plot_reconstruction_loss_from_df(df):
    # Extracting column names
    column_names = [col for col in df.columns if col.startswith("it_")]

    # Extracting specific columns
    specific_columns = df[column_names]
    reconstruction_loss = np.zeros((len(specific_columns), len(specific_columns.columns)))
    iter_values = []
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 5))
    color = cm.rainbow(np.linspace(0, 1, len(specific_columns)))
    for i, column in enumerate(specific_columns):
        iter_number = column.split("_")[1]  # Extract the number from the column name
        reconstruction_values = df[column].values
        reconstruction_loss[:, i] = reconstruction_values
        iter_values.append(iter_number)
    for i in range(reconstruction_loss.shape[0]):
        ax.plot(iter_values, reconstruction_loss[i], color=color[i], linestyle="dashed")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title(f"Reconstruction loss over iterations (4 significant digits) ({reconstruction_loss.shape[0]} curves)")
    ax.set_yscale("log")
    plt.show()
    # return reconstruction_loss, iter_values


def plot_conv_df(
    conv_df,
    date_and_run,
    merged_df=None,
    params_to_show=["load_round", "num_data_points", "num_data_per_local_update_step", "num_local_updates"],
):
    # Plot the convergence results
    sliced_df = conv_df[conv_df["file_path"].str.contains(date_and_run)]
    fig, ax = plt.subplots(figsize=(10, 5))
    if merged_df is not None:
        sliced_merged_df = merged_df[merged_df["file_path"].str.contains(date_and_run)]
        print(sliced_merged_df["load_round"].unique())
        str_params = "\n ".join([f"{p}: {sliced_merged_df[p].unique()}" for p in params_to_show])

    ax.set_title(f"Convergence results for {date_and_run}, {str_params}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reconstruction loss")
    ax.set_yscale("log")
    for user_idx in sliced_df["user_idx"].unique():
        user_df = sliced_df[sliced_df["user_idx"] == user_idx]
        rec_loss = user_df["Trial_0_Val"]
        iterations = user_df["step"]
        ax.plot(iterations, rec_loss, label=f"Run n°{user_idx}")
    ax.legend()
    plt.show()


def get_cfg_from_df(df):
    for _, row in df.iterrows():
        path = pathlib.Path(row["file_path"])
        path = path.parent.parent
        path = path / ".hydra/config.yaml"
        cfg = OmegaConf.load(path)
        df["cfg"] = OmegaConf.to_yaml(cfg)
    return df
