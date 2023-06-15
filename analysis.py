import re
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm


def get_df_from_config(df, real_config, path_to_config="/breaching/config/experiment"):
    # Sachant un df: pandas.DataFrame df, et un config: DictConfig, extrait les lignes du df qui correspondent au config
    pass


def plot_df(
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
            z = np.array(grouped_df[metric].values)

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
            if loss == "mse":
                ylabels = y[:nb_y]
            elif loss == "ssim" or "psnr":
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
