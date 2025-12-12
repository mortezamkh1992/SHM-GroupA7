import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def HI_graph(X, dir="", name="", legend=True):
    #Graph of HI against cycles
    #X is numpy list of HI at different states, & name is root to save at

    samples = ["PZT-FFT-HLB-L1-03", "PZT-FFT-HLB-L1-04", "PZT-FFT-HLB-L1-05", "PZT-FFT-HLB-L1-09", "PZT-FFT-HLB-L1-23"]
    markers = ['o', 's', '^', 'D', 'X']
    colours = ['purple', 'blue', 'red', 'green', 'orange']

    plt.figure()
    font = 20
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18

    for sample in range(len(X)):
        states = np.arange(len(X[sample]))
        cycles = states/30*100
        if str(sample) == str(name[-1]) or samples[sample] == name[:-7]:
            plt.plot(cycles, X[sample], marker=markers[sample], color=colours[sample], label="Sample "+str(sample+1) + ": Test")
        else:
            plt.plot(cycles, X[sample], marker=markers[sample], color=colours[sample], label="Sample " + str(sample+1) + ": Train")
    if legend:
        plt.legend()

    plt.xlabel('Lifetime (%)', fontsize=font)
    plt.ylabel('HI', fontsize=font)
    plt.tight_layout()
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name)
    else:
        plt.show()
    plt.close()

def criteria_chart(features, Mo, Pr, Tr, dir="", name=""):
    #Stacked bar chart of criteria against features
    #Features is list of feature names; Mo, Pr, Tr are numpy lists of floats in same order, dir is root to save at
    plt.figure()
    plt.bar(features, Mo, label="Mo")
    plt.bar(features, Pr, bottom=Mo, label="Pr")
    plt.bar(features, Tr, bottom=Pr+Mo, label="Tr")
    plt.legend()
    if features[0] == "050":
        plt.xlabel('Frequency (kHz)')
    else:
        plt.xlabel('Feature')
    plt.ylabel('Fitness')
    if dir != "" and name != "":
        plt.savefig(dir + "\\" + name + " PC")
    else:
        plt.show()
    plt.close()


def big_plot(dir, type, transform):
    """
        Assemble grid of HI graphs

        Parameters:
        - dir (str): Directory of HI graphs
        - type (string): "DeepSAD" or "VAE"
        - transform (string): "FFT" or "HLB"

        Returns: None
    """

    # Define variables
    panels = ("0", "1", "2", "3", "4")
    freqs = ("050", "100", "125", "150", "200", "250")

    markers = ['o', 's', '^', 'D', 'X']
    colours = ['purple', 'blue', 'red', 'green', 'orange']
    labels = ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5']
    #legend_data = [Line2D([0], [0], marker=marker, color=colour, markerfacecolor=colour, markersize=10, label=label) for marker, colour, label in zip(markers, colours, labels)]

    nrows = len(freqs)+1
    ncols = len(panels)
    fig, axs = plt.subplots(nrows, ncols, figsize=(37, 40))  # Adjusted figure size

    # For each frequency and panel
    for i, freq in enumerate(freqs):
        for j, panel in enumerate(panels):
            # Generate the filename
            filename = f"{freq}kHz_{type}_{transform}_{j}.png"

            # Check if the file exists
            if os.path.exists(os.path.join(dir, filename)):
                # Load the image
                img = mpimg.imread(os.path.join(dir, filename))

                # Display the image in the corresponding subplot
                axs[i, j].imshow(img)
                axs[i, j].axis('off')  # Hide the axes
            else:
                # If the image does not exist, print a warning and leave the subplot blank
                axs[i, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
                axs[i, j].axis('off')

    # WAE results
    for j, panel in enumerate(panels):
        filename = f"WAE_{type}_{transform}_{j}.png"
        if os.path.exists(os.path.join(dir, filename)):
            img = mpimg.imread(os.path.join(dir, filename))
            axs[-1, j].imshow(img)
            axs[-1, j].axis('off')

        else:
            axs[-1, j].text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12, color='red')
            axs[-1, j].axis('off')

    # Redefine freqs to include kHz
    freqs = ("050 kHz", "100 kHz", "125 kHz", "150 kHz", "200 kHz", "250 kHz")

    # Add row labels
    for ax, row in zip(axs[:, 0], freqs):
        ax.annotate(f'{row}', (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold', fontsize=40)
    axs[-1, 0].annotate("Fusion", (-0.1, 0.5), xycoords='axes fraction', rotation=90, va='center', fontweight='bold', fontsize=40)

    # Add column labels
    for ax, col in zip(axs[0], panels):
        ax.annotate(f'    Test Sample {panels.index(col) + 1}', (0.5, 1), xycoords='axes fraction', ha='center', fontweight='bold', fontsize=40)

    #fig.legend(handles=legend_data, loc="center", bbox_to_anchor=(0.5, 0.03), ncol=5, fontsize=40)

    # Adjust spacing between subplots and save
    # plt.tight_layout()
    plt.subplots_adjust(left=0.0, right=1.01, top=0.98, bottom=0.05, hspace=-0.03, wspace=-0.2)

    plt.savefig(os.path.join(dir, f"BigPlot_{type}_{transform}.pdf"))
    plt.savefig(os.path.join(dir, f"BigPlot_{type}_{transform}.png"))

def plot_sensitivity_single_vae(csv_path, out_path, gamma_slice=None):
    """
        Plot VAE sensitivity surfaces. For each gamma, we plot a 3D surface for F_test and a wireframe for F_all.

        Parameters:
        - csv_path (str): Directory containing sensitivity analysis data
        - out_path (str): Directory of output file
        - gamma_slice (slice or None): Optional slice applied to the sorted list
          of gamma values for reduced plots.

        Returns: None
    """

    df = pd.read_csv(csv_path)

    for col in ["mean_fitness_all", "mean_fitness_test"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean_fitness_all", "mean_fitness_test"])

    gammas = sorted(df["gamma"].unique())
    if gamma_slice is not None:
        gammas = gammas[gamma_slice]

    zmin, zmax = 1.00, 2.60

    cmin, cmax = 1.0, 2.10
    norm = Normalize(vmin=cmin, vmax=cmax)

    fig = plt.figure(figsize=(4 * len(gammas), 3.5))

    last_surf = None

    for i, g in enumerate(gammas, start=1):
        sub = df[np.isclose(df["gamma"], g)]

        alphas = np.sort(sub["alpha"].unique())
        betas  = np.sort(sub["beta"].unique())

        Z_all = sub.pivot_table(index="beta", columns="alpha",
                                values="mean_fitness_all").reindex(
                                    index=betas, columns=alphas).values
        Z_test = sub.pivot_table(index="beta", columns="alpha",
                                 values="mean_fitness_test").reindex(
                                     index=betas, columns=alphas).values

        X, Y = np.meshgrid(alphas, betas)

        ax = fig.add_subplot(1, len(gammas), i, projection='3d')

        surf = ax.plot_surface(X, Y, Z_test, cmap="plasma", norm=norm, alpha=0.95)
        wire = ax.plot_wireframe(X, Y, Z_all, color="black", linewidth=1.0)

        last_surf = surf

        ax.set_title(f"γ = {g}")
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_zlabel("Fitness", labelpad=8)
        ax.view_init(elev=30, azim=45)
        ax.invert_yaxis()
        ax.invert_xaxis()

        ax.set_zlim(zmin, zmax)
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        if i == 1:
            ax.legend([wire, surf], ["F_all", "F_test"], loc="upper left")

    if last_surf is not None:
        cax = fig.add_axes([0.92, 0.20, 0.015, 0.65])
        fig.colorbar(last_surf, cax=cax, label="F_test")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(out_path,
                dpi=300,
                bbox_inches="tight",
                format="pdf")
    plt.close(fig)

def plot_vae_sensitivity(csv_path):
    """
        Plot VAE sensitivity figures for both FFT and HLB VAE sensitivity CSVs.

        Parameters:
        - csv_path (str): Directory containing sensitivity analysis data

        Returns: None
    """

    files = [
        ("fft",
         os.path.join(csv_path,
                      "VAE_sensitivity_averaged_FFT_FT_Reduced.csv")),
        ("hlb",
         os.path.join(csv_path,
                      "VAE_sensitivity_averaged_HLB_FT_Reduced.csv")),
    ]

    for label, dir_path in files:
        out_path = os.path.join(csv_path, f"VAE_sensitivity_{label}.pdf")
        plot_sensitivity_single_vae(dir_path, out_path)

        out_path_r = os.path.join(csv_path, f"VAE_sensitivity_{label}_reduced.pdf")
        plot_sensitivity_single_vae(dir_path, out_path_r, gamma_slice=slice(3, 5))

    print("Plotted VAE sensitivity analysis.")

def plot_sensitivity_single_deepsad(csv_path, out_path, lambda_slice=None):
    """
        Plot DeepSAD sensitivity surfaces. For each lambda, plot a 3D surface for F_test and a wireframe for F_all.

        Parameters:
        - csv_path (str): Directory containing sensitivity analysis data
        - out_path (str): Directory of output file
        - lambda_slice (slice or None): Optional slice applied to the sorted list
          of lambda values for reduced plots.

        Returns: None
    """

    df = pd.read_csv(csv_path)

    for col in ["mean_fitness_all", "mean_fitness_test"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mean_fitness_all", "mean_fitness_test"])

    df["log_nu"] = np.log10(df["nu"])
    df["log_eta"] = np.log10(df["eta"])

    lambdas = sorted(df["lambda"].unique())
    if lambda_slice is not None:
        lambdas = lambdas[lambda_slice]

    zmin, zmax = 1.00, 2.60
    cmin, cmax = 1.00, 2.10
    norm = Normalize(vmin=cmin, vmax=cmax)

    fig = plt.figure(figsize=(4 * len(lambdas), 3.5))

    last_surf = None

    for i, lam in enumerate(lambdas, start=1):
        sub = df[np.isclose(df["lambda"], lam)]
        if sub.empty:
            continue

        nus = np.sort(sub["log_nu"].unique())
        etas = np.sort(sub["log_eta"].unique())

        Z_all = sub.pivot_table(index="log_eta", columns="log_nu",
                                values="mean_fitness_all").reindex(
                                    index=etas, columns=nus).values
        Z_test = sub.pivot_table(index="log_eta", columns="log_nu",
                                 values="mean_fitness_test").reindex(
                                     index=etas, columns=nus).values

        X, Y = np.meshgrid(nus, etas)

        ax = fig.add_subplot(1, len(lambdas), i, projection='3d')

        surf = ax.plot_surface(X, Y, Z_test, cmap="plasma", norm=norm, alpha=0.95)
        wire = ax.plot_wireframe(X, Y, Z_all, color="black", linewidth=1.0)

        last_surf = surf

        ax.set_title(f"λ = {lam}")
        ax.set_xlabel(r"$\log_{10}(\nu)$")
        ax.set_ylabel(r"$\log_{10}(\eta)$")
        ax.set_zlabel("Fitness", labelpad=8)
        ax.view_init(elev=30, azim=45)

        ax.set_zlim(zmin, zmax)

        ax.set_xticks([-3, -2, -1, 0, 1])
        ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
        ax.set_yticks([-3, -2, -1, 0, 1])
        ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10"])
        ax.invert_yaxis()
        ax.invert_xaxis()

        ax.zaxis.set_major_locator(plt.MaxNLocator(5))

        if i == 1:
            legend_handles = [
                Line2D([0], [0], color="black", linewidth=2),
                Patch(facecolor="orange", edgecolor="none")
            ]
            legend_labels = ["F_all", "F_test"]
            ax.legend(legend_handles, legend_labels, loc="upper left")

    if last_surf is not None:
        cax = fig.add_axes([0.92, 0.20, 0.015, 0.65])
        fig.colorbar(last_surf, cax=cax, label="F_test")

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(out_path,
                dpi=300,
                bbox_inches="tight",
                format="pdf")
    plt.close(fig)

def plot_deepsad_sensitivity(csv_path):
    """
        Plot DeepSAD sensitivity figures for both FFT and HLB DeepSAD sensitivity CSVs.

        Parameters:
        - csv_path (str): Directory containing sensitivity analysis data

        Returns: None
    """

    files = [
        ("fft",
         os.path.join(csv_path,
                      "deepsad_sensitivity_averaged_FFT_FT_Reduced.csv")),
        ("hlb",
         os.path.join(csv_path,
                      "deepsad_sensitivity_averaged_HLB_FT_Reduced.csv")),
    ]

    for label, dir_path in files:
        out_path = os.path.join(csv_path, f"DeepSAD_sensitivity_{label}.pdf")
        plot_sensitivity_single_deepsad(dir_path, out_path)

        out_path_r = os.path.join(csv_path, f"DeepSAD_sensitivity_{label}_reduced.pdf")
        plot_sensitivity_single_deepsad(dir_path, out_path_r, lambda_slice=slice(0, 2))

    print("Plotted DeepSAD sensitivity analysis.")

if __name__ == "__main__":
    plot_vae_sensitivity(r"C:\Users\Pablo\OneDrive - Delft University of Technology\Desktop\TUDelft\Sensitivity_Final")
    plot_deepsad_sensitivity(r"C:\Users\Pablo\OneDrive - Delft University of Technology\Desktop\TUDelft\Sensitivity_Final")