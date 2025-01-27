from .simulate import DataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt
import seaborn as sns # type: ignore
import pandas as pd
from .utils import deltas, nearest_square_dims, pairwise_deltas

class Plot(DataGenerator):
    def __init__(self, DG: DataGenerator):
        self.__dict__ = DG.__dict__.copy()
        self.DG = DG
    
    def grid(self, **kwargs):
        """Plot grid of data distributions
        
        Parameters
        ----------
        kwargs: dict
            Keys: 'idx', 'question_idx'
        """

        if kwargs.get('idx') == 'participant':
            rows, cols = nearest_square_dims(self.params["n"]["participant"])
            fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

            for ax, i in zip(axs.flatten(), range(self.params["n"]["participant"])):
                ax.hist(self.data[0][i, :, :].ravel(), label='image', alpha=0.5)
                ax.hist(self.data[1][i, :, :].ravel(), label='word', alpha=0.5)
                ax.set_title(f'Participant {i+1}')

                ax.set_xlabel('RT')
                ax.set_ylabel('Frequency')

                ymax = (max(ax.get_ylim())/2).round(0)

                i_mean = self.data[0][i, :, :].ravel().mean()
                w_mean = self.data[1][i, :, :].ravel().mean()

                ax.scatter(i_mean, ymax, color='red', marker='o')
                ax.scatter(w_mean, ymax, color='red', marker='o')

                x_min, x_max = ax.get_xlim()
                xmin_frac = (w_mean - x_min) / (x_max - x_min)
                xmax_frac = (i_mean - x_min) / (x_max - x_min)
                ax.axhline(xmin=xmin_frac, xmax=xmax_frac, y=ymax, color='red', linestyle='--', label=r'$\Delta$ {}'.format(np.abs(i_mean - w_mean).round(2)))
                
                ax.legend()

            plt.show()
        
        elif kwargs.get('idx') == 'question':
            rows, cols = nearest_square_dims(self.params["n"]["question"])
            fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

            for ax, i in zip(axs.flatten(), range(self.params["n"]["question"])):
                ax.hist(self.data[0][:, i, :].ravel(), label='image', alpha=0.5)
                ax.hist(self.data[1][:, i, :].ravel(), label='word', alpha=0.5)
                ax.set_title(f'Question {i+1}')

                ax.set_xlabel('RT')
                ax.set_ylabel('Frequency')                

                i_mean = self.data[0][:, i, :].ravel().mean()
                w_mean = self.data[1][:, i, :].ravel().mean()

                ax.scatter(i_mean, 400, color='red', marker='o')
                ax.scatter(w_mean, 400, color='red', marker='o')

                x_min, x_max = ax.get_xlim()
                xmin_frac = (w_mean - x_min) / (x_max - x_min)
                xmax_frac = (i_mean - x_min) / (x_max - x_min)
                ax.axhline(xmin=xmin_frac, xmax=xmax_frac, y=400, color='red', linestyle='--', label=r'$\Delta$ {}'.format(np.abs(i_mean - w_mean).round(2)))
                
                ax.legend()
            plt.show()

        elif kwargs.get('idx') == "item":  
            rows, cols = nearest_square_dims(self.params["n"]["item"])
            fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

            q = kwargs.get('question_idx')

            for ax, i in zip(axs.flatten(), range(self.params["n"]["item"])):
                ax.hist(self.data[0][:, q, i].ravel(), label='image', alpha=0.5)
                ax.hist(self.data[1][:, q, i].ravel(), label='word', alpha=0.5)
                ax.set_title(f'Trial {i+1}')

                ax.set_xlabel('RT')
                ax.set_ylabel('Frequency')

                ymax = (max(ax.get_ylim())/2).round(0)

                i_mean = self.data[0][:, q, i].ravel().mean()
                w_mean = self.data[1][:, q, i].ravel().mean()

                ax.scatter(i_mean, ymax, color='red', marker='o')
                ax.scatter(w_mean, ymax, color='red', marker='o')

                x_min, x_max = ax.get_xlim()
                xmin_frac = (w_mean - x_min) / (x_max - x_min)
                xmax_frac = (i_mean - x_min) / (x_max - x_min)
                ax.axhline(xmin=xmin_frac, xmax=xmax_frac, y=ymax, color='red', linestyle='--', label=r'$\Delta$ {}'.format(np.abs(i_mean - w_mean).round(2)))
                
                ax.legend()
            plt.show()
            
    def plot_bargraph(self, show_interaction:bool, title: str, hypothesis_title:str, point_alpah:float=0.2, point_size:int=1):
    
        df = self.DG.to_pandas()
        
        # Ensure 'question' is a categorical variable
        df['question'] = pd.Categorical(df['question'])

        colors = {'word': 'dodgerblue', 'image': 'forestgreen'}
        meanc = {'word': 'red', 'image': 'red'}

        if show_interaction:
            fig, ax = plt.subplots(2, 1, figsize=(15, 7))

            # Bar plots
            sns.barplot(
                x='question', y='rt', hue='modality', data=df, ax=ax[0],
                errorbar='sd', alpha=0.4, err_kws={'linewidth': 1.5},
                capsize=0.1, palette=colors, edgecolor='black', dodge=True, legend=False
            )
            sns.barplot(
                x='question', y='rt', hue='modality', data=df, ax=ax[1],
                errorbar='sd', alpha=0.4, err_kws={'linewidth': 1.5},
                capsize=0.1, palette=colors, edgecolor='black', dodge=True
            )

            sns.stripplot(
                x='question', y='rt', hue='modality', data=df, ax=ax[1],
                dodge=True, palette=colors, alpha=0.2, size=1, legend=False)

            means = df.groupby(['question', 'modality'], as_index=False).mean()

            # Define dodge amount based on the number of modalities
            dodge_amount = 0.4  # Adjust as needed
            modalities = ['word', 'image']
            modality_offsets = {modality: i * dodge_amount - dodge_amount / 2 for i, modality in enumerate(modalities)}

            # Apply dodge adjustment to x positions
            means['x_dodge'] = means.apply(lambda row: int(row['question']) + modality_offsets[row['modality']], axis=1)

            # Lines for each modality with dodged x values
            for modality in modalities:
                modality_means = means[means['modality'] == modality]
                ax[0].plot(
                    modality_means['x_dodge'], modality_means['rt'],
                    marker='o', color=meanc[modality], label=modality, linewidth=2
                )

            # Adjust legend placement
            plt.legend(loc='center left', bbox_to_anchor=(1, 1.1), title='Modality')

            # Clean up axes
            for a in ax:
                a.set_ylabel('Reaction time (ms)')
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

            ax[0].set_xlabel(None)
            ax[1].set_xlabel('Question')

            # Titles
            plt.suptitle(title, fontsize=16, y=1)
            plt.figtext(0.5, 0.9, hypothesis_title, ha='center', va='center', fontsize=12)

            plt.tight_layout()
            plt.show()

        else:
            plt.figure(figsize=(15, 3))

            sns.barplot(
                x='question', y='rt', hue='modality', data=df,
                errorbar='sd', alpha=0.4, err_kws={'linewidth': 1.5},ax=plt.gca(),
                capsize=0.1, palette=colors, edgecolor='black', dodge=True
            )

            sns.stripplot(
                x='question', y='rt', hue='modality', data=df, ax=plt.gca(),
                dodge=True, palette=colors, alpha=point_alpah, size=point_size, legend=False)
            
            # Adjust legend placement
            plt.title(title, fontsize=16, y=1.07)
            plt.figtext(0.5, 0.9, hypothesis_title, ha='center', va='center', fontsize=12)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Modality')
            plt.ylim(0, 800)
            plt.ylabel('Reaction time (ms)')
            plt.xlabel('Question')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)

def plot_deltas(DG1:DataGenerator, DG2:DataGenerator, idx:str, labels:list[str]) -> None:
    """Plot deltas
    """
    plt.plot(deltas(DG1, idx), marker='o', label=labels[0])
    plt.plot(deltas(DG2, idx), marker='o', label=labels[1])
    plt.title("$\\Delta$ in modality across {} and hypotheses".format(idx.capitalize()))

    plt.xlabel(idx.capitalize())
    plt.ylabel("$\\Delta$")

    plt.legend()

    plt.show()

def plot_pairwise_deltas(DG1: DataGenerator, DG2: DataGenerator, idx: str, labels: list[str]):
    """Plot pairwise deltas
    """
    # Calculate pairwise deltas
    deltas1 = np.tril(pairwise_deltas(DG1, idx=idx))
    deltas2 = np.tril(pairwise_deltas(DG2, idx=idx))

    # Determine the common color range
    vmin = min(deltas1.min(), deltas2.min())
    vmax = max(deltas1.max(), deltas2.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    a1 = ax1.imshow(deltas1, vmin=vmin, vmax=vmax)
    ax1.set_title(labels[0])
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a1, cax=cax1)

    a2 = ax2.imshow(deltas2, vmin=vmin, vmax=vmax)
    ax2.set_title(labels[1])
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(a2, cax=cax2)

    ticks = np.arange(deltas1.shape[0])
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)

    ax1.set_ylabel(f'{idx.capitalize()} Index')
    ax1.set_xlabel(f'{idx.capitalize()} Index')

    ax2.set_ylabel(f'{idx.capitalize()} Index')
    ax2.set_xlabel(f'{idx.capitalize()} Index')

    plt.subplots_adjust(wspace=0.4)
    plt.show()

def plot_scatter(DG1:DataGenerator, DG2:DataGenerator, idx:str, labels:list[str]):

    n = np.arange(1, DG1.params["n"][idx]+1)
    imagem = [DG1.data[0][:, i, :].mean() for i in range(DG1.params["n"][idx])]
    imagee = [DG1.data[0][:, i, :].std() for i in range(DG1.params["n"][idx])]  
    wordm = [DG1.data[1][:, i, :].mean() for i in range(DG1.params["n"][idx])]
    worde = [DG1.data[1][:, i, :].std() for i in range(DG1.params["n"][idx])]
    
    imagem1 = [DG2.data[0][:, i, :].mean() for i in range(DG2.params["n"][idx])]
    imagee1 = [DG2.data[0][:, i, :].std() for i in range(DG2.params["n"][idx])]
    wordm1 = [DG2.data[1][:, i, :].mean() for i in range(DG2.params["n"][idx])]
    worde1 = [DG2.data[1][:, i, :].std() for i in range(DG2.params["n"][idx])]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 10))

    ax1.errorbar(n, wordm, yerr=worde, fmt='o', color='blue', label='Word', capsize=5)
    ax1.errorbar(n, imagem, yerr=imagee, fmt='^', color='green', label='Image', capsize=5)
    ax1.set_xlabel(idx.capitalize())
    ax1.set_ylabel('Mean Score')
    ax1.set_title(labels[0])
    ax1.legend()

    ax2.errorbar(n, wordm1, yerr=worde1, fmt='s', color='red', label='Word', capsize=5, alpha=0.5)
    ax2.errorbar(n, imagem1, yerr=imagee1, fmt='d', color='orange', label='Image', capsize=5, alpha=0.5)
    ax2.set_xlabel(idx.capitalize())
    ax2.set_ylabel('Mean RT')
    ax2.set_title(labels[1])
    ax2.legend()

    ax1.set_xticks(n)
    ax2.set_xticks(n)

    # Adjust space between subplots
    plt.subplots_adjust(wspace=0.6)

    plt.show()

