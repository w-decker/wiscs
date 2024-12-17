from .simulate import DataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt
from .methods import deltas, nearest_square_dims, pairwise_deltas

class Plot(DataGenerator):
    def __init__(self, DG: DataGenerator):
        self.__dict__ = DG.__dict__.copy()
    
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

        elif kwargs.get('idx') == 'trial':  
            rows, cols = nearest_square_dims(self.params["n"]["trial"])
            fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

            q = kwargs.get('question_idx')

            for ax, i in zip(axs.flatten(), range(self.params["n"]["trial"])):
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