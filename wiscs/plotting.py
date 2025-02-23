from .simulate import DataGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.typing as npt
import seaborn as sns # type: ignore
import pandas as pd
from .utils import nearest_square_dims

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



