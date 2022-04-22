import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import time

from Surveillance.deployment.utils import calc_closest_factors

@dataclass
class ParamDynamicDisplay:
    figsize: tuple = (6,4)
    fontsize: int = 4
    window_title: str = 'Status Change'
    status_label: list = field(default_factory=lambda: ['UNKNOWN', 'MEASURED', 'TRACKED', 'INHAND', 'INVISIBLE', 'GONE'])
    xlimit: int = 20 # @< Time window.
    ylimit: int = 5  # @< Related to the label numbers for the piece status.
    num: int = 8

class DynamicDisplay():

    def __init__(self, param=ParamDynamicDisplay):
        self.param = param

        # row, col
        self.y_num, self.x_num = calc_closest_factors(self.param.num)
        # Set up plot
        self.figure, self.ax = plt.subplots(self.y_num, self.x_num, figsize=self.param.figsize)

        # Reshape ax
        if self.y_num == 1:
            if self.x_num != 1:
                self.ax = self.ax.reshape(1,-1)
            else:
                self.ax = np.array([[self.ax]])

        self.line = [[] for _ in range(self.param.num)]
        self.axbg_cache = [[] for _ in range(self.param.num)]

        for i in range(self.y_num):
            for j in range(self.x_num):

                self.line[i*self.x_num+j], = self.ax[i,j].plot([], [], lw=2, label= f'ID {i*self.x_num+j}')

                self.ax[i,j].set_xlim(0, self.param.xlimit)
                self.ax[i,j].set_ylim(0, self.param.ylimit)
                self.ax[i,j].set_xticks(np.arange(0, self.param.xlimit + 1, 2), fontsize=self.param.fontsize)
                self.ax[i,j].set_yticks(np.arange(0, self.param.ylimit + 1, 1), fontsize=self.param.fontsize)

                self.ax[i,j].set_yticklabels(self.param.status_label, fontsize=self.param.fontsize)

                self.ax[i,j].set_xlabel('Time', fontsize=self.param.fontsize)
                self.ax[i,j].set_ylabel('Status', fontsize=self.param.fontsize)
                self.ax[i, j].legend(loc='upper right')

                self.ax[i,j].grid()
                self.ax[i,j].set_xticks([])
                self.ax[i, j].set_yticks([])

        self.figure.canvas.set_window_title(self.param.window_title)
        self.figure.canvas.draw()

        self.xdata = [[] for _ in range(self.param.num)]
        self.ydata = [[] for _ in range(self.param.num)]

    def __call__(self, data, blit=False):

        for idx in range(len(data[1])):

            self.xdata[idx].append(data[0])
            self.ydata[idx].append(data[1][idx])

            i = idx//self.x_num
            j = idx-i*self.x_num

            xmin, xmax = self.ax[i,j].get_xlim()

            if data[0] >= xmax:
                self.ax[i,j].set_xlim(xmin + self.param.xlimit / 2, xmax + self.param.xlimit / 2)
                # self.ax[i,j].figure.canvas.draw()

                # self.ax[i,j].set_xticks(np.arange(xmin + self.param.xlimit / 2, xmax + self.param.xlimit / 2 + 1, 2))
                # # Relabel from t=0 to t=self.param.xlimit
                # self.ax[i,j].set_xticklabels(np.arange(0, self.param.xlimit + 1, 2))
                #
                # self.ax[i, j].set_xticks([])

            self.line[idx].set_data(self.xdata[idx], self.ydata[idx])


        self.figure.canvas.flush_events()

if __name__ == '__main__':

    plt.ion()
    d = DynamicDisplay(ParamDynamicDisplay(num=9))
    for i in range(5000):
        d((i, np.random.randint(4, size=d.param.num)), blit=True)
