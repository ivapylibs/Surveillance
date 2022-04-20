import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import time

@dataclass
class ParamDynamicDisplay:
    figsize: tuple = (6,4)
    fontsize: int = 6
    status_label: list = field(default_factory=lambda: ['UNKNOWN', 'MEASURED', 'TRACKED', 'INHAND', 'INVISIBLE', 'GONE'])
    xlimit: int = 20 # @< Time window.
    ylimit: int = 5  # @< Related to the label numbers for the piece status.
    id: int = 0

class DynamicDisplay():

    def __init__(self, param=ParamDynamicDisplay):
        self.param = param

        # Set up plot
        self.figure, self.ax = plt.subplots(figsize=self.param.figsize)
        self.line, = self.ax.plot([], [], lw=2)

        self.ax.set_xlim(0, self.param.xlimit)
        self.ax.set_ylim(0, self.param.ylimit)
        self.ax.set_xticks(np.arange(0, self.param.xlimit + 1, 1), fontsize=self.param.fontsize)
        self.ax.set_yticks(np.arange(0, self.param.ylimit + 1, 1), fontsize=self.param.fontsize)

        self.ax.set_yticklabels(self.param.status_label, fontsize=self.param.fontsize)

        self.ax.set_xlabel('Time', fontsize=self.param.fontsize)
        self.ax.set_ylabel('Status', fontsize=self.param.fontsize)
        self.ax.set_title(f'ID {self.param.id}: Status Changes', fontsize=self.param.fontsize)

        self.ax.grid()

        self.xdata = []
        self.ydata = []

    def __call__(self, data):
        self.xdata.append(data[0])
        self.ydata.append(data[1])

        xmin, xmax = self.ax.get_xlim()

        if data[0] >= xmax:
            self.ax.set_xlim(xmin + self.param.xlimit / 2, xmax + self.param.xlimit / 2)
            self.ax.figure.canvas.draw()

            self.ax.set_xticks(np.arange(xmin + self.param.xlimit / 2, xmax + self.param.xlimit / 2 + 1, 1))
            self.ax.set_xticklabels(np.arange(0, self.param.xlimit + 1, 1))
            self.ax.set_yticks(np.arange(0, self.param.ylimit + 1, 1))

        self.line.set_data(self.xdata, self.ydata)
        self.figure.canvas.flush_events()

if __name__ == '__main__':
    plt.ion()
    d = DynamicDisplay()
    for i in range(5000):
        d((i, np.random.randint(4)))
