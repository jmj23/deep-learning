import time
import random
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk

root = Tk.Tk()

fig = Figure(figsize=(5,4), dpi=100)
subplot = fig.add_subplot(111)

res = 300
data = np.array([[random.random() for x in range(res)] for x in range(res)])
image = subplot.imshow(data, interpolation='nearest') 
mod = 1.03

def refresh():
    global data
    start = time.time()
    data *= mod
    data %= 1
    image.set_data(data)
    canvas.draw()
    print(round(1./(time.time()-start),2))
    root.after(0,refresh)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

root.after(0,refresh)
Tk.mainloop()