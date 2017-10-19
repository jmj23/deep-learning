import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pyqtgraph as pg


class sinus_wave(QtWidgets.QMainWindow):
    def __init__(self):
        super(sinus_wave, self).__init__()
        self.init_ui()
        self.qt_connections()
        self.plotcurve = pg.PlotCurveItem()
        self.plotwidget.addItem(self.plotcurve)
        self.amplitude = 10
        self.t = 0
        self.updateplot()

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.moveplot)
        self.timer.start(50)
        
    def init_ui(self):
        self.setWindowTitle('Sinuswave')
        hbox = QtWidgets.QVBoxLayout()
        self.setLayout(hbox)

        self.plotwidget = pg.PlotWidget()
        hbox.addWidget(self.plotwidget)

        self.increasebutton = QtWidgets.QPushButton("Increase Amplitude")
        self.decreasebutton = QtWidgets.QPushButton("Decrease Amplitude")

        hbox.addWidget(self.increasebutton)
        hbox.addWidget(self.decreasebutton)

        self.setGeometry(10, 10, 1000, 600)
        self.show()

    def qt_connections(self):
        self.increasebutton.clicked.connect(self.on_increasebutton_clicked)
        self.decreasebutton.clicked.connect(self.on_decreasebutton_clicked)

    def moveplot(self):
        self.t+=1
        self.updateplot()

    def updateplot(self):
        print("Update")
        data1 = self.amplitude*np.sin(np.linspace(0,30,121)+self.t)
        self.plotcurve.setData(data1)

    def on_increasebutton_clicked(self):
        print("Amplitude increased")
        self.amplitude += 1
        self.updateplot()

    def on_decreasebutton_clicked(self):
        print("Amplitude decreased")
        self.amplitude -= 1
        self.updateplot()
        
    def closeEvent(self, evnt):
        self.timer.stop()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Sinuswave')
    ex = sinus_wave()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()