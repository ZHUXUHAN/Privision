import sys
from PyQt5.QtWidgets import *
import qdarkstyle
import argparse


from mainwindow_privision import MainWindow

__appname__ = 'PriVision'

# Parse arguments
parser = argparse.ArgumentParser(description='PriVision Config')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/person_analysis_config.yaml', type=str)
args = parser.parse_args()
print('==> Called with args:')
print(args)


def main(argv):
    """Standard boilerplate Qt application code."""
    app = QApplication(argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setApplicationName(__appname__)
    # app.setWindowIcon(QIcon('./files/pose_icon.png'))
    # app.setWindowIcon(newIcon("app"))
    win = MainWindow(__appname__, args.cfg_file)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
