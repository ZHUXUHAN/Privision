# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import datetime
import pprint
import importlib

from priv_config import cfg_priv, merge_priv_cfg_from_file
from libs.lib import *
from libs.canvas import *
from libs.zoomwidget import *
from libs.cameradevice import *
from libs.utils import *

from mainwindow_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = range(3)
    resized = QtCore.pyqtSignal()

    def __init__(self, app_name, cfg_file):
        super(MainWindow, self).__init__()

        # setup config
        self.cfg_file = cfg_file
        self.setup_config()

        # setup workers
        self.worker01 = GetImage()
        self.worker02 = GetImage()
        self.worker03 = GetImage()
        self.worker04 = GetImage()
        self.worker05 = GetImage()
        self.setup_workers()

        # setup ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.banner.setText(cfg_priv.GLOBAL.BANNER)

        # canvas main window
        self.setWindowIcon(QIcon(cfg_priv.ICONS.LOGO))
        self.canvas = Canvas()
        self.canvas.image = QImage(cfg_priv.ICONS.BACKGROUND)
        self.canvas.zoomRequest.connect(self.zoomRequest)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.canvas.scrollRequest.connect(self.scrollRequest)
        self.ui.main_video_layout.addWidget(scroll)

        self.zoomWidget = ZoomWidget()
        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"), fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self.zoomMode = self.FIT_WINDOW
        self.canvas.setEnabled(True)
        self.adjustScale(initial=True)
        self.paintCanvas()
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        self.resized.connect(self.adjustScale)

        # camera
        self.camera_device = CameraDevice(video_path=cfg_priv.ICONS.BACKGROUND)
        self.camera_device.newFrame.connect(self.onNewImage)
        self.camera_device.video_time_out.connect(self.clear)

        # top button functions
        self.ui.thread1_bn.clicked.connect(self.start_camera)
        self.ui.thread2_bn.clicked.connect(self.start_video1)
        self.ui.thread3_bn.clicked.connect(self.start_video2)
        self.ui.thread4_bn.clicked.connect(self.start_pic)
        self.ui.thread5_bn.clicked.connect(self.load_video)
        # left button functions
        self.ui.play_bn.clicked.connect(self.start_cap)
        self.ui.pause_bn.clicked.connect(self.stop_cap)
        self.ui.record_bn.clicked.connect(self.save_video)
        self.ui.exit_bn.clicked.connect(self.close)
        # right button functions
        self.ui.model1_bn.clicked.connect(self.apply_model1)
        self.ui.model2_bn.clicked.connect(self.apply_model2)
        self.ui.model3_bn.clicked.connect(self.apply_model3)
        self.ui.model4_bn.clicked.connect(self.apply_model4)
        self.ui.model5_bn.clicked.connect(self.apply_model5)

        # top button functions
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1))
        self.ui.thread1_bn.setIconSize(QSize(cfg_priv.ICONS.TOP.SIZE[0], cfg_priv.ICONS.TOP.SIZE[1]))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2))
        self.ui.thread2_bn.setIconSize(QSize(cfg_priv.ICONS.TOP.SIZE[0], cfg_priv.ICONS.TOP.SIZE[1]))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3))
        self.ui.thread3_bn.setIconSize(QSize(cfg_priv.ICONS.TOP.SIZE[0], cfg_priv.ICONS.TOP.SIZE[1]))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4))
        self.ui.thread4_bn.setIconSize(QSize(cfg_priv.ICONS.TOP.SIZE[0], cfg_priv.ICONS.TOP.SIZE[1]))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5))
        self.ui.thread5_bn.setIconSize(QSize(cfg_priv.ICONS.TOP.SIZE[0], cfg_priv.ICONS.TOP.SIZE[1]))
        # left button functions
        self.ui.play_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP1))
        self.ui.play_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        self.ui.pause_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP2))
        self.ui.pause_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        self.ui.record_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP3))
        self.ui.record_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        self.ui.empty_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP4))
        self.ui.empty_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        self.ui.setting_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP5))
        self.ui.setting_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        self.ui.exit_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP6))
        self.ui.exit_bn.setIconSize(QSize(cfg_priv.ICONS.LEFT.SIZE[0], cfg_priv.ICONS.LEFT.SIZE[1]))
        # right button icons
        self.ui.model1_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC1.ICON))
        self.ui.model1_bn.setIconSize(QSize(cfg_priv.FUNC_OPT.FUNC1.ICON_SIZE[0], cfg_priv.FUNC_OPT.FUNC1.ICON_SIZE[1]))
        self.ui.model2_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC2.ICON))
        self.ui.model2_bn.setIconSize(QSize(cfg_priv.FUNC_OPT.FUNC2.ICON_SIZE[0], cfg_priv.FUNC_OPT.FUNC2.ICON_SIZE[1]))
        self.ui.model3_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC3.ICON))
        self.ui.model3_bn.setIconSize(QSize(cfg_priv.FUNC_OPT.FUNC3.ICON_SIZE[0], cfg_priv.FUNC_OPT.FUNC3.ICON_SIZE[1]))
        self.ui.model4_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC4.ICON))
        self.ui.model4_bn.setIconSize(QSize(cfg_priv.FUNC_OPT.FUNC4.ICON_SIZE[0], cfg_priv.FUNC_OPT.FUNC4.ICON_SIZE[1]))
        self.ui.model5_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC5.ICON))
        self.ui.model5_bn.setIconSize(QSize(cfg_priv.FUNC_OPT.FUNC5.ICON_SIZE[0], cfg_priv.FUNC_OPT.FUNC5.ICON_SIZE[1]))

        # task special param
        self.image_render = GetImage()
        self.init_openfolder = './'
        self.flag_savevideo = False
        self.info_header = u"Hello from PriVision!\n  "
        self.shown_info(self.info_header)
        self.allframes = []
        self.video_writer = None
        self.savevideo_counting = 0
        self.savevideo_max = cfg_priv.GLOBAL.SAVE_VIDEO_MAX_SECOND

    def setup_config(self):
        print('==> Using config:')
        merge_priv_cfg_from_file(self.cfg_file)
        pprint.pprint(cfg_priv)

    def setup_workers(self):
        if len(cfg_priv.FUNC_OPT.FUNC1.MODULE):
            func1 = importlib.import_module('modules.{}'.format(cfg_priv.FUNC_OPT.FUNC1.MODULE))
            self.worker01 = eval('func1.{}(gpu_id={})'.
                                 format(cfg_priv.FUNC_OPT.FUNC1.CLASS, cfg_priv.FUNC_OPT.FUNC1.GPU_ID))
            ###w
            print('Setup function: {}.{} at GPU: {}'.format(cfg_priv.FUNC_OPT.FUNC1.MODULE,
                                                            cfg_priv.FUNC_OPT.FUNC1.CLASS,
                                                            cfg_priv.FUNC_OPT.FUNC1.GPU_ID))
        if len(cfg_priv.FUNC_OPT.FUNC2.MODULE):
            func2 = importlib.import_module('modules.{}'.format(cfg_priv.FUNC_OPT.FUNC2.MODULE))
            self.worker02 = eval('func2.{}(gpu_id={})'.
                                 format(cfg_priv.FUNC_OPT.FUNC2.CLASS, cfg_priv.FUNC_OPT.FUNC2.GPU_ID))
            print('Setup function: {}.{} at GPU: {}'.format(cfg_priv.FUNC_OPT.FUNC2.MODULE,
                                                            cfg_priv.FUNC_OPT.FUNC2.CLASS,
                                                            cfg_priv.FUNC_OPT.FUNC2.GPU_ID))
        if len(cfg_priv.FUNC_OPT.FUNC3.MODULE):
            func3 = importlib.import_module('modules.{}'.format(cfg_priv.FUNC_OPT.FUNC3.MODULE))
            self.worker03 = eval('func3.{}(gpu_id={})'.
                                 format(cfg_priv.FUNC_OPT.FUNC3.CLASS, cfg_priv.FUNC_OPT.FUNC3.GPU_ID))
            print('Setup function: {}.{} at GPU: {}'.format(cfg_priv.FUNC_OPT.FUNC3.MODULE,
                                                            cfg_priv.FUNC_OPT.FUNC3.CLASS,
                                                            cfg_priv.FUNC_OPT.FUNC3.GPU_ID))
        if len(cfg_priv.FUNC_OPT.FUNC4.MODULE):
            func4 = importlib.import_module('modules.{}'.format(cfg_priv.FUNC_OPT.FUNC4.MODULE))
            self.worker04 = eval('func4.{}(gpu_id={})'.
                                 format(cfg_priv.FUNC_OPT.FUNC4.CLASS, cfg_priv.FUNC_OPT.FUNC4.GPU_ID))
            print('Setup function: {}.{} at GPU: {}'.format(cfg_priv.FUNC_OPT.FUNC4.MODULE,
                                                            cfg_priv.FUNC_OPT.FUNC4.CLASS,
                                                            cfg_priv.FUNC_OPT.FUNC4.GPU_ID))
        if len(cfg_priv.FUNC_OPT.FUNC5.MODULE):
            func5 = importlib.import_module('modules.{}'.format(cfg_priv.FUNC_OPT.FUNC5.MODULE))
            self.worker05 = eval('func5.{}(gpu_id={})'.
                                 format(cfg_priv.FUNC_OPT.FUNC5.CLASS, cfg_priv.FUNC_OPT.FUNC5.GPU_ID))
            print('Setup function: {}.{} at GPU: {}'.format(cfg_priv.FUNC_OPT.FUNC5.MODULE,
                                                            cfg_priv.FUNC_OPT.FUNC5.CLASS,
                                                            cfg_priv.FUNC_OPT.FUNC5.GPU_ID))

    def resizeEvent(self, event):
        self.resized.emit()
        return super(MainWindow, self).resizeEvent(event)

    def update_image(self):
        pass

    def zoomRequest(self, delta):
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def setFitWindow(self, value=True):
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.width() * 0.65 - e
        h1 = self.height() * 0.65 - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.image.width() - 0.0
        h2 = self.canvas.image.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() / 5 * 3 - 2.0
        return w / self.canvas.pixmap.width()

    def paintCanvas(self):
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        # self.canvas.scale = 0.5
        self.canvas.adjustSize()
        self.canvas.update()

    def start_cap(self):
        # self.ui.play_bn.setIcon(QIcon("./files/icons/icon-left/play-color.png"))
        self.ui.pause_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP2))
        self.camera_device.paused = False

    def stop_cap(self):
        self.ui.pause_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP2.replace('bright', 'color')))
        self.camera_device.paused = True

    def save_video(self):
        if not self.flag_savevideo:
            self.flag_savevideo = True
            self.savevideo_counting = 0
            video_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter('{}/{}_save.avi'.format(cfg_priv.GLOBAL.SAVE_VIDEO_PATH, video_name),
                                                fourcc,
                                                cfg_priv.GLOBAL.SAVE_VIDEO_FPS,
                                                tuple(cfg_priv.GLOBAL.SAVE_VIDEO_SIZE))
            self.ui.record_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP3.replace('bright', 'color')))
        else:
            self.video_writer.release()
            self.savevideo_counting = 0
            self.flag_savevideo = False
            self.video_writer = None
            self.ui.record_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP3))
        pass

    def clear(self):
        self.allframes = []

    def shown_info(self, info):
        self.ui.info_display.setPlainText(info)

    def set_all_func_false(self):
        cfg_priv.GLOBAL.F_MODEL1 = False
        self.ui.model1_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC1.ICON))
        cfg_priv.GLOBAL.F_MODEL2 = False
        self.ui.model2_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC2.ICON))
        cfg_priv.GLOBAL.F_MODEL3 = False
        self.ui.model3_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC3.ICON))
        cfg_priv.GLOBAL.F_MODEL4 = False
        self.ui.model4_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC4.ICON))
        cfg_priv.GLOBAL.F_MODEL5 = False
        self.ui.model5_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC5.ICON))

    def apply_model1(self):
        tmp_status = cfg_priv.GLOBAL.F_MODEL1
        self.set_all_func_false()
        if not tmp_status:
            cfg_priv.GLOBAL.F_MODEL1 = True
            self.ui.model1_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC1.ICON.replace('bright', 'color')))

    def apply_model2(self):
        tmp_status = cfg_priv.GLOBAL.F_MODEL2
        self.set_all_func_false()
        if not tmp_status:
            cfg_priv.GLOBAL.F_MODEL2 = True
            self.ui.model2_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC2.ICON.replace('bright', 'color')))

    def apply_model3(self):
        tmp_status = cfg_priv.GLOBAL.F_MODEL3
        self.set_all_func_false()
        if not tmp_status:
            cfg_priv.GLOBAL.F_MODEL3 = True
            self.ui.model3_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC3.ICON.replace('bright', 'color')))

    def apply_model4(self):
        tmp_status = cfg_priv.GLOBAL.F_MODEL4
        self.set_all_func_false()
        if not tmp_status:
            cfg_priv.GLOBAL.F_MODEL4 = True
            self.ui.model4_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC4.ICON.replace('bright', 'color')))

    def apply_model5(self):
        tmp_status = cfg_priv.GLOBAL.F_MODEL5
        self.set_all_func_false()
        if not tmp_status:
            cfg_priv.GLOBAL.F_MODEL5 = True
            self.ui.model5_bn.setIcon(QIcon(cfg_priv.FUNC_OPT.FUNC5.ICON.replace('bright', 'color')))

    def start_camera(self):
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1.replace('bright', 'color')))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5))
        self.camera_device = CameraDevice(cameraId=0)
        self.camera_device.newFrame.connect(self.onNewImage)
        self.clear()

    def start_video1(self):
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2.replace('bright', 'color')))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5))
        self.camera_device.set_video_path(cfg_priv.GLOBAL.VIDEO1)
        self.clear()

    def start_video2(self):
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3.replace('bright', 'color')))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5))
        self.camera_device.set_video_path(cfg_priv.GLOBAL.VIDEO2)
        self.clear()

    def start_pic(self):
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4.replace('bright', 'color')))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5))
        image_name, image_type = QFileDialog.getOpenFileName(self, "select file", self.init_openfolder,
                                                             "IMAGE (*.*)")
        self.init_openfolder = image_name
        self.camera_device.set_image_path(image_name)
        self.clear()

    def load_video(self):
        self.ui.thread1_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT1))
        self.ui.thread2_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT2))
        self.ui.thread3_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT3))
        self.ui.thread4_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT4))
        self.ui.thread5_bn.setIcon(QIcon(cfg_priv.ICONS.TOP.LEFT5.replace('bright', 'color')))
        video_name, video_type = QFileDialog.getOpenFileName(self, "select file", self.init_openfolder,
                                                             "VIDEO (*.*)")
        self.init_openfolder = video_name
        self.camera_device.set_video_path(video_name)
        self.clear()

    @QtCore.pyqtSlot(np.ndarray)
    def onNewImage(self, frame):
        self.adjustScale()

        frame = np.asarray(frame[:, :])

        t = clock()
        if cfg_priv.GLOBAL.F_MODEL1:
            vis = self.worker01(frame)
        elif cfg_priv.GLOBAL.F_MODEL2:
            vis = self.worker02(frame)
        elif cfg_priv.GLOBAL.F_MODEL3:
            vis = self.worker03(frame)
        elif cfg_priv.GLOBAL.F_MODEL4:
            vis = self.worker04(frame)
        elif cfg_priv.GLOBAL.F_MODEL5:
            vis = self.worker05(frame)
        else:
            vis = frame.copy()
        dt = clock() - t

        if self.flag_savevideo and self.savevideo_counting <= self.savevideo_max:
            save_im = cv2.resize(vis, tuple(cfg_priv.GLOBAL.SAVE_VIDEO_SIZE))
            self.video_writer.write(save_im)
            self.savevideo_counting += 1
        elif self.savevideo_counting > self.savevideo_max:
            self.savevideo_counting = 0
            self.flag_savevideo = False
            self.ui.record_bn.setIcon(QIcon(cfg_priv.ICONS.LEFT.TOP3))
        draw_str(vis, 30, 30, 'speed: %.1f fps' % (min(1.0 / dt, 30)))

        cur_info = self.info_header + u'--------------------\n  '
        if self.flag_savevideo:
            cur_info += u'Saving Video~~\n--------------------\n'
        cur_info += u'当前视频频率为: {:.1f}fps\n  '.format(min(1.0 / dt, 30))
        cur_info += u'--------------------\n  '
        self.shown_info(cur_info)

        vis = cv2.resize(vis, tuple(cfg_priv.GLOBAL.IM_SHOW_SIZE))
        image = QImage(vis.tostring(), vis.shape[1], vis.shape[0], QImage.Format_RGB888).rgbSwapped()

        self.canvas.update_image(image)
