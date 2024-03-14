# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS, JunkiEDM'

if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

import time
import os
import numpy as np
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from inference import predict_with_model
import torch


root = dict()


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, options):
        super().__init__()
        self.options = options

    def run(self):
        global root
        # Here we pass the update_progress (uncalled!)
        self.options['update_percent_func'] = self.update_progress
        for f in self.options['input_audio']:
            predict_with_model({**self.options, 'input_audio': [f]})
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        self.finished.emit()

    def update_progress(self, percent):
        self.progress.emit(percent)


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        global root

        Dialog.setObjectName('Settings')
        Dialog.resize(370, 520)

        self.checkbox_cpu = QCheckBox('Use CPU instead of GPU', Dialog)
        self.checkbox_cpu.setText('Use CPU instead of GPU')
        self.checkbox_cpu.resize(320, 40)
        self.checkbox_cpu.move(30, 10)
        if root['cpu']:
            self.checkbox_cpu.setChecked(True)

        self.checkbox_single_onnx = QCheckBox('Only use single ONNX model for vocals (low VRAM)', Dialog)
        self.checkbox_single_onnx.setText('Only use single ONNX model for vocals (low VRAM)')
        self.checkbox_single_onnx.resize(320, 40)
        self.checkbox_single_onnx.move(30, 40)
        if root['single_onnx']:
            self.checkbox_single_onnx.setChecked(True)

        self.checkbox_large_gpu = QCheckBox('Store models on GPU (Requires 11GB+ VRAM)', Dialog)
        self.checkbox_large_gpu.setText('Store models on GPU (Requires 11GB+ VRAM)')
        self.checkbox_large_gpu.resize(320, 40)
        self.checkbox_large_gpu.move(30, 70)
        if root['large_gpu']:
            self.checkbox_large_gpu.setChecked(True)

        self.BigShifts_label = QLabel(Dialog)
        self.BigShifts_label.setText('MDX BigShifts')
        self.BigShifts_label.resize(300, 40)
        self.BigShifts_label.move(30, 100)
        self.BigShifts = QSpinBox(Dialog)
        self.BigShifts.setMinimum(1)
        self.BigShifts.setMaximum(41)
        self.BigShifts.setFixedWidth(100)
        self.BigShifts.move(240, 110)
        self.BigShifts.setValue(root['BigShifts'])

        self.overlap_InstVoc_label = QLabel(Dialog)
        self.overlap_InstVoc_label.setText('MDXv3 overlap')
        self.overlap_InstVoc_label.resize(300, 40)
        self.overlap_InstVoc_label.move(30, 130)
        self.overlap_InstVoc = QSpinBox(Dialog)
        self.overlap_InstVoc.setMinimum(1)
        self.overlap_InstVoc.setMaximum(40)
        self.overlap_InstVoc.setFixedWidth(100)
        self.overlap_InstVoc.move(240, 140)
        self.overlap_InstVoc.setValue(root['overlap_InstVoc'])

        self.overlap_VitLarge_label = QLabel(Dialog)
        self.overlap_VitLarge_label.setText('VitLarge overlap')
        self.overlap_VitLarge_label.resize(300, 40)
        self.overlap_VitLarge_label.move(30, 160)
        self.overlap_VitLarge = QSpinBox(Dialog)
        self.overlap_VitLarge.setMinimum(1)
        self.overlap_VitLarge.setMaximum(40)
        self.overlap_VitLarge.setFixedWidth(100)
        self.overlap_VitLarge.move(240, 170)
        self.overlap_VitLarge.setValue(root['overlap_VitLarge'])

        self.weight_InstVoc_label = QLabel(Dialog)
        self.weight_InstVoc_label.setText('MDXv3 weight')
        self.weight_InstVoc_label.resize(300, 40)
        self.weight_InstVoc_label.move(30, 190)
        self.weight_InstVoc = QDoubleSpinBox(Dialog)
        self.weight_InstVoc.setMinimum(0)
        self.weight_InstVoc.setMaximum(10)
        self.weight_InstVoc.setSingleStep(0.1)
        self.weight_InstVoc.setFixedWidth(100)
        self.weight_InstVoc.move(240, 200)
        self.weight_InstVoc.setValue(root['weight_InstVoc'])

        self.weight_VitLarge_label = QLabel(Dialog)
        self.weight_VitLarge_label.setText('VitLarge weight')
        self.weight_VitLarge_label.resize(300, 40)
        self.weight_VitLarge_label.move(30, 220)
        self.weight_VitLarge = QDoubleSpinBox(Dialog)
        self.weight_VitLarge.setMinimum(0)
        self.weight_VitLarge.setMaximum(10)
        self.weight_VitLarge.setSingleStep(0.1)
        self.weight_VitLarge.setFixedWidth(100)
        self.weight_VitLarge.move(240, 230)
        self.weight_VitLarge.setValue(root['weight_VitLarge'])

        self.checkbox_use_VOCFT = QCheckBox('Use VOC-FT?', Dialog)
        self.checkbox_use_VOCFT.setText('Use VOC-FT?')
        self.checkbox_use_VOCFT.resize(320, 40)
        self.checkbox_use_VOCFT.move(30, 250)
        if root['use_VOCFT']:
            self.checkbox_use_VOCFT.setChecked(True)

        self.overlap_VOCFT_label = QLabel(Dialog)
        self.overlap_VOCFT_label.setText('VOC-FT overlap')
        self.overlap_VOCFT_label.resize(300, 40)
        self.overlap_VOCFT_label.move(30, 280)
        self.overlap_VOCFT = QDoubleSpinBox(Dialog)
        self.overlap_VOCFT.setMinimum(0)
        self.overlap_VOCFT.setMaximum(0.95)
        self.overlap_VOCFT.setSingleStep(0.05)
        self.overlap_VOCFT.setFixedWidth(100)
        self.overlap_VOCFT.move(240, 290)
        self.overlap_VOCFT.setValue(root['overlap_VOCFT'])

        self.weight_VOCFT_label = QLabel(Dialog)
        self.weight_VOCFT_label.setText('VOC-FT weight')
        self.weight_VOCFT_label.resize(300, 40)
        self.weight_VOCFT_label.move(30, 310)
        self.weight_VOCFT = QDoubleSpinBox(Dialog)
        self.weight_VOCFT.setMinimum(0)
        self.weight_VOCFT.setMaximum(10)
        self.weight_VOCFT.setSingleStep(0.1)
        self.weight_VOCFT.setFixedWidth(100)
        self.weight_VOCFT.move(240, 320)
        self.weight_VOCFT.setValue(root['weight_VOCFT'])

        self.checkbox_vocals_only = QCheckBox('Generate only vocals/instrumental', Dialog)
        self.checkbox_vocals_only.setText('Generate only vocals/instrumental')
        self.checkbox_vocals_only.resize(320, 40)
        self.checkbox_vocals_only.move(30, 340)
        if root['vocals_only']:
            self.checkbox_vocals_only.setChecked(True)

        self.overlap_demucs_label = QLabel(Dialog)
        self.overlap_demucs_label.setText('Demucs overlap')
        self.overlap_demucs_label.resize(300, 40)
        self.overlap_demucs_label.move(30, 370)
        self.overlap_demucs = QDoubleSpinBox(Dialog)
        self.overlap_demucs.setFixedWidth(100)
        self.overlap_demucs.move(240, 380)
        self.overlap_demucs.setValue(root['overlap_demucs'])

        self.output_format_label = QLabel(Dialog)
        self.output_format_label.setText('Output format')
        self.output_format_label.resize(300, 40)
        self.output_format_label.move(30, 400)
        self.output_format = QComboBox(Dialog)
        self.output_format.addItems(['PCM_16', 'FLOAT'])
        self.output_format.setFixedWidth(100)
        self.output_format.move(240, 410)
        self.output_format.setCurrentIndex(['PCM_16', 'FLOAT'].index(root['output_format']))

        self.chunk_size_label = QLabel(Dialog)
        self.chunk_size_label.setText('Chunk size')
        self.chunk_size_label.resize(300, 40)
        self.chunk_size_label.move(30, 430)
        self.chunk_size = QSpinBox(Dialog)
        self.chunk_size.setMinimum(100000)
        self.chunk_size.setMaximum(10000000)
        self.chunk_size.setFixedWidth(100)
        self.chunk_size.move(240, 440)
        self.chunk_size.setValue(root['chunk_size'])

        self.pushButton_save = QPushButton(Dialog)
        self.pushButton_save.setObjectName('save')
        self.pushButton_save.resize(145, 35)
        self.pushButton_save.move(30, 470)
        self.pushButton_save.clicked.connect(self.return_save)

        self.pushButton_cancel = QPushButton(Dialog)
        self.pushButton_cancel.setObjectName('cancel')
        self.pushButton_cancel.resize(145, 35)
        self.pushButton_cancel.move(195, 470)
        self.pushButton_cancel.clicked.connect(self.return_cancel)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        self.Dialog = Dialog

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Settings", "Settings"))
        self.pushButton_cancel.setText(_translate("Settings", "Cancel"))
        self.pushButton_save.setText(_translate("Settings", "Save settings"))

    def return_save(self):
        global root
        # print("save")
        root['cpu'] = self.checkbox_cpu.isChecked()
        root['single_onnx'] = self.checkbox_single_onnx.isChecked()
        root['large_gpu'] = self.checkbox_large_gpu.isChecked()
        root['vocals_only'] = self.checkbox_vocals_only.isChecked()
        root['use_VOCFT'] = self.checkbox_use_VOCFT.isChecked()
        root['output_format'] = self.output_format.currentText()
        root['chunk_size'] = self.chunk_size.value()
        root['overlap_demucs'] = self.overlap_demucs.value()
        root['overlap_VOCFT'] = self.overlap_VOCFT.value()
        root['overlap_VitLarge'] = self.overlap_VitLarge.value()
        root['overlap_InstVoc'] = self.overlap_InstVoc.value()
        root['weight_InstVoc'] = self.weight_InstVoc.value()
        root['weight_VOCFT'] = self.weight_VOCFT.value()
        root['weight_VitLarge'] = self.weight_VitLarge.value()
        root['BigShifts'] = self.BigShifts.value()

        self.Dialog.close()

    def return_cancel(self):
        global root
        # print("cancel")
        self.Dialog.close()


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(560, 360)
        self.move(300, 300)
        self.setWindowTitle('MVSEP music separation model')
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        global root
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)

    def execute_long_task(self):
        global root

        if len(root['input_files']) == 0 and 1:
            QMessageBox.about(root['w'], "Error", "No input files specified!")
            return

        root['progress_bar'].show()
        root['button_start'].setDisabled(True)
        root['button_finish'].setDisabled(False)
        root['start_proc'] = True

        options = {
            'input_audio': root['input_files'],
            'output_folder': root['output_folder'],
            'cpu': root['cpu'],
            'single_onnx': root['single_onnx'],
            'large_gpu': root['large_gpu'],
            'chunk_size': root['chunk_size'],
            'overlap_demucs': root['overlap_demucs'],
            'overlap_VitLarge': root['overlap_VitLarge'],
            'overlap_VOCFT': root['overlap_VOCFT'],
            'overlap_InstVoc': root['overlap_InstVoc'],
            'weight_InstVoc': root['weight_InstVoc'],
            'weight_VitLarge': root['weight_VitLarge'],
            'weight_VOCFT': root['weight_VOCFT'],
            'BigShifts': root['BigShifts'],
            'output_format': root['output_format'],
            'use_VOCFT': root['use_VOCFT'],
            'vocals_only': root['vocals_only'],
        }

        self.update_progress(0)
        self.thread = QThread()
        self.worker = Worker(options)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)

        self.thread.start()

    def stop_separation(self):
        global root
        self.thread.terminate()
        root['button_start'].setDisabled(False)
        root['button_finish'].setDisabled(True)
        root['start_proc'] = False
        root['progress_bar'].hide()

    def update_progress(self, progress):
        global root
        root['progress_bar'].setValue(progress)

    def open_settings(self):
        global root
        dialog = QDialog()
        dialog.ui = Ui_Dialog()
        dialog.ui.setupUi(dialog)
        dialog.exec_()


def dialog_select_input_files():
    global root
    files, _ = QFileDialog.getOpenFileNames(
        None,
        "QFileDialog.getOpenFileNames()",
        "",
        "All Files (*);;Audio Files (*.wav, *.mp3, *.flac)",
    )
    if files:
        txt = ''
        root['input_files'] = []
        for f in files:
            root['input_files'].append(f)
            txt += f + '\n'
        root['input_files_list_text_area'].insertPlainText(txt)
        root['progress_bar'].setValue(0)
    return files


def dialog_select_output_folder():
    global root
    foldername = QFileDialog.getExistingDirectory(
        None,
        "Select Directory"
    )
    root['output_folder'] = foldername + '/'
    root['output_folder_line_edit'].setText(root['output_folder'])
    return foldername


def create_dialog():
    global root
    app = QApplication(sys.argv)

    w = MyWidget()

    root['input_files'] = []
    root['output_folder'] = os.path.dirname(os.path.abspath(__file__)) + '/results/'
    root['cpu'] = False
    root['large_gpu'] = False
    root['single_onnx'] = False
    root['chunk_size'] = 1000000
    root['overlap_demucs'] = 0.6
    root['overlap_VOCFT'] = 0.1
    root['overlap_VitLarge'] = 1
    root['overlap_InstVoc'] = 1
    root['BigShifts'] = 7
    root['weight_InstVoc'] = 8
    root['weight_VOCFT'] = 2
    root['weight_VitLarge'] = 5
    root['use_VOCFT'] = False
    root['output_format'] = 'PCM_16'
    root['vocals_only'] = False

    t = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)
    if t > 11.5:
        print('You have enough GPU memory ({:.2f} GB), so we set fast GPU mode. You can change in settings!'.format(t))
        root['large_gpu'] = True
        root['single_onnx'] = False
    elif t < 8:
        root['large_gpu'] = False
        root['single_onnx'] = False
        # root['single_onnx'] = True  # crashes
        root['chunk_size'] = 1000000

    button_select_input_files = QPushButton(w)
    button_select_input_files.setText("Input audio files")
    button_select_input_files.clicked.connect(dialog_select_input_files)
    button_select_input_files.setFixedHeight(35)
    button_select_input_files.setFixedWidth(150)
    button_select_input_files.move(30, 20)

    input_files_list_text_area = QTextEdit(w)
    input_files_list_text_area.setReadOnly(True)
    input_files_list_text_area.setLineWrapMode(QTextEdit.NoWrap)
    font = input_files_list_text_area.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    input_files_list_text_area.move(30, 60)
    input_files_list_text_area.resize(500, 100)

    button_select_output_folder = QPushButton(w)
    button_select_output_folder.setText("Output folder")
    button_select_output_folder.setFixedHeight(35)
    button_select_output_folder.setFixedWidth(150)
    button_select_output_folder.clicked.connect(dialog_select_output_folder)
    button_select_output_folder.move(30, 180)

    output_folder_line_edit = QLineEdit(w)
    output_folder_line_edit.setReadOnly(True)
    font = output_folder_line_edit.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    output_folder_line_edit.move(30, 220)
    output_folder_line_edit.setFixedWidth(500)
    output_folder_line_edit.setText(root['output_folder'])

    progress_bar = QProgressBar(w)
    # progress_bar.move(30, 310)
    progress_bar.setValue(0)
    progress_bar.setGeometry(30, 310, 500, 35)
    progress_bar.setAlignment(QtCore.Qt.AlignCenter)
    progress_bar.hide()
    root['progress_bar'] = progress_bar

    button_start = QPushButton('Start separation', w)
    button_start.clicked.connect(w.execute_long_task)
    button_start.setFixedHeight(35)
    button_start.setFixedWidth(150)
    button_start.move(30, 270)

    button_finish = QPushButton('Stop separation', w)
    button_finish.clicked.connect(w.stop_separation)
    button_finish.setFixedHeight(35)
    button_finish.setFixedWidth(150)
    button_finish.move(200, 270)
    button_finish.setDisabled(True)

    button_settings = QPushButton('⚙', w)
    button_settings.clicked.connect(w.open_settings)
    button_settings.setFixedHeight(35)
    button_settings.setFixedWidth(35)
    button_settings.move(495, 270)
    button_settings.setDisabled(False)

    mvsep_link = QLabel(w)
    mvsep_link.setOpenExternalLinks(True)
    font = mvsep_link.font()
    font.setFamily("Courier")
    font.setPointSize(10)
    mvsep_link.move(415, 30)
    mvsep_link.setText('Powered by <a href="https://mvsep.com">MVSep.com</a>')

    root['w'] = w
    root['input_files_list_text_area'] = input_files_list_text_area
    root['output_folder_line_edit'] = output_folder_line_edit
    root['button_start'] = button_start
    root['button_finish'] = button_finish
    root['button_settings'] = button_settings

    # w.showMaximized()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    print('Version: 2.3.1')
    create_dialog()
