# This file contains the ICCS app class that inherits from ICVT AppAncestor class

# Part of python
import os
import sys

# Extra packages
import PyQt5
from PyQt5.QtCore import pyqtSignal, QThread

# Get the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Add to the PATH environment variable
os.environ["PATH"] = main_script_dir + os.pathsep + os.environ["PATH"]
import mpv
pyqt = os.path.dirname(PyQt5.__file__)
os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt, "Qt5/plugins")


class MPVThread(QThread):
    shift_click_signal = pyqtSignal(int)

    def __init__(self, video_filepath):
        #print("MPVThread __init__ called")
        self.filepath = video_filepath
        self.last_emit_time = 0
        super(MPVThread, self).__init__()

    def run(self):
        self.player = mpv.MPV(player_operation_mode='pseudo-gui',
                         input_default_bindings=True,
                         osc=True)
        self.player.geometry = f'50%+-0+-0'
        self.player.register_key_binding('shift+MBTN_LEFT', self.emit_shift_click)
        self.player.play(self.filepath)
        self.player.wait_for_playback()
        self.player.terminate()

    def emit_shift_click(self, *args):

        import time
        emmit_time = time.time()
        if emmit_time - self.last_emit_time < 1:  # 1 second throttle
            return
        self.last_emit_time = emmit_time
        current_time = self.player.time_pos
        self.shift_click_signal.emit(int(current_time))