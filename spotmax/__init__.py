import sys
import os

spotmax_path = os.path.dirname(os.path.abspath(__file__))
settings_path = os.path.join(spotmax_path, 'settings')

is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))
