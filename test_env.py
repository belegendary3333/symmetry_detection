print('Testing Python environment...')
import sys
print('Python version:', sys.version)

try:
    import cv2
    print('OpenCV version:', cv2.__version__)
    print('cv2 imported successfully!')
except ImportError as e:
    print('Error importing cv2:', e)
    
try:
    import numpy
    print('NumPy version:', numpy.__version__)
except ImportError as e:
    print('Error importing numpy:', e)

try:
    import matplotlib
    print('Matplotlib version:', matplotlib.__version__)
except ImportError as e:
    print('Error importing matplotlib:', e)