import numpy as np

BANDS = {
    'B02': 10, 'B03': 10, 'B04': 10, 'B08': 10,
    'B05': 20, 'B07': 20, 'B06': 20, 'B8A': 20, 'B11': 20, 'B12': 20,
    'B01': 60, 'B09': 60, 'B10': 60
}

# Extract patches based on this band
REFERENCE_BAND = 'B02'

IMG_SIZE = 366

MEDIANS_DTYPE = np.float32
LABEL_DTYPE = np.int16
