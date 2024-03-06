# --- For the selected classes, uncomment this section ---
SELECTED_CLASSES = [
    (110, 'Wheat'),
    (120, 'Maize'),
    (140, 'Sorghum'),
    (150, 'Barley'),
    (160, 'Rye'),
    (170, 'Oats'),
    (330, 'Grapes'),
    (435, 'Rapeseed'),
    (438, 'Sunflower'),
    (510, 'Potatoes'),
    (770, 'Peas'),
]

LINEAR_ENCODER = {val[0]: i + 1 for i, val in enumerate(SELECTED_CLASSES)}
LINEAR_ENCODER[0] = 0
