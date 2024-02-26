# --- For the selected classes, uncomment this section ---
SELECTED_CLASSES = [
    110,   # 'Wheat'
    120,   # 'Maize'
    140,   # 'Sorghum'
    150,   # 'Barley'
    160,   # 'Rye'
    170,   # 'Oats'
    330,   # 'Grapes'
    435,   # 'Rapeseed'
    438,   # 'Sunflower'
    510,   # 'Potatoes'
    770,   # 'Peas'
]

LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
LINEAR_ENCODER[0] = 0


CLASS_WEIGHTS = {
    0: 0,
    110: 0.24919847862509978,
    120: 1.1980338175954461,
    140: 9.361927845094968,
    150: 0.3617731585557118,
    160: 37.10448039555864,
    170: 1.70706483652164,
    330: 1.6318220911149515,
    435: 1.1009523253620337,
    438: 1.6449674601314823,
    510: 3.4364602852011052,
    770: 4.021317821460993
}

