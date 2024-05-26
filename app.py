import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow_hub as hub

def load_model_with_custom_objects(model_path):
    custom_objects = {"KerasLayer": hub.KerasLayer}
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

model = load_model_with_custom_objects("DogBreedModel.h5")

class_names = {
    0: 'affenpinscher',
    1: 'afghan_hound',
    2: 'african_hunting_dog',
    3: 'airedale',
    4: 'american_staffordshire_terrier',
    5: 'appenzeller',
    6: 'australian_terrier',
    7: 'basenji',
    8: 'basset',
    9: 'beagle',
    10: 'bedlington_terrier',
    11: 'bernese_mountain_dog',
    12: 'black-and-tan_coonhound',
    13: 'blenheim_spaniel',
    14: 'bloodhound',
    15: 'bluetick',
    16: 'border_collie',
    17: 'border_terrier',
    18: 'borzoi',
    19: 'boston_bull',
    20: 'bouvier_des_flandres',
    21: 'boxer',
    22: 'brabancon_griffon',
    23: 'briard',
    24: 'brittany_spaniel',
    25: 'bull_mastiff',
    26: 'cairn',
    27: 'cardigan',
    28: 'chesapeake_bay_retriever',
    29: 'chihuahua',
    30: 'chow',
    31: 'clumber',
    32: 'cocker_spaniel',
    33: 'collie',
    34: 'curly-coated_retriever',
    35: 'dandie_dinmont',
    36: 'dhole',
    37: 'dingo',
    38: 'doberman',
    39: 'english_foxhound',
    40: 'english_setter',
    41: 'english_springer',
    42: 'entlebucher',
    43: 'eskimo_dog',
    44: 'flat-coated_retriever',
    45: 'french_bulldog',
    46: 'german_shepherd',
    47: 'german_short-haired_pointer',
    48: 'giant_schnauzer',
    49: 'golden_retriever',
    50: 'gordon_setter',
    51: 'great_dane',
    52: 'great_pyrenees',
    53: 'greater_swiss_mountain_dog',
    54: 'groenendael',
    55: 'ibizan_hound',
    56: 'irish_setter',
    57: 'irish_terrier',
    58: 'irish_water_spaniel',
    59: 'irish_wolfhound',
    60: 'italian_greyhound',
    61: 'japanese_spaniel',
    62: 'keeshond',
    63: 'kelpie',
    64: 'kerry_blue_terrier',
    65: 'komondor',
    66: 'kuvasz',
    67: 'labrador_retriever',
    68: 'lakeland_terrier',
    69: 'leonberg',
    70: 'lhasa',
    71: 'malamute',
    72: 'malinois',
    73: 'maltese_dog',
    74: 'mexican_hairless',
    75: 'miniature_pinscher',
    76: 'miniature_poodle',
    77: 'miniature_schnauzer',
    78: 'newfoundland',
    79: 'norfolk_terrier',
    80: 'norwegian_elkhound',
    81: 'norwich_terrier',
    82: 'old_english_sheepdog',
    83: 'otterhound',
    84: 'papillon',
    85: 'pekinese',
    86: 'pembroke',
    87: 'pomeranian',
    88: 'pug',
    89: 'redbone',
    90: 'rhodesian_ridgeback',
    91: 'rottweiler',
    92: 'saint_bernard',
    93: 'saluki',
    94: 'samoyed',
    95: 'schipperke',
    96: 'scotch_terrier',
    97: 'scottish_deerhound',
    98: 'sealyham_terrier',
    99: 'shetland_sheepdog',
    100: 'shih-tzu',
    101: 'siberian_husky',
    102: 'silky_terrier',
    103: 'soft-coated_wheaten_terrier',
    104: 'staffordshire_bullterrier',
    105: 'standard_poodle',
    106: 'standard_schnauzer',
    107: 'sussex_spaniel',
    108: 'tibetan_mastiff',
    109: 'tibetan_terrier',
    110: 'toy_poodle',
    111: 'toy_terrier',
    112: 'vizsla',
    113: 'walker_hound',
    114: 'weimaraner',
    115: 'welsh_springer_spaniel',
    116: 'west_highland_white_terrier',
    117: 'whippet',
    118: 'wire-haired_fox_terrier',
    119: 'yorkshire_terrier'
}

st.title("Dog Breed Classifier")
st.sidebar.title("Upload Image")
uploaded_image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)

    predicted_class_name = class_names.get(predicted_class_index)
    st.write(f"Predicted Breed: {predicted_class_name}")
