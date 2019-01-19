# -*- coding: utf-8 -*-

import colorgram
from PIL import Image

def test_extract_from_file():
    colorgram.extract('data/test.png', 1)

def test_extract_from_image_object():
    image = Image.open('data/test.png')
    colorgram.extract(image, 1)

def test_color_access():
    color = colorgram.Color(255, 151, 210, 0.15)

    assert color.rgb == (255, 151, 210)
    assert (color.rgb.r, color.rgb.g, color.rgb.b) == color.rgb

    assert color.hsl == (230, 255, 203)
    assert (color.hsl.h, color.hsl.s, color.hsl.l) == color.hsl

    assert color.proportion == 0.15

def test_color_prediction():

    model = colorgram.Model(model_type='svm')
    model.train()

    print(model.predict([245, 245, 98]))
    assert model.predict([245, 245, 98]) == 'yellow'
    assert model.predict([98, 245, 127]) == 'green'
    assert model.predict([98, 132, 245]) == 'blue'
    assert model.predict([114, 114, 114]) == 'grey'
