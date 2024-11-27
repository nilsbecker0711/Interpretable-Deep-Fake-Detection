#1. 2x2 grids genereieren (1 fake)
#2. das als bild an bcos übergeben
#3. schau das prediction an
import numpy as np
import os
import random

def create_2x2(fake_imgs, real_imgs, output_imgs, grid_size = (2,2)):
    fake_img