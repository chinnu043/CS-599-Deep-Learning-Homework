zip_url = "https://github.com/tdhock/cs570-spring-2022/raw/master/data/zip.test.gz"
import os
from urllib.request import urlretrieve
if not os.path.exists("zip.test.gz"):
    print("Downloading!")
    urlretrieve(zip_url, "zip.test.gz")

import pandas as pd
df = pd.read_csv("zip.test.gz", sep = " ", header = None)
print(df.shape)
print(df)

import numpy as np
from math import sqrt
image_number = 50
one_image_label = df.iloc[image_number, 0]
intensity_vec = df.iloc[image_number, 1:]
n_pixels = int(sqrt(len(intensity_vec)))
np.flip(np.repeat(np.arange(n_pixels),n_pixels))
np.tile(np.arange(n_pixels),n_pixels)
one_image_df = pd.DataFrame({
    "intensity": intensity_vec,
    "row": np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
    "column": np.tile(np.arange(n_pixels),n_pixels)
})
import plotnine as p9
gg = p9.ggplot()+\
    p9.geom_tile(#draw squares
        p9.aes(
            x = "column",
            y = "row",
            fill = "intensity"
            ),
        data = one_image_df
        )+\
        p9.scale_fill_gradient(
            low = "black",
            high = "white")+\
        p9.ggtitle("Label = %d"%one_image_label)+\
        p9.coord_equal()
gg.save("HW2_image.png")
list_images = []
for image_num in [5,50,500]:
    multi_image_label = df.iloc[image_num, 0]
    intensity_m_vec = df.iloc[image_num, 1:]
    multi_image_df =  pd.DataFrame({
    "observation" : image_num,
    "label" : multi_image_label,
    "intensity": intensity_m_vec,
    "row": np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
    "column": np.tile(np.arange(n_pixels),n_pixels)
    })
    list_images.append(multi_image_df)
    several_images = pd.concat(list_images)
    gg1 =  p9.ggplot()+\
        p9.facet_wrap(["observation", "label"], labeller = "label_both")+\
        p9.geom_tile(#draw squares
            p9.aes(
                x = "column",
                y = "row",
                fill = "intensity"
        ),
            data = several_images
    )+\
    p9.scale_fill_gradient(
            low = "black",
            high = "white")+\
        p9.ggtitle("Multi image Data Visualization")+\
        p9.coord_equal()    
    
gg1.save("HW2_multi_image.png")

n_rows = 9
list_images = []
for image_num in range(n_rows):
    multi_image_label = df.iloc[image_num, 0]
    intensity_m_vec = df.iloc[image_num, 1:]
    multi_image_df =  pd.DataFrame({
    "observation" : image_num,
    "label" : multi_image_label,
    "intensity": intensity_m_vec,
    "row": np.flip(np.repeat(np.arange(n_pixels),n_pixels)),
    "column": np.tile(np.arange(n_pixels),n_pixels)
    })
    list_images.append(multi_image_df)
    several_images = pd.concat(list_images)
    gg2 =  p9.ggplot()+\
        p9.facet_wrap(["observation", "label"], labeller = "label_both")+\
        p9.geom_tile(#draw squares
            p9.aes(
                x = "column",
                y = "row",
                fill = "intensity"
        ),
            data = several_images
    )+\
    p9.scale_fill_gradient(
            low = "black",
            high = "white")+\
        p9.ggtitle("Multi image Data Visualization")+\
        p9.coord_equal()

gg2.save("Multi_image_Visualization.png")


