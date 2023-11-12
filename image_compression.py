from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


image = imread('china-original.png')
plt.imshow(image)
plt.show()


def pca_reduction(image, axis, n_components=None):
    # preparing channels
    if axis:  # check if reduction is by length(normal state)
        r, g, b = image[:, :, 0]/255, image[:, :, 1]/255, image[:, :, 2]/255
    else:  # check if reduction is by width
        r, g, b = image[:, :, 0].T, image[:, :, 1].T, image[:, :, 2].T
    # auto selects the best n_components
    if not n_components:
        df = pd.DataFrame({'r': [], 'g': [], 'b': []})
        for i in range(30, 50):  # dimension range
            df.loc[i] = [PCA(i).fit(r).explained_variance_ratio_.sum(), PCA(i).fit(
                g).explained_variance_ratio_.sum(), PCA(i).fit(b).explained_variance_ratio_.sum()]
        df['cv'] = minmax_scale([row.std()/row.mean()
                                for _, row in df.iterrows()])
        df['n_components'] = minmax_scale(df.index)
        df['score'] = [(1 - row[4]) * ((1 - row[3]) ** 100)
                       for _, row in df.iterrows()]
        n_components = df.sort_values('score', ascending=False).iloc[0].name
    pca_r, pca_g, pca_b = PCA(n_components).fit(r), PCA(
        n_components).fit(g), PCA(n_components).fit(b)
    image = np.empty(shape=image.shape)
    # reconstructing the rgb form(check if transposed form is needed)
    if axis:
        image[:, :, 0], image[:, :, 1], image[:, :, 2] = pca_r.inverse_transform(pca_r.transform(
            r)), pca_g.inverse_transform(pca_g.transform(g)), pca_b.inverse_transform(pca_b.transform(b))
    else:
        image[:, :, 0], image[:, :, 1], image[:, :, 2] = pca_r.inverse_transform(pca_r.transform(
            r)).T, pca_g.inverse_transform(pca_g.transform(g)).T, pca_b.inverse_transform(pca_b.transform(b)).T
    return image, n_components


def image_compression(image, new_dim=(None, None)):  # new_dim = (width,length)
    image, new_length = pca_reduction(
        image, 1, new_dim[1])  # reduce image by length
    image, new_width = pca_reduction(
        image * 255, 0, new_dim[0])  # reduce image by width
    return image, (new_width, new_length)


# if dimensions are automatically chosen
# image, new_dims = image_compression(image)
image, new_dims = image_compression(image, new_dim=(
    100, 100))  # if dimensions are determined by user

plt.imshow(image)
plt.show()
Image.fromarray((image * 255).astype(np.uint8)
                ).save('china-compressed.png', 'png')
