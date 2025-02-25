from matplotlib import pyplot as plt
from .utils_generate_samples import generate_shape_image

def plot_samples(num_rows, num_cols, size=None):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for i in range(num_rows * num_cols):
        img, _ = generate_shape_image(min_size=3, s=size)

        i1 = i // num_rows
        i2 = i % num_cols
        if num_cols > 1 and num_rows > 1:
            index = (i1, i2)
        elif num_cols > 1:
            index = (i2)
        else:
            index = (i1)

        axs[index].imshow(img, cmap='Greys')
        axs[index].set_xticks([])
        axs[index].set_yticks([])

    plt.show()
    
    
def plot_rotated_squares(angle_step, size):
    num_rows = 3
    num_cols = 6
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

    for i, angle in enumerate(range(0, angle_step * num_rows * num_cols, angle_step)):
        img, _ = generate_shape_image(shape="square", min_size=3, s=size, angle=angle, position=(16, 16))

        i1 = i // num_cols
        i2 = i % num_cols
        if num_cols > 1 and num_rows > 1:
            index = (i1, i2)
        elif num_cols > 1:
            index = (i2)
        else:
            index = (i1)

        axs[index].imshow(img, cmap='Greys')
        axs[index].set_xticks([])
        axs[index].set_yticks([])
        axs[index].set_title('Angle {}'.format(angle))

    plt.show()