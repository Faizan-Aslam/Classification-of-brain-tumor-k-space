import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from math import e, pi
from torch.fft import fft2 as nfft2, ifft2 as nifft2
from torch.fft import ifftshift, fftshift

def fft2(x):
    """
    fft function that is simply a wrapper around torch's (very fast) 2-D FFT,
    modified to correct the scaling factor to be consistent with 6.003's
    definitions.
    """
    return nfft2(x) 


def ifft2(x):
    """
    Similarly, a wrapper around torch's IFFT to correct the scaling factor to
    be consistent with 6.003's definitions.
    """
    return nifft2(x) * (x.size)


def png_read(fname, zero_loc="center"):
    """
    Load the PNG image with the given filename into a torch array.  The
    resulting array will have values between 1 (brightest white) and 0 (darkest
    black).

    fname: string
        The filename to load (including .png)

    zero_loc (optional): string
        This parameter controls which point in the original image is considered
        to be (0,0).  By default, it is the center of the given image
        ('center').  The other option is 'topleft'.
    """
    assert zero_loc in {'center', 'topleft'}, "zero_loc must be 'center' or 'topleft', not %r" % zero_loc
    assert os.path.isfile(fname), "Not a file: %r" % fname

    with Image.open(fname) as image:
        im_arr = torch.frombuffer(image.convert('L').tobytes(), dtype=torch.uint8)
        im_arr = im_arr.reshape((image.size[1], image.size[0]))
    if zero_loc == 'center':
        im_arr = ifftshift(im_arr)
    return im_arr.astype(torch.float64)/255


def png_write(array, fname, zero_loc="center", normalize=False, zoom=1):
    """
    Save the given torch array as an image with the given filename.  1
    translates to the brightest white in the output image, and 0 to the darkest
    black.  Values above 1 will be displayed as white, and values below 0 will
    be displayed as black.

    array: torch.ndarray
        An array containing pixel values

    fname: string
        The name of the file to save (including .png)

    zero_loc (optional): string
        This parameter controls where the point (0,0) should be rendered in the
        output image.  By default, (0,0) is at the center of the image
        ('center').  The other option is 'topleft'.

    normalize (optional): boolean
        If True, normalize values so that the largest element maps to 1
        (brightest white) and the smallest element maps to 0 (darkest black).
        Default: False

    zoom (optional): int
        Factor by which to zoom when saving.
    """
    assert zero_loc in {'center', 'topleft'}, "zero_loc must be 'center' or 'topleft', not %r" % zero_loc
    # assert (abs(array.imag) <= 1e-6).all(), "input array must contain real values only"
    if zero_loc == 'center':
        array = fftshift(array)
    if normalize:
        array = array - torch.min(array)
        array = array / torch.max(array)
    h, w = array.shape
    out = Image.new(mode='L', size=(w, h))
    out.putdata(torch.around(255*array).reshape(w*h))
    out.resize((int(w*zoom), int(h*zoom)), Image.NEAREST).save(fname)


def show_image(array, colorscale='linear', zero_loc='center', cmap='gray',
               show=True, title='', xlabel='', ylabel='', vmax=None, vmin=None):
    """
    Display an image using matplotlib.

    array: torch.ndarray
        An array containing the image to be displayed

    colorscale(optional): string
        If set to 'log', the colors are plotted on a log scale.  Defaults to
        'linear'.

    zero_loc (optional): string
        This parameter controls where the point (0,0) should be rendered when
        plotting the image.  By default, (0,0) is at the center of the image
        ('center').  The other option is 'topleft'.

    cmap (optional): string
        The color map to use when displaying the image.  Defaults to 'gray'.
        See https://matplotlib.org/examples/color/colormaps_reference.html
        for options.

    show (optional): bool
        If set to True (the default), the plot will be immediately displayed.
        Otherwise, plt.show() will need to be called later to display the plot.

    title (optional): string
        A title for the plot

    xlabel (optional): string
        A label for the horizontal axis

    ylabel (optional): string
        A label ver the vertical axis
    """
    assert colorscale in {'linear', 'log'}, "coloscale must be 'linear' or 'log', not %r" % colorscale
    assert zero_loc in {'center', 'topleft'}, "zero_loc must be 'center' or 'topleft', not %r" % zero_loc
    assert (abs(array.imag) <= 1e-6).all(), "input array must contain real values only"
    array = array.real
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if colorscale == 'log':
        array[array != 0] = torch.log10(array[array != 0])
        array[array == 0] = torch.min(array-1)
    h, w = array.shape
    if zero_loc == 'center':
        bounds = [-w//2-0.5+(w%2), w//2-0.5+(w%2), h//2-0.5+(w%2), -h//2-0.5+(w%2)]
        array = fftshift(array)
    else:
        bounds = [0, w, h, 0]
    i = plt.imshow(array, cmap=cmap, interpolation='nearest', origin='upper',
                   extent=bounds, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if show:
        plt.show()


def show_dft(array, color_scale='log', color_limits=None,
             color_factor=1, show=True):
    """
    Given an image in the spatial domain, plot the magnitude and phase of its
    DFT coefficients using the show_image function.  Magnitude is plotted on a
    log scale.

    array: torch.ndarray
        An array containing a spatial-domain image (NOT an array of DFT
        coefficients)

    color_scale (optional): string
        Defaults to 'linear'.  If 'log', magnitude colors are on a log scale.

    color_limits (optional): tuple of numbers
        The values of magnitude that should be displayed as black and white,
        respectively.  Values outside this range will be clipped to white or
        black.

    color_factor (optional): number
        A value by which the magnitude should be scaled for display purposes.

    show (optional): bool
        If set to True (the default), the plot will be immediately displayed.
        Otherwise, plt.show() will need to be called later to display the plot.
    """
    X = fft2(array)
    phase = torch.angle(X)
    mag = abs(X)
    phase[mag <= 1e-6] = 0
    if color_limits is None:
        if color_scale == 'log':
            color_limits = (None, None)
        else:
            color_limits = (0, 1)
    show_image(mag*color_factor, colorscale=color_scale, show=False,
               title=r'$\left|X[k_x, k_y]\right|$',
               xlabel='$k_x$', ylabel='$k_y$',
               vmin=color_limits[0], vmax=color_limits[1])
    show_image(phase, cmap='gray',
               title=r'$\angle X[k_x, k_y]$',
               xlabel='$k_x$', ylabel='$k_y$',
               vmax=pi, vmin=-pi, show=show)
