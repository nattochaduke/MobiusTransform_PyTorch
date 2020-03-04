import numpy as np
from scipy.ndimage import geometric_transform

class MobiusTransform():

    def __init__(self, p=0.6, image_size=(224, 224), interpolation_order=3):
        """

        :param p (float): probability that the Mobius transform operation will be performed.
        :param image_size (int, int) or int : input image size without (height, width), without channel dimension.
                If the input were a single value v, this class interprets it as the side of square.
        :param interpolation_order (int): the interpolation order of mobius transformation. Smaller is faster.
        """
        self.p = p
        self.interpolation_order = interpolation_order
        if type(image_size) is int:
            height, width = image_size, image_size
        else:
            height, width = image_size

        zws = [
            [  # 1. Clockwise Twist
                [1 + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.6 * width + 0.5 * height * 1j],

                [0.5 * width + (height - 1) * 1j,
                 0.5 * width + 0.3 * np.sin(0.4 * np.pi) * height + (
                             0.5 * height + 0.3 * np.cos(0.4 * np.pi) * height) * 1j,
                 0.5 * width + 0.1 * np.cos(0.1 * np.pi) * height + (
                             0.5 * height - 0.1 * np.sin(0.1 * np.pi) * width) * 1j]
            ],
            [  # 2. Clockwise half-twist
                [1 + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.6 * width + 0.5 * height * 1j],

                [0.5 * width + (height - 1) * 1j,
                 0.5 * width + 0.4 * height + 0.5 * height * 1j,
                 0.5 * width + (0.5 * height - 0.1 * width) * 1j]
            ],

            [  # 3. Spread
                [0.3 * width + 0.5 * height * 1j,
                 0.5 * width + 0.7 * height * 1j,
                 0.7 * width + 0.5 * height * 1j],

                [0.2 * width + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.8 * width + 0.5 * height * 1j]
            ],

            [  # 4. Spread twist
                [0.3 * width + 0.3 * height * 1j,
                 0.6 * width + 0.8 * height * 1j,
                 0.7 * width + 0.3 * height * 1j],

                [0.2 * width + 0.3 * height * 1j,
                 0.6 * width + 0.9 * height * 1j,
                 0.8 * width + 0.2 * height * 1j, ]
            ],

            [  # 5. Counter clockwise twist
                [1 + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.6 * width + 0.5 * height * 1j],

                [0.5 * width + (height - 1) * 1j,
                 0.5 * width + 0.4 * height + 0.5 * height * 1j,
                 0.5 * width + (0.5 * height - 0.1 * width) * 1j]
            ],

            [  # 6. Counter clockwise half-twist
                [1 + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.6 * width + 0.5 * height * 1j, ],

                [0.5 * width + (height - 1) * 1j,
                 (0.5 * width + 0.3 * np.sin(0.4 * np.pi) * height) + (
                             0.5 * height + 0.3 * np.cos(0.4 * np.pi) * height) * 1j,
                 (0.5 * width + 0.1 * np.cos(0.1 * np.pi) * width) + (
                             0.5 * height - 0.1 * np.sin(0.1 * np.pi) * width) * 1j]
            ],

            [  # 7. Inverse
                [1 + 0.5 * height * 1j,
                 0.5 * width + 0.9 * height * 1j,
                 (width - 1) + 0.5 * height * 1j],

                [(width - 1) + 0.5 * height * 1j,
                 0.5 * width + 0.1 * height * 1j,
                 1 + 0.5 * height * 1j]
            ],

            [  # 8. Inverse spread
                [0.1 * width + 0.5 * height * 1j,
                 0.5 * width + 0.8 * height * 1j,
                 0.9 * width + 0.5 * height * 1j, ],

                [(width - 1) + 0.5 * height * 1j,
                 0.5 * width + 0.1 * height * 1j,
                 1 + 0.5 * height * 1j]
            ]
        ]

        self.abcds = [MobiusTransform.calc_coords(z, w) for z, w in zws]

    def shift_func(self, coords, abcd):
        # Adopted from https://glowingpython.blogspot.com/2011/08/applying-moebius-transformation-to.html
        """ Define the moebius transformation, though backwards """
        # turn the first two coordinates into an imaginary number
        a, b, c, d = abcd
        z = coords[0] + 1j * coords[1]
        w = (d * z - b) / (-c * z + a)  # the inverse mobius transform
        return np.real(w), np.imag(w), coords[2]  # take the color along for the ride

    def __call__(self, sample):
        if np.random.uniform() > self.p:
            return sample
        abcd = self.abcds[np.random.randint(0, 8)]
        sample = geometric_transform(sample, self.shift_func, cval=255, output_shape=sample.shape,
                                     order=self.interpolation_order,
                                     extra_keywords={'abcd': abcd})
        return sample

    @staticmethod
    def calc_coords(z, w):
        a = np.linalg.det([
            [z[0] * w[0], w[0], 1],
            [z[1] * w[1], w[1], 1],
            [z[2] * w[2], w[2], 1]
        ])

        b = np.linalg.det([
            [z[0] * w[0], z[0], w[0]],
            [z[1] * w[1], z[1], w[1]],
            [z[2] * w[2], z[2], w[2]],
        ])

        c = np.linalg.det([
            [z[0], w[0], 1],
            [z[1], w[1], 1],
            [z[2], w[2], 1],
        ])

        d = np.linalg.det([
            [z[0] * w[0], z[0], 1],
            [z[1] * w[1], z[1], 1],
            [z[2] * w[2], z[2], 1],
        ])
        return (a, b, c, d)
