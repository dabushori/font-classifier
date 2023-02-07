from torch.utils.data import Dataset
import torch
import h5py
import numpy as np
import cv2 as cv


class SynthTextCharactersDataset(Dataset):
    """
    How does it gonna work?
    The database contains a lot of images, each image contains multiple text segments which we have the bounding box of.
    Each text segment contains multiple characters which we have the bounding box of.
    Eventually each character is an item so the get item will return a specific character with its label (if it is a train dataset).
    The __getitem__ function will perform a projective transform to transform the character into an image of the given shape.
    """

    def __init__(
        self,
        filename: str,
        start_idx: int = None,
        end_idx: int = None,
        shape: tuple[int] = (100, 200),
        train: bool = True,
        transform=None,
        target_transform=None,
        on_get_item_transform=None,
        on_get_item_target_transform=None,
    ):
        """
        Create the dataset. The items that will be saved are the characters, each item is a character from an image.
        If this is a train dataset, each item will be saved as ((img_name, charBB), font).
        If this is a test dataset, each item will be saved as ((img_name, charBB)).
        The start_idx and end_idx are used to specify the indexes of the files from the database to be used.
        The shape will be the shape of each character's image after the projective transform.

        The `transform` and `target_transform` are applied to the input vector and its label on the initialization.
        The `on_get_item_transform` `on_get_item_target_transform` are applied to the input vector and its label on a call to get item.
        These 2 are used to save RAM usage and avoid loading all the dataset to the RAM.
        """

        self.on_get_item_transform = on_get_item_transform
        self.on_get_item_target_transform = on_get_item_target_transform

        self.db = h5py.File(filename, "r")
        self.db_data = self.db["data"]

        self.train = train
        self.shape = shape

        im_names = list(self.db_data.keys())[start_idx:end_idx]

        self.items = []
        if train:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    font = curr_img_data.attrs["font"][idx]
                    font = target_transform(font) if target_transform else font

                    x = (im, charBB)
                    x = transform(x) if transform else x

                    self.items.append((x, font))

        else:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    x = (im, charBB)
                    x = transform(x) if transform else x

                    self.items.append(((im, charBB)))

    def __len__(self):
        """
        Get the length of the dataset
        """

        return len(self.items)

    def __getitem__(self, idx):
        """
        Get an item from the dataset
        """

        x, y = None, None
        if self.train:
            x, y = self.items[idx]
        else:
            x = self.items[idx]

        # Get the image of the chatacter (after the projective transform)
        x = self.get_char_data(x[0], x[1], self.shape).float()
        # The images are given as (h,w,c), and pytorch requires (c,h,w)
        x = x.permute((2, 0, 1))
        # Perform the specified "on get item transform"
        x = self.on_get_item_transform(x) if self.on_get_item_transform else x

        # Perform the specified "on get item target transform"
        y = (
            self.train and self.on_get_item_target_transform(y)
            if self.on_get_item_target_transform
            else y
        )
        return (x, y) if self.train else x

    def get_image_data(self, img_name: str):
        """
        Get an image given it's name
        """

        return self.db_data[img_name][:]

    def get_char_data(self, img_name: str, charBB, shape: tuple[int]):
        """
        Get an image of a character given the full image's name and it's bounding box using a prjective transform
        """

        img = self.db_data[img_name][:]

        src_points = np.array(
            list(charBB[:, i] for i in range(charBB.shape[1])), dtype=np.float32
        )
        dst_points = np.array(
            [[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]],
            dtype=np.float32,
        )

        proj_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        out_img = cv.warpPerspective(img, proj_matrix, shape)

        return torch.from_numpy(out_img)
