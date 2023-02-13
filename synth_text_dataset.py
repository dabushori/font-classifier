import typing

import cv2 as cv
import h5py
import numpy as np
from torch.utils.data import Dataset


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
        transform: typing.Union[typing.Callable[[np.array], np.array], None] = None,
        target_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        on_get_item_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        on_get_item_target_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        full_image_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
    ) -> None:
        """
        Create the dataset. The items that will be saved are the characters, each item is a character from an image.
        If this is a train dataset, each item will be saved as ((img_name, charBB), font).
        If this is a test dataset, each item will be saved as ((img_name, charBB)).
        The start_idx and end_idx are used to specify the indexes of the files from the database to be used.
        The shape will be the shape of each character's image after the projective transform.

        The `transform` and `target_transform` are applied to the input vector and its label on the initialization.
        The `on_get_item_transform` and `on_get_item_target_transform` are applied to the input vector and its label on a call to get item.
        The `full_image_transform` is applied to the whole image, before the character is taken from it.
        These transforms are used to save RAM usage and avoid loading all the dataset to the RAM.
        """

        self.on_get_item_transform = on_get_item_transform
        self.on_get_item_target_transform = on_get_item_target_transform
        self.full_image_transform = full_image_transform

        self.db = h5py.File(filename, "r")
        self.db_data = self.db["data"]

        self.train = train
        self.shape = shape

        im_names = list(self.db_data.keys())[start_idx:end_idx]

        self.x_items = []
        self.y_items = []
        if train:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    # Get the label and apply the `target_transform` on it (if available)
                    font = curr_img_data.attrs["font"][idx]
                    font = target_transform(font) if target_transform else font

                    # Get the input vector and apply the `transform` on it (if available)
                    x = (im, charBB)
                    x = transform(x) if transform else x

                    self.x_items.append(x)
                    self.y_items.append(font)

        else:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    # Get the input vector and apply the `transform` on it (if available)
                    x = (im, charBB)
                    x = transform(x) if transform else x

                    self.x_items.append(x)
            self.x_items = np.array(self.x_items, np.float32)

    def __len__(self) -> int:
        """
        Get the length of the dataset
        """

        return len(self.x_items)

    def __getitem__(
        self, idx
    ) -> typing.Union[tuple[np.array, typing.Any], tuple[np.array]]:
        """
        Get an item from the dataset
        """

        x, y = None, None
        if self.train:
            x, y = self.x_items[idx], self.y_items[idx]
        else:
            x = self.x_items[idx]

        # Get the chatacter out of the image (using a projective transform)
        x = self.get_char_data(x[0], x[1], self.shape)

        # The images are given as (h,w,c), and pytorch requires (c,h,w)
        if len(x.shape) == 3:
            x = x.swapaxes(1, 2).swapaxes(0, 1)

        # Apply the specified `on_get_item_transform` (if available)
        x = self.on_get_item_transform(x) if self.on_get_item_transform else x

        # Apply the specified `on_get_item_target_transform` (if available)
        y = (
            self.train and self.on_get_item_target_transform(y)
            if self.on_get_item_target_transform
            else y
        )

        return (x, y) if self.train else x

    def get_item_raw(self, idx) -> tuple[tuple[np.array, np.array], typing.Any]:
        """
        Get an raw item from the dataset
        """

        return self.x_items[idx], self.y_items[idx]

    def get_image_data(self, img_name: str) -> np.array:
        """
        Get an image given it's name
        """

        # Apply the `full_image_transform` if available
        img = np.array(self.db_data[img_name][:], np.float32)
        if self.full_image_transform:
            return self.full_image_transform(img)
        return img

    def get_char_data(
        self, img: typing.Union[str, np.array], charBB, shape: tuple[int]
    ) -> np.array:
        """
        Get an image of a character given the full image's name and it's bounding box using a prjective transform
        """

        img = self.get_image_data(img) if isinstance(img, str) else img

        src_points = np.array(
            list(charBB[:, i] for i in range(charBB.shape[1])), dtype=np.float32
        )
        dst_points = np.array(
            [[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]],
            dtype=np.float32,
        )

        proj_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        out_img = cv.warpPerspective(img, proj_matrix, shape)

        return out_img


class SynthTextCharactersDatasetRAM(SynthTextCharactersDataset):
    """
    This is a dataset that is similar to SynthTextCharactersDataset, the only difference is that it loads all the characters to the memory
    on the init function and don't load them only when needed. This is used to save the time spent on reading the same image multiple times
    when reading different characters of the same image in different times.
    The previous dataset took 7m 19.2s to load all the images to the memory, and this one took 36.01s.
    All the given transforms will keep the same logic, but will be applied on the init function.
    """

    def __init__(
        self,
        filename: str,
        start_idx: int = None,
        end_idx: int = None,
        shape: tuple[int] = (100, 200),
        train: bool = True,
        permutation: np.array = None,
        transform: typing.Union[typing.Callable[[np.array], np.array], None] = None,
        target_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        on_get_item_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        on_get_item_target_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
        full_image_transform: typing.Union[
            typing.Callable[[np.array], np.array], None
        ] = None,
    ) -> None:
        """
        Create the dataset. The items that will be saved are the characters, each item is a character from an image.
        If this is a train dataset, each item will be saved as ((img_name, charBB), font).
        If this is a test dataset, each item will be saved as ((img_name, charBB)).
        The start_idx and end_idx are used to specify the indexes of the characters to be used, after randomly shuffled.
        The shape will be the shape of each character's image after the projective transform.

        The `transform` and `target_transform` are applied to the input vector and its label on the initialization.
        The `on_get_item_transform` and `on_get_item_target_transform` are applied to the input vector and its label on a call to get item.
        The `full_image_transform` is applied to the whole image, before the character is taken from it.
        These transforms are used to save RAM usage and avoid loading all the dataset to the RAM.
        """

        self.on_get_item_transform = on_get_item_transform
        self.on_get_item_target_transform = on_get_item_target_transform
        self.full_image_transform = full_image_transform

        self.db = h5py.File(filename, "r")
        self.db_data = self.db["data"]

        self.train = train
        self.shape = shape

        im_names = list(self.db_data.keys())

        self.x_items = []
        self.y_items = []

        if train:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                img_data = self.get_image_data(im)

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    # Get the label and apply the `target_transform` on it (if available)
                    y = curr_img_data.attrs["font"][idx]
                    y = target_transform(y) if target_transform else y

                    # Get the input vector and apply the `transform` on it (if available)
                    x = (im, charBB)
                    x = transform(x) if transform else x

                    # Get the chatacter out of the image (using a projective transform)
                    x = self.get_char_data(img_data, x[1], self.shape)

                    # The images are given as (h,w,c), and pytorch requires (c,h,w)
                    if len(x.shape) == 3:
                        x = x.swapaxes(1, 2).swapaxes(0, 1)

                    # Apply the specified `on_get_item_transform` (if available)
                    x = (
                        self.on_get_item_transform(x)
                        if self.on_get_item_transform
                        else x
                    )

                    # Apply the specified `on_get_item_target_transform` (if available)
                    y = (
                        self.train and self.on_get_item_target_transform(y)
                        if self.on_get_item_target_transform
                        else y
                    )

                    self.x_items.append(x)
                    self.y_items.append(y)
            self.x_items = np.array(self.x_items, np.float32)
            self.y_items = np.array(self.y_items)

            if permutation is not None:
                self.x_items = self.x_items[permutation][start_idx:end_idx]
                self.y_items = self.y_items[permutation][start_idx:end_idx]
        else:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs["charBB"].shape[2]

                img_data = self.get_image_data(im)

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle,
                    # and the third axis is the index of the character.
                    charBB = curr_img_data.attrs["charBB"][:, :, idx]

                    # Get the input vector and apply the `transform` on it (if available)
                    x = (im, charBB)
                    x = transform(x) if transform else x

                    # Get the chatacter out of the image (using a projective transform)
                    x = self.get_char_data(img_data, x[1], self.shape)

                    # The images are given as (h,w,c), and pytorch requires (c,h,w)
                    if len(x.shape) == 3:
                        x = x.swapaxes(1, 2).swapaxes(0, 1)

                    # Apply the specified `on_get_item_transform` (if available)
                    x = (
                        self.on_get_item_transform(x)
                        if self.on_get_item_transform
                        else x
                    )

                    self.x_items.append(x)
            self.x_items = np.array(self.x_items, float)
            self.x_items = self.x_items[start_idx:end_idx]

    def __getitem__(
        self, idx
    ) -> typing.Union[tuple[np.array, typing.Any], tuple[np.array]]:
        """
        Get an item from the dataset
        """

        return self.x_items[idx], self.y_items[idx] if self.train else self.x_items[idx]
