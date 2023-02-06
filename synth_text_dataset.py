from torch.utils.data import Dataset
import h5py



class SynthTextCharactersDataset(Dataset):
    '''
    How does it gonna work?
    The database contains a lot of images, each image contains multiple text segments which we have the bounding box of. Each text segment contains multiple characters which we have the bounding box of. 
    Eventually each character is an item so the get item will return a specific character with its label (if it is a train dataset).
    '''
    
    def __init__(self, filename: str, train: bool = True):
        '''
        Create the dataset. The items that will be saved are the characters, each item is a character from an image.
        If this is a train dataset, each item will be saved as ((img_name, charBB), font).
        If this is a test dataset, each item will be saved as ((img_name, charBB)).
        '''
        
        self.db = h5py.File(filename, 'r')
        self.db_data = self.db['data']
        
        im_names = list(self.db_data.keys())

        self.items = []
        if train:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs['charBB'].shape[2]

                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle, and the third axis is the index of the character.
                    charBB = curr_img_data.attrs['charBB'][:, :, idx]
                    font = curr_img_data.attrs['font'][idx]
                    self.items.append(((im, charBB), font))
                    
        else:
            for im in im_names:
                curr_img_data = self.db_data[im]
                num_chars = curr_img_data.attrs['charBB'].shape[2]
                
                for idx in range(num_chars):
                    # The charBB shape is (2, 4, num_chars). The first axis is the x,y coordinates, the second axis is the index of the corner of the rectangle, and the third axis is the index of the character.
                    charBB = curr_img_data.attrs['charBB'][:, :, idx]
                    self.items.append(((im, charBB)))
                
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def get_image_data(self, img_name: str):
        return self.db_data[img_name][:]

