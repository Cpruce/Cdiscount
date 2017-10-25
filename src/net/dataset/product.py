from net.common import *
from net.dataset.tool import *
import pandas as pd

import struct
from tqdm import tqdm
import pickle

IDS_MAPPING = {}
ID_MAP_PATH = BASE_DIR + "/ids.pkl"

# helper functions -------------
def img_to_tensor(img, mean=0, std=255.):
    img = img.astype(np.float32)
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    tensor = torch.from_numpy(img).float()
    return tensor

#data = bson.decode_file_iter(open(DATA_DIR+'train.bson', 'rb'))

def produce_split_dataset():
    global IDS_MAPPING
    global classes_dict
    try:
        IDS_MAPPING = pickle.load(open(ID_MAP_PATH, "rb"))
    except:
        create_offset_map('train.bson')    
    
    train_df = pd.read_csv(BASE_DIR+'input/split/train_images_split.csv')
    valid_df = pd.read_csv(BASE_DIR+'input/split/valid_images_split.csv')
    classes_df = train_df['category_id'].append(valid_df['category_id'])
    classes_df = classes_df.drop_duplicates()
    classes_dict = classes_df.to_dict()
    classes_dict = { classes_dict[k] : i for i, k in enumerate(classes_dict) }
    train_dataset = ProductDataset(IDS_MAPPING, classes_dict, train_df, transform=[
        lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45),
        lambda x: randomFlip(x),
        lambda x: randomTranspose(x),
        lambda x: img_to_tensor(x),
        ], is_test=False, height=SIZE, width=SIZE) #, debug=True)
    valid_dataset = ProductDataset(IDS_MAPPING, classes_dict, valid_df, transform=[
        lambda x: img_to_tensor(x),
        ], is_test=False, height=SIZE, width=SIZE) #, debug=True) 
    return (train_dataset, valid_dataset, classes_dict)

# thanks: https://www.kaggle.com/vfdev5/random-item-access/notebook 
def create_offset_map(filename):
    num_dicts = 7069896 # according to data page

    length_size = 4 # number of bytes decoding item length

    with open(DATA_DIR + filename, 'rb') as f, tqdm(total=num_dicts) as bar:
        item_data = []
        offset = 0
        while True:        
            bar.update()
            f.seek(offset)
            
            item_length_bytes = f.read(length_size)     
            if len(item_length_bytes) == 0:
                break                
            # Decode item length:
            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length, "%i vs %i" % (len(item_data), length)
            
            # Check if we can decode
            item = bson.BSON.decode(item_data)
            
            IDS_MAPPING[item['_id']] = (offset, length)        
            offset += length      
    pickle.dump(IDS_MAPPING, open(ID_MAP_PATH, 'wb'))
    #return open(DATA_DIR + 'train.bson', 'rb') # return the goods

#file_handler = open(os.path.join(DATA_DIR, 'train.bson'), 'rb')
def get_product(product_id, bson_file, MAPPINGS):
    assert product_id in MAPPINGS
    with open(os.path.join(DATA_DIR, bson_file), 'rb') as file_handler:
        offset, length = MAPPINGS[product_id]
        file_handler.seek(offset)
        item_data = file_handler.read(length)
    return bson.BSON.decode(item_data)['imgs']

def decode(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Method to compose a single image from 1 - 4 images
def decode_images(item_imgs):
    nx = 2 if len(item_imgs) > 1 else 1
    ny = 2 if len(item_imgs) > 2 else 1
    composed_img = np.zeros((ny * 180, nx * 180, 3), dtype=np.uint8)
    for i, img_dict in enumerate(item_imgs):
        img = decode(img_dict['picture'])
        h, w, _ = img.shape        
        xstart = (i % nx) * 180
        xend = xstart + w
        ystart = (i // nx) * 180
        yend = ystart + h
        composed_img[ystart:yend, xstart:xend] = img
    return composed_img

## custom data loader -----------------------------------
class ProductDataset(Dataset):
    def __init__(self, mappings, classes_dict, df, transform=None, height=180, width=180, is_test=False, debug=False): 
        self.height = height
        self.width = width
        self.mappings = mappings
        self.classes_dict = classes_dict
        self.df = df
        self.in_channels = 3
        self.transform = transform
        self.is_test = is_test
        self.debug = debug
        self.num = self.df.count()[0]
        self.idx_id_map = {}

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        if self.is_test:
            bson_file = 'test.bson'
        else:
            bson_file = 'train.bson'

        product_meta = self.df.iloc[index]
        product_id, num_imgs = product_meta['_id'], product_meta['num_imgs']
        
        imgs = get_product(product_id, bson_file, self.mappings)
        img = decode_images(imgs)

        shp = img.shape
        if shp[0] != self.height or shp[1] != self.width:
            img = cv2.resize(img, (self.height, self.width))

        if self.debug:
            im_show('image',img, 1) 
            cv2.waitKey(0)
        
        if self.transform is not None:
            for t in self.transform:
                img = t(img)

        if self.is_test == True:
            return img, index
        else:
            category_id = product_meta['category_id']
            classes_dict = self.classes_dict
            label_ind = classes_dict[category_id] 
            return img, label_ind, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return self.num 
