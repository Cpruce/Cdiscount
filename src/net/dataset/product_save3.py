from net.common import *
from net.dataset.tool import *
import pandas as pd

import struct
from tqdm import tqdm
import pickle
import hashlib
from PIL import Image
import io as baseIO
from tqdm import tqdm 

IDS_MAPPING = {}
ID_MAP_PATH = BASE_DIR + "/ids.pkl"

def create_image(image, width=180, height=180):
    h,w,c = image.shape

    if c==3:
        jpg_src=0
        jpg_dst=0

    img = np.zeros((h,w,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] #*255
        jpg_green = image[:,:,jpg_src+1] #*255
        jpg_red   = image[:,:,jpg_src+2] #*255

        img[:,jpg_dst*w:(jpg_dst+1)*w] = np.dstack((jpg_blue,jpg_green,jpg_red)).astype(np.uint8)

    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img

def img_to_tensor(img, mean=0, std=255.):
    img = img.astype(np.float32)
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    tensor = torch.from_numpy(img).float()
    return tensor

#data = bson.decode_file_iter(open(DATA_DIR+'train.bson', 'rb'))

def produce_split_dataset():
    global IDS_MAPPING
    try:
        IDS_MAPPING = pickle.load(open(ID_MAP_PATH, "rb"))
    except:
        create_offset_map()    
    
    train_df = pd.read_csv(BASE_DIR+'input/split/train_images_split.csv')
    valid_df = pd.read_csv(BASE_DIR+'input/split/valid_images_split.csv')
    classes_df = train_df['category_id'].append(valid_df['category_id'])
    classes_df = classes_df.drop_duplicates()
    classes_dict = classes_df.to_dict()
    classes_dict = { classes_dict[k] : i for i, k in enumerate(classes_dict) }
    # num_per_split 2033333 
    train_dataset = ProductDataset("train", classes_dict, train_df, 300000, transform=[
        lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45),
        lambda x: randomFlip(x),
        lambda x: randomTranspose(x),
        lambda x: img_to_tensor(x),
        ], is_test=False, height=SIZE, width=SIZE) #, debug=True)
    valid_dataset = ProductDataset("valid", classes_dict, valid_df, 171296, transform=[
        lambda x: img_to_tensor(x),
        ], is_test=False, height=SIZE, width=SIZE) #, debug=True) 
    return (train_dataset, valid_dataset, classes_dict)

# thanks: https://www.kaggle.com/vfdev5/random-item-access/notebook 
def create_offset_map():
    num_dicts = 7069896 # according to data page

    length_size = 4 # number of bytes decoding item length

    with open(DATA_DIR + 'train.bson', 'rb') as f, tqdm(total=num_dicts) as bar:
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
def get_image(product_id, idx):
    assert product_id in IDS_MAPPING
    with open(os.path.join(DATA_DIR, 'train.bson'), 'rb') as file_handler:
        offset, length = IDS_MAPPING[product_id]
        file_handler.seek(offset)
        item_data = file_handler.read(length)
    return bson.BSON.decode(item_data)['imgs'][idx]['picture'] 

def decode(data):
    arr = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## custom data loader -----------------------------------
class ProductDataset(Dataset):
    def __init__(self, name, classes_dict, df, num_per_split,
            transform=None, height=180, width=180, is_test=False, debug=False): 
        self.height = height
        self.width = width
        self.classes_dict = classes_dict
        self.df = df
        self.in_channels = 3
        self.transform = transform
        self.is_test = is_test
        self.debug = debug
        self.num = self.df.num.count()
        self.name = name
        labels = {}
        self.num_per_split = num_per_split
        self.num_split = self.num // num_per_split

        #print('Loading {}'.format(name))
        img_file = AUX_DIR+name+'_'+str(SIZE)#+'.npy' 
        lbl_file = DATA_DIR+name+'_label.pkl' 
 
        try:
            labels = pickle.load(open(lbl_file, 'rb'))
            self.labels = labels
            print('Loaded labels')
        except Exception as e:
            print('Could not load label file!')
            sys.exit(1)
       
        """
        try:
            print('Checking cache files. Num split = {}'.format(self.num_split))
            # check img cache files
            for i in range(self.num_split):
                cur_file = img_file+'_'+str(i)+'.npy' 
                print('Loading {}'.format(cur_file))
                images = np.load(cur_file)
            print('All cache segments present')
        except Exception as e:
            #read images
            print(e)
            print(num_per_split, self.num_split, num_per_split*height*width*3)
            cur_split = 0
            images = np.zeros((num_per_split,height,width, 3),dtype=np.uint8)
            for i in tqdm(range(self.num)):
                
                if i % num_per_split == 0:
                    images = np.zeros((num_per_split,height,width, 3),dtype=np.uint8)
                
                product_meta = self.df.iloc[i]
                product_id, idx= product_meta['_id'], product_meta['ind']
                img = decode(get_image(product_id, idx))

                #print(img.shape)
                h,w = img.shape[0:2]
                if height!=h or width!=w:
                    img = cv2.resize(img,(height,width))

                #print(img.shape)
                images[i%num_per_split,:,:] = img
                
                if self.debug:
                    im_show('image',img, 1) 
                    cv2.waitKey(0)

                if (num_per_split-1 == (i % num_per_split)) or i == self.num-1:
                    cur_img_file = img_file + '_' + str(cur_split) + '.npy'
                    print('Saving {}'.format(cur_img_file))
                    np.save(cur_img_file, images)
                    print('Saved {}'.format(cur_img_file))
                    cur_split+=1

        """
        self.img_file_base = img_file
        self.images = None
        self.cur_split = -1

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)
        cur_split = index // self.num_per_split 
        if cur_split != self.cur_split:
            cur_file = self.img_file_base+'_'+str(cur_split)+'.npy' 
            print('Loading new segment {}'.format(cur_file))
            self.images = np.load(cur_file)
            self.cur_split = cur_split

        img = self.images[index%self.num_per_split]

        if self.debug:
            im_show('image',img, 1) 
            cv2.waitKey(0)
        
        if self.transform is not None:
            for t in self.transform:
                img = t(img)

        if self.is_test == True:
            return img, index
        else:
            classes_dict = self.classes_dict
            labels = self.labels
            cat_id = labels[index]
            label_idx = classes_dict[cat_id]
            #print(img.shape, cat_id, label_idx)
            return img, label_idx, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return self.num #len(self.images)
