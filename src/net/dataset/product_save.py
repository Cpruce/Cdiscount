from net.common import *
from net.dataset.tool import *
import pandas as pd

import struct
from tqdm import tqdm
import pickle

IDS_MAPPING = {}
ID_MAP_PATH = BASE_DIR + "/ids.pkl"

# helper functions -------------
def score_to_class_names(prob, class_names, threshold = 0.5, nil=''):

    N = len(class_names)
    if not isinstance(threshold,(list, tuple, np.ndarray)) : threshold = [threshold]*N

    s=nil
    for n in range(N):
        if prob[n]>threshold[n]:
            if s==nil:
                s = class_names[n]
            else:
                s = '%s %s'%(s, class_names[n])
    return s


def draw_class_names(image, prob, class_names, threshold=0.5):

    weather = CLASS_NAMES[:4]
    s = score_to_class_names(prob, class_names, threshold, nil=' ')
    for i, ss in enumerate(s.split(' ')):
        if ss in weather:
            color = (255,255,0)
        else:
            color = (0, 255,255)

        draw_shadow_text(image, ' '+ss, (5,30+(i)*15),  0.5, color, 1)



def create_image(image, width=256, height=256):
    h,w,c = image.shape

    if c==3:
        jpg_src=0
        tif_src=None

        M=1
        jpg_dst=0

    if c==4:
        jpg_src=None
        tif_src=0

        M=2
        tif_dst=0

    if c==7:
        jpg_src=0
        tif_src=3

        M=3
        jpg_dst=0
        tif_dst=1


    img = np.zeros((h,w*M,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] *255
        jpg_green = image[:,:,jpg_src+1] *255
        jpg_red   = image[:,:,jpg_src+2] *255

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
    train_dataset = ProductDataset(classes_dict, train_df, transform=[
        lambda x: randomShiftScaleRotate(x, u=0.75, shift_limit=16, scale_limit=0.1, rotate_limit=45),
        lambda x: randomFlip(x),
        lambda x: randomTranspose(x),
        lambda x: img_to_tensor(x),
        ], is_test=False) #, debug=True)
    valid_dataset = ProductDataset(classes_dict, valid_df, transform=[
        lambda x: img_to_tensor(x),
        ], is_test=True) #, debug=True) # height=SIZE, width=SIZE
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
    def __init__(self, classes_dict, df, transform=None, height=180, width=180, is_test=False, debug=False): 
        self.height = height
        self.width = width
        self.classes_dict = classes_dict
        self.df = df
        self.in_channels = 3
        self.transform = transform
        self.is_test = is_test
        self.debug = debug
        self.num = self.df.num.count()

    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        product_meta = self.df.iloc[index]
        product_id, idx= product_meta['_id'], product_meta['ind']
        img = decode(get_image(product_id, idx))

        if self.debug:
            #image = create_image(img)
            im_show('image',img, 1) 
            cv2.waitKey(0)
        
        if self.transform is not None:
            for t in self.transform:
                img = t(img)

        #if self.labels is None:
        if self.is_test == True:
            return img, index
        else:
            category_id = product_meta['category_id']
            classes_dict = self.classes_dict
            label_ind = classes_dict[category_id] 
            return img, label_ind, index


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return self.num #len(self.images)
