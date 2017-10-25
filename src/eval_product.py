from net.common import *
from net.dataset.tool import *
from net.utility.tool import *
from net.rates import *

from train_product import acc_measure
from net.dataset.product import *

#from net.model.pyramidnet import PyNet_12  as Net
#--------------------------------------------
#from net.model.resnet import resnet18 as Net
#from net.model.resnet import resnet34 as Net
from net.model.resnet import resnet50 as Net
#from net.model.resnet import resnet152 as Net

#from net.model.densenet import densenet121 as Net
#from net.model.densenet import densenet161 as Net
#from net.model.densenet import densenet169 as Net
#from net.model.densenet import densenet201 as Net

#from net.model.inceptionv2 import Inception2 as Net
#from net.model.inceptionv3 import Inception3 as Net
#from net.model.inceptionv4 import Inception4 as Net


#from net.model.vggnet import vgg16 as Net
#from net.model.vggnet import vgg16_bn as Net
#from net.model.vggnet import vgg19 as Net


## global setting ################
SIZE = 180
BASE_DIR = '/home/cory/Kaggle/Cdiscount/'
DATA_DIR = BASE_DIR+'input/'
OUT_DIR = '/home/cory/Kaggle/Cdiscount/output/resnet152-redemption/'
#BASE_DIR+'output/inception-2/' #'output/resnet50-pretrain-5/' 
BATCH_SIZE = 832 #96 #4 #18
SET_SIZE = 1768183 

## main functions ############################################################
def predict(net, test_loader):

    test_dataset = test_loader.dataset
    classes_dict = test_dataset.classes_dict
    classes_dict = {classes_dict[i]:i for i in classes_dict}
    predictions_df = pd.DataFrame(index=range(test_dataset.num), columns=["_id", "category_id"])
    pickle.dump(classes_dict, open('./classes_dict.pkl', 'wb')) 
    idx_id_map = list(TEST_IDS_MAPPING.keys())
    test_num  = 0
    predictions_arr = np.empty(0, dtype=np.uint8)
    for i, (images, indices) in tqdm(enumerate(test_loader, 0)):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))
        _, predictions = torch.max(probs, 1)

        predictions_arr = np.concatenate((predictions_arr, predictions.data.cpu().numpy()))
        test_num+=len(images)

    assert(test_dataset.num==test_num)
    return predictions_arr #df

def hck_predict(net, test_loader):
    test_num  = len(test_loader.dataset)
    scores = np.zeros((test_num,CDISCOUNT_NUM_CLASSES), np.uint8)
    n = 0
    start = timer()
    for i, (images, indices) in enumerate(test_loader, 0):
      print('\r%5d:  %10d/%d (%0.0f %%)  %0.2f min'%(i, n, test_num, 100*n/test_num,
                       (timer() - start) / 60), end='',flush=True)

      # forward
      images = Variable(images,volatile=True).cuda(async=True)
      logits = net(images)
      #logits = net(images.half())  #
      probs = F.softmax(logits)
      probs = probs.data.cpu().numpy()*255

      batch_size = len(indices)
      scores[n:n+batch_size]=probs
      n += batch_size

    print('\n')
    assert(n == len(test_loader.sampler) and n == test_num)
    return scores
    

TEST_IDS_MAPPING = {}
TEST_ID_MAP_PATH = BASE_DIR + "/test_ids.pkl"
# thanks: https://www.kaggle.com/vfdev5/random-item-access/notebook
def create_test_offset_map(filename):
    num_dicts = SET_SIZE # according to data page

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

            TEST_IDS_MAPPING[item['_id']] = (offset, length)
            offset += length
    pickle.dump(TEST_IDS_MAPPING, open(TEST_ID_MAP_PATH, 'wb'))


##-----------------------------------------
def do_submission():
    global TEST_IDS_MAPPING
    out_dir = OUT_DIR 
    model_file = out_dir +'/snap/final.torch'  #final

    log = Logger()
    log.open(out_dir+'/log.submissions.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    ## dataset ----------------------------
    log.write('** dataset setting **\n')
    batch_size    = BATCH_SIZE #36 #20#48  #128

    try:
        TEST_IDS_MAPPING = pickle.load(open(TEST_ID_MAP_PATH, "rb"))
    except:
        create_test_offset_map('data/test.bson')

    train_df = pd.read_csv(BASE_DIR+'input/split/train_images_split.csv')
    valid_df = pd.read_csv(BASE_DIR+'input/split/valid_images_split.csv')
    classes_df = train_df['category_id'].append(valid_df['category_id'])
    classes_df = classes_df.drop_duplicates()
    classes_dict = classes_df.to_dict()
    classes_dict = { classes_dict[k] : i for i, k in enumerate(classes_dict) }
    test_df = pd.read_csv(BASE_DIR+'input/split/test_images.csv')
    test_dataset = ProductDataset(TEST_IDS_MAPPING, classes_dict, test_df,
                                    transform=[
                                         #transforms.Lambda(lambda x: cropCenter(x, height=224, width=224)),
                                         lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE,width=SIZE,
                                    is_test=True)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),  #None,
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 10,
                        pin_memory  = True)

    height, width , in_channels   = test_dataset.height, test_dataset.width, test_dataset.in_channels 
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttest_dataset.num  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size        = %d\n'%batch_size)
    log.write('\n')

    #from net.model.cdiscount.inception_v3 import Inception3
    #net = Inception3(in_shape=(3,180,180),num_classes=5270)
    #model_file = '/home/cory/Kaggle/Cdiscount/HCK/release/trained_models/LB=0.69565_inc3_00075000_model.pth'
    #net.load_state_dict(torch.load(model_file))

    ## net ----------------------------------------
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()

    predictions_arr = predict( net, test_loader )
    test_dir = out_dir +'/submissions/'
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)

    assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
    np.save(test_dir+'/results', predictions_arr)
    print('Saved '+test_dir+'/results.npy')

def do_submission2():
    out_dir = OUT_DIR
    test_dir = out_dir +'/submissions/'
    sample_df = pd.read_csv(BASE_DIR+'input/data/sample_submission.csv')
    predictions_arr = np.load(test_dir+'/results.npy')
    print('loaded '+test_dir+'/results.npy')
    train_df = pd.read_csv(BASE_DIR+'input/split/train_images_split.csv')
    valid_df = pd.read_csv(BASE_DIR+'input/split/valid_images_split.csv')
    classes_df = train_df['category_id'].append(valid_df['category_id'])
    classes_df = classes_df.drop_duplicates()
    classes_dict = classes_df.to_dict()
    classes_dict = { i : classes_dict[k] for i, k in enumerate(classes_dict) }
    #classes_dict = pickle.load('./label_to_category_id') 
    predictions_df = pd.DataFrame(predictions_arr, columns=["category_id"])
    predictions_df["category_id"] = predictions_df["category_id"].apply(lambda x: classes_dict[x])
    predictions_df["_id"] = sample_df["_id"]
    predictions_df = predictions_df.set_index(["_id"])
    predictions_df.to_csv('./submit.csv')
    print(predictions_df.head())

if __name__ == '__main__':
    print('Running eval main')
    do_submission()
    do_submission2() 
    print('Done!')
