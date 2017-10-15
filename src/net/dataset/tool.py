from net.common import *
# common tool for dataset

# draw -----------------------------------
def im_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def draw_mask(image, mask, color=(255,255,255), α=1,  β=0.25, λ=0., threshold=32 ):
    # image * α + mask * β + λ

    if threshold is None:
        mask = mask/255
    else:
        mask = clean_mask(mask,threshold,1)

    mask  = np.dstack((color[0]*mask,color[1]*mask,color[2]*mask)).astype(np.uint8)
    image[...] = cv2.addWeighted(image, α, mask, β, λ)



## custom data transform  -----------------------------------
def tensor_to_img(img, mean=0, std=1, dtype=np.uint8):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img*std + mean
    img = img.astype(dtype)
    #img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

## transform (input is numpy array, read in by cv2)
def img_to_tensor(img, mean=0, std=255.): #std=1.):
    img = img.astype(np.float32)
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    tensor = torch.from_numpy(img)   ##.float()
    return tensor


## for debug
def dummy_transform1(image):
    print ('\t\tdummy_transform1')
    return image
def dummy_transform2(image):
    print ('\t\tdummy_transform2')
    return image

def augment(x, u=0.5):
    if random.random()<u:
        if random.random()>0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)

        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, \
                rotate_limit=45,u=0.5)
        #x = randomFlip(x, u=0.5)
        x = randomTranspose(x, u=0.5)
        x = randomFlip(x, u=0.5)
        x = randomContrast(x, limit=0.2, u=0.5)
        #x = randomSaturation(x, limit=0.2, u=0.5),

    return x

#http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
#http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,0)  #np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img

def randomHorizontalFlip(img, u=0.5):
    shape=img.shape
    if random.random() < u:
        img = cv2.flip(img,1)  #np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img


def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1,0,2)  #cv2.transpose(img)
    return img


#http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees
def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle=random.randint(1,3)*90
        if angle == 90:
            img = img.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(img,1)
            #return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img,-1)
            #return img[::-1,::-1,:]
        elif angle == 270:
            img = img.transpose(1,0,2)  #cv2.transpose(img)
            img = cv2.flip(img,0)
            #return  img.transpose((1,0, 2))[::-1,:,:]
    return img


def randomRotate(img, u=0.25, limit=90):
    if random.random() < u:
        angle = random.uniform(-limit,limit)  #degree

        height,width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width/2,height/2),angle,1.0)
        img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        #img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    return img



def randomShift(img, u=0.25, limit=4):
    if random.random() < u:
        dx = round(random.uniform(-limit,limit))  #pixel
        dy = round(random.uniform(-limit,limit))  #pixel

        height,width,channel = img.shape
        img1 =cv2.copyMakeBorder(img, limit+1, limit+1, limit+1, limit+1,borderType=cv2.BORDER_REFLECT_101)
        y1 = limit+1+dy
        y2 = y1 + height
        x1 = limit+1+dx
        x2 = x1 + width
        img = img1[y1:y2,x1:x2,:]

    return img


def randomShiftScale(img, u=0.25, limit=4):
    if random.random() < u:
        height,width,channel = img.shape
        assert(width==height)
        size0 = width
        size1 = width+2*limit
        img1  = cv2.copyMakeBorder(img, limit, limit, limit, limit,borderType=cv2.BORDER_REFLECT_101)
        size  = round(random.uniform(size0,size1))


        dx = round(random.uniform(0,size1-size))  #pixel
        dy = round(random.uniform(0,size1-size))


        y1 = dy
        y2 = y1 + size
        x1 = dx
        x2 = x1 + size

        if size ==size0:
            img = img1[y1:y2,x1:x2,:]
        else:
            img = cv2.resize(img1[y1:y2,x1:x2,:],(size0,size0),interpolation=cv2.INTER_LINEAR)

    return img




# def randomShiftScaleRotate(img, u=0.5, shift_limit=4, scale_limit=4, rotate_limit=45):
#     if random.random() < u:
#         height,width,channel = img.shape
#         assert(width==height)
#         size0 = width
#         size1 = width+2*scale_limit
#
#         angle = random.uniform(-rotate_limit,rotate_limit)  #degree
#         size  = round(random.uniform(size0,size1))
#         dx    = round(random.uniform(0,size1-size))  #pixel
#         dy    = round(random.uniform(0,size1-size))
#
#         cc = math.cos(angle/180*math.pi)*(size/size0)
#         ss = math.sin(angle/180*math.pi)*(size/size0)
#         rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])
#
#
#         box0 = np.array([ [0,0], [size0,0],  [size0,size0], [0,size0], ])
#         box1 = box0 - np.array([width/2,height/2])
#         box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])
#
#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv2.getPerspectiveTransform(box0,box1)
#         img = cv2.warpPerspective(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
#
#     return img

def randomShiftScaleRotate(img, u=0.5, shift_limit=4, scale_limit=0.1, rotate_limit=45):
    if random.random() < u:
        height,width,channel = img.shape

        angle = random.uniform(-rotate_limit,rotate_limit)  #degree
        scale = random.uniform(1-scale_limit,1+scale_limit)
        dx    = round(random.uniform(-shift_limit,shift_limit))  #pixel
        dy    = round(random.uniform(-shift_limit,shift_limit))

        cc = math.cos(angle/180*math.pi)*(scale)
        ss = math.sin(angle/180*math.pi)*(scale)
        rotate_matrix = np.array([ [cc,-ss], [ss,cc] ])


        box0 = np.array([ [0,0], [width,0],  [width,height], [0,height], ])
        box1 = box0 - np.array([width/2,height/2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width/2+dx,height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        img = cv2.warpPerspective(img, mat, (width,height),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)  #cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img



def cropCenter(img, height, width):

    h,w,c = img.shape
    dx = (h-height)//2
    dy = (w-width )//2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2,x1:x2,:]

    return img







#return fix data for debug
class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        #print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        #print ('\tcalling Sampler:__len__')
        return self.num_samples








## kaggle evaluations  -----------------------------------

def run_f2_from_csv():
    predict_csv = './share/project/pytorch/results/kaggle-forest/2cls-all-02-tif/submission/results-train-40479.csv'
    true_csv = KAGGLE_DATA_DIR + '/image/train_label.csv'

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    true_df = pd.read_csv(true_csv)
    predict_df = pd.read_csv(predict_csv)
    for c in class_names:
        true_df   [c] = true_df   ['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)
        predict_df[c] = predict_df['tags'].apply(lambda x: 1 if c in x.split(' ') else 0)


    #get true labels
    num = predict_df.shape[0]
    labels = np.zeros((num,num_classes),dtype=np.float32)

    names  = predict_df.iloc[:,0].values
    df1 = true_df.set_index('image_name')
    for n in range(num):
        shortname = names[n]
        labels[n] = df1.loc[shortname].values[1:]

    #get predict
    predictions = predict_df.values[:,2:].astype(np.float32)

    #f2 score
    f2 = sklearn.metrics.fbeta_score(labels, predictions, beta=2, average='samples')
    print('predictions.shape=%s'%str(predictions.shape))
    print('f2=%f'%f2)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_f2_from_csv()



