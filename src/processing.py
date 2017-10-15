## do alignment for tif and jpg

from net.common import *
from net.dataset.tool import *
from net.utility.tool import *
from net.dataset.kgforest import *

DATA_DIR = '/mnt/media/PlanetData/'


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

        M=4
        jpg_dst=0
        tif_dst=1


    img = np.zeros((h,w*M,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] *255
        jpg_green = image[:,:,jpg_src+1] *255
        jpg_red   = image[:,:,jpg_src+2] *255

        img[:,jpg_dst*w:(jpg_dst+1)*w] = np.dstack((jpg_blue,jpg_green,jpg_red)).astype(np.uint8)

    if tif_src is not None:
        tif_blue  = np.clip(image[:,:,tif_src  ] *4095*255/65536.0*6 -25-30,a_min=0,a_max=255)
        tif_green = np.clip(image[:,:,tif_src+1] *4095*255/65536.0*6    -30,a_min=0,a_max=255)
        tif_red   = np.clip(image[:,:,tif_src+2] *4095*255/65536.0*6 +25-30,a_min=0,a_max=255)
        tif_nir   = np.clip(image[:,:,tif_src+3] *4095*255/65536.0*4,a_min=0,a_max=255)

        img[:,tif_dst*w:(tif_dst+1)*w] = np.dstack((tif_blue,tif_green,tif_red)).astype(np.uint8)
        img[:,(tif_dst+1)*w:(tif_dst+2)*w ] = np.dstack((tif_nir,tif_nir,tif_nir)).astype(np.uint8)

    if M==4:
        im1 = img[:,jpg_dst*w:(jpg_dst+1)*w]
        im2 = img[:,tif_dst*w:(tif_dst+1)*w]
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2[:,:,0:3],cv2.COLOR_BGR2GRAY)
        zz = np.zeros((h,w),np.uint8)
        img[:,3*w: ] = np.dstack((im1_gray,zz,im2_gray)).astype(np.uint8)


    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img

def norm_channel(data):
    h,b  = np.histogram(data, bins=100)

    dmin=b[ 10]
    dmax=b[-10]
    data = (data-dmin)/(dmax-dmin)
    data = np.clip(data,a_min=0,a_max=1)


    step = 16
    data = ((data*255).astype(np.int32)//step)*step /255
    data = np.clip(data,a_min=0,a_max=1)


    return data





def align_tif_to_jpg(image_jpg, image_tif):
    # http://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
    # “Parametric Image Alignment using Enhanced Correlation Coefficient Maximization” - Evangelidis, G.D.
    #     and Psarakis E.Z, PAMI 2008
    #

    # Convert images to grayscale
    im1= image_jpg
    im2= image_tif
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2[:,:,0:3],cv2.COLOR_BGR2GRAY)

    sz = im1.shape
    warp_mode = cv2.MOTION_TRANSLATION # Define the motion model
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Define termination criteria
    number_of_iterations = 100
    termination_eps = 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    try:
        correlation, warp_matrix = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    except cv2.error:
        return im2, 0, 0, -1

    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))
    tx = warp_matrix[0,2]
    ty = warp_matrix[1,2]

    return im2_aligned, tx, ty, correlation

def get_rect(tx,ty,width=256,height=256):

    tx = int(round(tx))
    ty = int(round(ty))
    x1 = max(tx,0)
    y1 = max(ty,0)
    x2 = min(tx+width,width)-1
    y2 = min(ty+height,height)-1

    return x1,y1,x2,y2


def run_one():


    width,height = 64,64  #256, 256


    #jpg_file  = './share/data/kaggle-forest/classification/dummy/train_4.jpg'
    #tif_file  = './share/data/kaggle-forest/classification/dummy/train_4.tif'
    jpg_file  = DATA_DIR + 'share/data/kaggle-forest/classification/image/train-jpg/train_4.jpg'  #9,4,66  ## 21
    tif_file  = DATA_DIR + 'share/data/kaggle-forest/classification/image/train-tif/train_4.tif'

    image_jpg = cv2.imread(jpg_file,1)
    image_tif = io.imread(tif_file)
    image = np.zeros((height,width,7),dtype=np.float32)


    h,w = image_jpg.shape[0:2]
    if height!=h or width!=w:
        image_jpg = cv2.resize(image_jpg,(height,width))
        image_tif = cv2.resize(image_tif,(height,width))

        #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
        #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

    image[:,:,:3] = image_jpg.astype(np.float32)/255.0
    image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
    img = create_image(image)
    im_show('img_before',img,1)
    cv2.waitKey(1)


    ## nomalised and cut
    im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
    im2 = image[:,:,3:]  #image_tif.astype(np.float32)
    im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

    image[:,:,3:] = im2_aligned
    img = create_image(image)
    #draw alignment
    if correlation >=0.5:
        x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )


    im_show('img_after',img,1)
    cv2.waitKey(1)

    # im1 = im1.astype(np.uint8)
    # im2 = im2.astype(np.uint8)
    # im2_aligned = im2_aligned.astype(np.uint8)

    # Show final results
    print('correlation=%f'%correlation)
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)
    # cv2.imshow("im2_aligned", im2_aligned)
    cv2.waitKey(0)


def run_one():


    width,height = 256, 256


    #jpg_file  = './share/data/kaggle-forest/classification/dummy/train_4.jpg'
    #tif_file  = './share/data/kaggle-forest/classification/dummy/train_4.tif'
    jpg_file  = DATA_DIR  + 'share/data/kaggle-forest/classification/image/train-jpg/train_4.jpg'  #9,4,66  ## 21
    tif_file  = DATA_DIR  +'share/data/kaggle-forest/classification/image/train-tif/train_4.tif'

    image_jpg = cv2.imread(jpg_file,1)
    image_tif = io.imread(tif_file)
    image = np.zeros((height,width,7),dtype=np.float32)


    h,w = image_jpg.shape[0:2]
    if height!=h or width!=w:
        image_jpg = cv2.resize(image_jpg,(height,width))
        image_tif = cv2.resize(image_tif,(height,width))

        #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
        #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

    image[:,:,:3] = image_jpg.astype(np.float32)/255.0
    image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
    img = create_image(image)
    im_show('img_before',img,1)
    cv2.waitKey(1)


    ## nomalised and cut
    im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
    im2 = image[:,:,3:]  #image_tif.astype(np.float32)
    im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

    image[:,:,3:] = im2_aligned
    img = create_image(image)
    #draw alignment
    if correlation >=0.5:
        x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
        cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )


    im_show('img_after',img,1)
    cv2.waitKey(1)

    # im1 = im1.astype(np.uint8)
    # im2 = im2.astype(np.uint8)
    # im2_aligned = im2_aligned.astype(np.uint8)

    # Show final results
    print('correlation=%f'%correlation)
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)
    # cv2.imshow("im2_aligned", im2_aligned)
    cv2.waitKey(0)

def run_many():


    width,height = 256, 256
    image = np.zeros((height,width,7),dtype=np.float32)

    for n in range(1000):
        jpg_file  = DATA_DIR+'share/data/kaggle-forest/classification/image/train-jpg/train_%d.jpg'%n
        tif_file  = DATA_DIR+'share/data/kaggle-forest/classification/image/train-tif/train_%d.tif'%n

        image_jpg = cv2.imread(jpg_file,1)
        image_tif = io.imread(tif_file)


        h,w = image_jpg.shape[0:2]
        if height!=h or width!=w:
            image_jpg = cv2.resize(image_jpg,(height,width))
            image_tif = cv2.resize(image_tif,(height,width))

            #cv2.circle(image_jpg, (0,0), 20,(0,0,255),-1) #mark orgin for debug
            #cv2.circle(image_tif, (0,0), 20,(0,0,255),-1)

        image[:,:,:3] = image_jpg.astype(np.float32)/255.0
        image[:,:,3:] = image_tif.astype(np.float32)/4095.0  #2^12=4096
        img = create_image(image)
        im_show('img_before',img,1)
        cv2.waitKey(1)


        ## nomalised and cut
        im1 = image[:,:,:3]  #image_jpg.astype(np.float32)
        im2 = image[:,:,3:]  #image_tif.astype(np.float32)
        im2_aligned, tx, ty, correlation = align_tif_to_jpg(im1, im2)

        image[:,:,3:] = im2_aligned
        img = create_image(image)
        #draw alignment
        if correlation >=0.5:
            x1,y1,x2,y2 = get_rect(-tx,-ty,width=256,height=256)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1 )
            cv2.rectangle(img, (x1+256,y1), (x2+256,y2), (0,0,255),1 )
            cv2.rectangle(img, (x1+2*256,y1), (x2+2*256,y2), (0,0,255),1 )
        else:
            pass

        print('correlation=%f'%correlation)
        im_show('img_after',img,1)
        cv2.waitKey(0)




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_many()

