## make baseline results using standard nets like vgg, resnet, resenxt, densenet, inception

## https://github.com/colesbury/examples/blob/8a19c609a43dc7a74d1d4c71efcda102eed59365/imagenet/main.py
## https://github.com/pytorch/examples/blob/master/imagenet/main.py



from net.common import *
from net.dataset.tool import *
from net.utility.tool import *
from net.rates import *

from net.dataset.product import *

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

#from net.model.fusenet import FuseNet as Net
## max image sizes (w/batch size = 96) ###############

## global setting ################
BASE_DIR = '/home/cory/Kaggle/Cdiscount/'
DATA_DIR = BASE_DIR+'input/'
OUT_DIR = BASE_DIR+'output/resnet152-redemption/'#'output/resnet50-pretrain-7/' 
TRAIN_BATCH_SIZE = 96#50 #96 #140 # 
VALID_BATCH_SIZE = 32 #4 #16 #32 

##--- helper functions  -------------

def change_images(images, agument):

    num = len(images)
    if agument == 'left-right' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,1)

    if agument == 'up-down' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,0)

    # if agument == 'rotate':
    #     for n in range(num):
    #         image = images[n]
    #         images[n] = randomRotate90(image)  ##randomRotate90  ##randomRotate
    #

    if agument == 'transpose' :
        for n in range(num):
            image = images[n]
            images[n] = image.transpose(1,0,2)


    if agument == 'rotate90' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,1)


    if agument == 'rotate180' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,-1)


    if agument == 'rotate270' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,0)

    return images

def do_training():

    out_dir = OUT_DIR 
    initial_checkpoint = None #'./share/project/pytorch/results/kaggle-forest/densenet121-40479-jpg-0/checkpoint0/030.pth'  #None
    initial_model      = None
    pretrained_file = None 

    
    pretrained_file = './share/project/pytorch/pretrained-models/resnet50-19c8e357.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/densenet161-17b70270.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/inception_v3_google-1a9a5a14.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/resnet152-b121ed2d.pth'
    
    #pretrained_file = './share/project/pytorch/pretrained-models/densenet121-241335ed.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/densenet201-4c113574.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/densenet169-6f0f7f60.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/vgg19-dcbb9e9d.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/vgg16-397923af.pth'
    
    #pretrained_file = './share/project/pytorch/pretrained-models/resnet34-333f7ec4.pth'
    #pretrained_file =  './share/project/pytorch/pretrained-models/inceptionv4-58153ba9.pth'
    #pretrained_file = './share/project/pytorch/pretrained-models/resnet18-5c106cde.pth'
    ## ------------------------------------

    initial_checkpoint = '/home/cory/Kaggle/Cdiscount/output/resnet152-redemption/checkpoint/00000011_model.pth' 
    #'../output/resnet50-pretrain-4/checkpoint/00000007_model.pth' 

    # 5 , 3
    ## ------------------------------------

    os.makedirs(out_dir +'/snap', exist_ok=True)
    os.makedirs(out_dir +'/checkpoint', exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED    = %u\n' % SEED)
    log.write('\tfile    = %s\n' % __file__)
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    #batch_size  = BATCH_SIZE 

    train_dataset, valid_dataset, classes_dict = produce_split_dataset()
    num_classes = len(classes_dict)
    log.write('num_classes={}\n'.format(num_classes))
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = SequentialSampler(train_dataset), #RandomSampler(train_dataset),  ##
                        batch_size  = TRAIN_BATCH_SIZE,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)

    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),  #None,
                        batch_size  = VALID_BATCH_SIZE, 
                        drop_last   = False,
                        num_workers = 3,
                        pin_memory  = True)

    height, width , in_channels   = valid_dataset.height, valid_dataset.width, valid_dataset.in_channels
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttrain_dataset.num    = %d\n'%(train_dataset.num))
    log.write('\tvalid_dataset.num     = %d\n'%(valid_dataset.num))
    log.write('\ttrain_batch_size           = %d\n'%(TRAIN_BATCH_SIZE))
    log.write('\tvalid_batch_size           = %d\n'%(VALID_BATCH_SIZE))
    log.write('\ttrain_loader.sampler = %s\n'%(str(train_loader.sampler)))
    log.write('\tvalid_loader.sampler  = %s\n'%(str(valid_loader.sampler)))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = Net(in_shape = (in_channels, height, width), num_classes=int(num_classes))
    #, pretrained=True)

    net.cuda()
    #log.write('\n%s\n'%(str(net)))
    log.write('%s\n\n'%(type(net)))
    log.write(inspect.getsource(net.__init__)+'\n')
    log.write(inspect.getsource(net.forward)+'\n')
    log.write('\n')


    #optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
    momentum = 0.9
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    # Adagrad does well @ 0.001!!

    #resume from previous?
    start_epoch=0
    
    if initial_checkpoint is not None:
        net.load_state_dict(torch.load(initial_checkpoint))

        checkpoint = torch.load(initial_checkpoint.replace('_model.pth','_optimizer.pth'))
        start_epoch = checkpoint['iter']+1
        optimizer.load_state_dict(checkpoint['optimizer'])
        #adjust_momentum(optimizer, momentum)
        #print('momentum adjusted to {}'.format(get_momentum(optimizer)))
    elif pretrained_file is not None:  #pretrain
        skip_list = ['fc.weight', 'fc.bias']
        load_valid(net, pretrained_file, skip_list=skip_list)

    if initial_model is not None:
       net = torch.load(initial_model)

    ## optimiser ----------------------------------

    #lr_steps = {0: 0.01, 1: 0.005, 2: -1} 0.47
    #lr_steps = {0: 0.01, 3: 0.005, 4: 0.001, 5: -1} 0.57
    #lr_steps = {0: 0.005, 3: 0.001, 5: 0.0005, 6: -1} # .613
    #lr_steps = {0: 0.01, 1: 0.005, 3: 0.001, 5: 0.0005, 6: 0.0001, 7: 0.00005, 8: -1} # .64
    #lr_steps = {0: 0.01, 1: 0.005, 4: 0.001, 7: 0.0005, 9: 0.0001, 10: 0.00005, 11: 0.000001, 12: -1} 
    lr_steps = {11: 0.00001, 13: 0.000005, 15: 0.0000005, 16: -1} 
    # hijack session!
    #lr_steps = {0: 0.01, 1: 0.005, 3: 0.001, 5: 0.0005, 6: 0.0001, 7: 0.0005, 9:  0.00005, 10: -1}

    LR = StepLR(steps=list(lr_steps.keys()), \
                rates=list(lr_steps.values()))
 
    num_epoches = 17 #8 #7 #6 #2 
    it_print    = 10 
    epoch_valid  = 1
    epoch_save  = 2 #8 #5

    ## start training here! ##############################################3
    ## start training here! ##############################################3
    log.write('** start training here! **\n')

    log.write(' optimizer=%s\n'%str(optimizer) )
    log.write(' LR=%s\n\n'%str(LR) )
    log.write(' epoch   iter   rate  |  smooth_loss   |  train_loss  (acc)  |  valid_loss  (acc)  | min\n')
    log.write('----------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    valid_loss   = np.nan
    valid_acc    = np.nan
    time = 0

    start0 = timer()
    for epoch in range(start_epoch, num_epoches):  # loop over the dataset multiple times
        #print ('epoch=%d'%epoch)
        start = timer()

        #---learning rate schduler ------------------------------
        lr =  LR.get_rate(epoch, num_epoches)
        if lr<0 :break

        adjust_learning_rate(optimizer, lr)
        rate =  get_learning_rate(optimizer)[0] #check
        #print(get_momentum(optimizer))
        #--------------------------------------------------------
        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):
            #log.write("batch {}x training with size of {}\n".format(it, batch_size))
            logits, probs = net(Variable(images.cuda()))
            loss  = criterion(logits, labels.cuda())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #additional metrics
            sum_smooth_loss += loss.data[0]
            sum += 1
            ##print(net.features.state_dict()['0.conv.weight'][0,0])

            #print('it:', it, it_print, it % it_print)
            # print statistics
            if it % it_print == it_print-1:
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = acc_measure(probs.data, labels.cuda()) 
                train_loss = loss.data[0]
                #print('loss: ' + str(loss.data[0]))

                print('\r%5.1f   %5d    %0.5f   |  %0.3f  | %0.3f  %5.3f | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_loss, train_acc),\
                        end='',flush=True)


        #---- end of one epoch -----
        end = timer()
        time = (end - start)/60

        if epoch % epoch_valid == epoch_valid-1  or epoch == num_epoches-1:
            #print('running val')
            net.cuda().eval()
            valid_loss,valid_acc = evaluate(net, valid_loader)
            #print('valid:', valid_loss, valid_acc)
            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.5f   |  %0.3f  | %0.3f  %5.3f | %0.3f  %5.3f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_loss, train_acc, valid_loss,valid_acc, time))

        if (epoch+1)%epoch_save==0: #epoch in epoch_save:
            
            #if epoch % epoch_save == epoch_save-1 or epoch == num_epoches-1:
            print('saving checkpoint')
            torch.save(net.state_dict(),out_dir +'/checkpoint/%08d_model.pth'%(epoch))
            torch.save({
                'optimizer' : optimizer.state_dict(),
                'iter'      : epoch,
            }, out_dir +'/checkpoint/%08d_optimizer.pth'%(epoch))
            #torch.save(net, out_dir +'/snap/%03d.torch'%(epoch+1))
            #torch.save({
            #    'state_dict': net.state_dict(),
            #    'optimizer' : optimizer.state_dict(),
            #    'epoch'     : epoch,
            #}, out_dir +'/checkpoint/%03d.pth'%(epoch+1))
            ## https://github.com/pytorch/examples/blob/master/imagenet/main.py




    #---- end of all epoches -----
    end0  = timer()
    time0 = (end0 - start0) / 60

    ## check : load model and re-valid
    torch.save(net,out_dir +'/snap/final.torch')
    if 1:
        net = torch.load(out_dir +'/snap/final.torch')

        net.cuda().eval()
        valid_loss, valid_acc, predictions = evaluate_and_predict( net, valid_loader, num_classes)

        log.write('\n')
        log.write('%s:\n'%(out_dir +'/snap/final.torch'))
        log.write('\tall time to train=%0.1f min\n'%(time0))
        log.write('\tvalid_loss=%f, valid_acc=%f\n'%(valid_loss,valid_acc))

        #assert(type(valid_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        #dump_results(out_dir +'/train' , predictions, valid_dataset.labels, valid_dataset.names)

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    do_training()
    print('\nsucess!')
