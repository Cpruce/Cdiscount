# https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py
# https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb
# https://discuss.pytorch.org/t/print-autograd-graph/692
#


from net.common import *
from graphviz import Digraph
from torch.autograd import Variable
import builtins
import time

# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# net ------------------------------------

## this is broken !!!
def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '('+(', ').join(['%d'% v for v in var.size()])+')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])
    add_nodes(var.creator)
    return dot


# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_momentum(optimizer, momentum):
    for param_group in optimizer.param_groups:
        param_group['momentum'] = momentum

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def get_momentum(optimizer):
    momentum=[]
    for param_group in optimizer.param_groups:
       momentum +=[ param_group['momentum'] ]
    return momentum


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3 ######

#params, stats = state_dict['params'], state_dict['stats']
#https://github.com/szagoruyko/attention-transfer/blob/master/imagenet.py
"""
def load_valid(model, pretrained_file, skip_list=None):

    pretrained_dict = torch.load(pretrained_file)
    model_dict = model.state_dict()
    print("len(model_dict)=", len(model_dict), model_dict.keys())
    # 1. filter out unnecessary keys
    #pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }
    pretrained_dict1 = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and k not in skip_list:
            print(k, len(v), len(model_dict[k]))
            pretrained_dict1[k] = v
    print("len(pretrained_dict)=",len(pretrained_dict), pretrained_dict.keys())
    print("len(pretrained_dict1)=",len(pretrained_dict1), pretrained_dict1.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)
"""
def load_valid(model, pretrained_file, skip_list=[], log=None):

    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_file)
    # 1. filter out unnecessary keys
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }

    ## debug
    if 1:
      print('model_dict.keys()')
      print(model_dict.keys())
      print('pretrained_dict.keys()')
      print(pretrained_dict.keys())
      print('pretrained_dict1.keys()')
      print(pretrained_dict1.keys())

    #pring missing keys
    if log is not None:
      log.write('--missing keys at load_valid():--\n')
      for k in model_dict.keys():
        if k not in pretrained_dict1.keys():
          log.write('\t %s\n'%k)

      log.write('------------------------\n')
    else:
      print('--missing keys at load_valid():--')
      for k in model_dict.keys():
        if k not in pretrained_dict1.keys():
          print('\t %s'%k)
          pretrained_dict1[k] = model_dict[k]

      print('------------------------')


    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)
