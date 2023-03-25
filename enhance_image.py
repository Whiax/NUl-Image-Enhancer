# =============================================================================
# Imports
# =============================================================================
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
from models import MLP, normalized_pt, get_transform_to_params
from os.path import basename, join, exists
from PIL import Image
import numpy as np
import argparse
import random
import torch
import clip
import time
import os 

#flag/folder/temp
print('Starting image enhancer...')
start_time = time.time()
id_run = int(start_time)
device = "cuda" if torch.cuda.is_available() else "cpu"
runif = np.random.uniform
#clean tmp
cur_folder = os.path.dirname(os.path.realpath(__file__))
tmp_folder = join(cur_folder, '_tmp')
if not exists(tmp_folder):
    os.mkdir(tmp_folder)
else:
    for base in os.listdir(tmp_folder):
        pth_to_rm = join(tmp_folder, base)
        assert len(pth_to_rm) > 10, "safety check"
        os.remove(pth_to_rm)


# =============================================================================
# Methods
# =============================================================================
#image => aesthetic score
def measure_aesthetic(pil_image, model_clip, model_evalaesthetic, flip=True):
    image = pil_image
    image = ToTensor()(image)
    image = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        images = [image]
        if flip:
            image_flip = torch.flip(image,(2,))
            images += [image_flip]
        aesthetic_values = []
        for img in images:
            image_features = model_clip.encode_image(img)
            im_emb_arr = normalized_pt(image_features)
            aesthetic_values += [model_evalaesthetic(im_emb_arr.float())]
        aesthetic_value = torch.stack(aesthetic_values).mean(0)
    return round(aesthetic_value[0].item(),4)

#get a random transform
def get_random_transform(transforms_list, transform_to_params):
    #get random transform & params
    t = random.choice(transforms_list)
    random_kwargs = transform_to_params[t]
    kwargs = {}
    for k,v in random_kwargs.items():
        if len(v) == 1:
            kwargs[k] = round(runif(*v[0]),3)
        else:
            vx = v[0](*v[1])
            if v[0] != np.random.randint:
                vx = round(v, 3)
            if k == 'kernel_size':
                if vx % 2 == 0:
                    vx += 1
            kwargs[k] = vx
    return t, kwargs

#get an optimal transform 
def get_optimal_transform(image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic):
    t = transforms_list[it%len(transforms_list)]
    best_score = 0
    best_kwargs = {}
    param_keys = list(transform_to_params[t].keys())
    v2s = {}
    for key in param_keys:
        min_v = transform_to_params[t][key][-1][0]
        max_v = transform_to_params[t][key][-1][1]
        values_to_try = np.linspace(min_v, max_v, N_TRY)
        if transform_to_params[t][key][0] == np.random.randint:
            values_to_try = np.linspace(min_v, max_v-1, N_TRY)
            values_to_try = set([int(np.floor(e)) for e in values_to_try])
        if key == 'kernel_size':
            values_to_try = [v for v in values_to_try if v % 2 != 0]
        for v in values_to_try:
            kwargs = {k:1 for k in param_keys}
            kwargs[key] = v
            post_image = t(image, **kwargs)
            score = measure_aesthetic(post_image, model_clip, model_evalaesthetic)
            v2s[v] = score
            if score > best_score:
                best_score = score
                best_kwargs = kwargs
    return t, best_kwargs

# =============================================================================
# Read parameters
# =============================================================================
print('- reading args...')
image_pth = None

#setup args in ipython
# image_pth = './examples/1.png'
# manual_args = None

#auto parse args
manual_args = image_pth
parser = argparse.ArgumentParser()
parser.add_argument("image_pth")
parser.add_argument("--maxruntime", nargs='?', type=int, help='maximum runtime in seconds (default: 60s)', default=60)
args = parser.parse_args([manual_args]) if manual_args else parser.parse_args()
image_pth = args.image_pth
maxruntime = args.maxruntime

#Flags / init / constant
n_iter = 2000
max_delay = maxruntime 
mode = 'soft'
N_TRY = 20 if mode == 'hard' else 10

# =============================================================================
# Data
# =============================================================================
print('- loading data...')
#read image
base_pil_image = Image.open(image_pth).convert("RGB")
base_pil_image.save(join(tmp_folder,f'{id_run}_step_0.jpg'))
base_pil_image_c = CenterCrop(224)(Resize(224)(base_pil_image))
total_light_base = ToTensor()(base_pil_image).mean()


# =============================================================================
# Models
# =============================================================================
print('- loading models...')
kwargs = {} if device == 'cuda' else {'map_location':torch.device('cpu')}
model_evalaesthetic = MLP(768)  
model_evalaesthetic.load_state_dict(torch.load("sac+logos+ava1-l14-linearMSE.pth",**kwargs))
model_evalaesthetic.to(device)
model_evalaesthetic.eval()
model_clip, preprocess = clip.load("ViT-L/14", device=device)  
 
#evaluate base score
base_score = measure_aesthetic(base_pil_image_c, model_clip, model_evalaesthetic)
print('- base_score:', base_score)



# =============================================================================
# LOOP
# =============================================================================
#init decision tree
node_to_score = {}
dtree = [[0]]
node_to_score[0] = base_score
#node parameters
thres_bestscorelow = 0.2
thres_basescorelow = 0.1
max_it_try_random = 500
p_non_optimal = 0.8
ratio_light_thres = 0.85

#init transforms
transform_to_params = get_transform_to_params(mode)
transforms_list = list(transform_to_params.keys())

#reproduce the transforms of a node of the tree (recursive)
def apply_node(image, node):
    if node == 0:
        return image
    elif isinstance(node, tuple):
        return node[0](image, **node[1])
    elif isinstance(node, int):
        nodes = dtree[node]
        for node in nodes:
            image = apply_node(image, node)
    return image

#init
it = 1
best_score = base_score
best_node = 0
start_loop_time = time.time()
last_save_time = time.time() #-9999
last_improv_it = 0
print('- starting loop')
#for loop
for it in range(it, n_iter):
    #decide of previous node 
    do_optimal = it < last_improv_it + len(transform_to_params)
    if it < max_it_try_random:
        prev_node = int(runif(0, max(1, len(dtree))))
    else:
        #start from random node 
        if runif(0,1) < p_non_optimal and not do_optimal:
            prev_node = int(runif(0, len(dtree)))
            #filter good
            worst_than_best = node_to_score[prev_node] < best_score - thres_bestscorelow
            worst_than_base = node_to_score[prev_node] < base_score - thres_basescorelow
            if worst_than_best or worst_than_base: continue
        #use best node
        else:
            prev_node = best_node
    
    #apply previous node
    image = base_pil_image_c
    image = apply_node(image, prev_node)
    #apply new node
    if do_optimal:
        t, kwargs = get_optimal_transform(image, it, transforms_list, transform_to_params, model_clip, model_evalaesthetic)
    else:
        t, kwargs = get_random_transform(transforms_list, transform_to_params)
    image = t(image, **kwargs)
    
    #process
    score = measure_aesthetic(image, model_clip, model_evalaesthetic)
    
    #post process score (fix "too dark = great" as evaluated by the unperfect model)
    total_light_post = ToTensor()(image).mean()
    ratio_light = total_light_post / total_light_base
    if ratio_light < ratio_light_thres:
        score -= (1-ratio_light) * 6
    
    #add to tree
    current_node = len(dtree)
    dtree += [[prev_node, (t, kwargs)]]
    #log
    log_it = f'- it: {it}/{n_iter} | score: {score:.2f} | best_score: {best_score:.2f} / {base_score:.2f}'
    logged = False
    if len(node_to_score) == 0 or score > best_score:
        best_node = current_node
        # if manual_args is not None:
        #     plt.imshow(image)
        #     plt.show()
        if time.time() - last_save_time > 5 or it < 100:
            apply_node(base_pil_image, best_node).save(join(tmp_folder,f'{id_run}_step_{it}.jpg'))
            last_save_time = time.time()
        best_score = score
        last_improv_it = it
        #log
        log_it = f'- it: {it}/{n_iter} | score: {score:.2f} | best_score: {best_score:.2f} / {base_score:.2f}'
        logged = True
        print(log_it)
    node_to_score[current_node] = score
    if (it % 100 == 0 or it < 10) and logged == False: #may be skipped
        print(log_it)
    
    #if too long, stop
    if time.time() - start_loop_time > max_delay:
        print(f'- process stopped, over {max_delay}s')
        break

best_score = max(node_to_score.values())
print('- base_score:', base_score)
print('- best_score:', best_score, f'(+{best_score-base_score:.2f})')


# #display best image
# best_node = [k for k,v in node_to_score.items() if v == max(node_to_score.values())][0]
# image = apply_node(base_pil_image_c, best_node)
# score = measure_aesthetic(image)
# assert score == max(node_to_score.values())

#save out
out_folder = join(cur_folder, 'results')
if not exists(out_folder):
    os.mkdir(out_folder)
base_pth = basename(image_pth)
out_pth = join(out_folder, f'{id_run}_best_{base_pth}')
apply_node(base_pil_image, best_node).save(out_pth)

print('- out_pth:', out_pth)
print('- total runtime:', int(time.time() - start_time), 's')


















