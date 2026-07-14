import os
import sys
sys.path.append(os.path.abspath('vdst'))
import torch
import torch.amp as amp

from utils.config import load_config
from dataset.wildrgbd import WildRGBDDataset
from model.model import VDST


n_frames = 1
device = 'cuda:0'
render_resolution = (256, 256)
window_resolution = (800, 800)

config_path = './vdst/config.yaml'
model_path = './checkpoints/vdst.pt'

config, _ = load_config(config_path, [])

dataset = WildRGBDDataset(
    config.train.data.datasets.wildrgbd.path,
    config.train.data.n_sources,
    1,
    output_dims=config.train.data.output_dims,
    dataset_type=config.train.data.datasets.wildrgbd.dataset_type,
    val_split=0.005,
    test_split=0.02,
    split='train',
    test_category='truck',
    seed=42
)

scene = dataset[6]

scene.targets.images = None
scene.targets.depths = None
scene.targets.depth_masks = None

for p in (scene.sources, scene.targets):
    for k in p.keys():
        if isinstance(p[k], torch.Tensor):
            p[k] = p[k].to(device)

model = VDST(config.model).to(device)
checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=False
)
model.load_state_dict(checkpoint)
model.eval()

R, t = scene.sources.R[0], scene.sources.t[0]
initial_T = torch.concat([torch.concat([R, t.unsqueeze(-1)], dim=-1), torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).reshape(1, -1)], dim=-2)

def render(T, frame_index):
    global model
    global scene
    
    print(T.shape)
    
    scene.targets.R, scene.targets.t = T[:3, :3].unsqueeze(0), T[:3, 3].unsqueeze(0)
    with amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        outputs = model(scene).gen_targets
    image, depth = outputs.images[0], outputs.depths[0, 0]
    
    return image, depth
