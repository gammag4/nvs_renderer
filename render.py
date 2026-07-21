from importlib import util
import os
import argparse
import math
from easydict import EasyDict as edict
import torch
import pygame
import pygame_gui
import einx

import profiler


def compute_transform_matrix(controls, current_state, device):
    z, x, y = -controls.forward, -controls.right, controls.up
    rotX, rotY = current_state[3:]
    rotX += -controls.mouseDelta[0]
    rotY += controls.mouseDelta[1]
    
    theta = rotY
    ct, st = math.cos(theta), math.sin(theta)
    RX = torch.tensor([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    
    theta = rotX
    ct, st = math.cos(theta), math.sin(theta)
    RY = torch.tensor([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    
    R = RX @ RY
    
    current_pos = torch.tensor(current_state[:3], dtype=torch.float32).reshape(3, 1)
    current_pos = current_pos + R.T @ torch.tensor([[x, y, z]], dtype=torch.float32).T
    
    R = torch.concat([torch.concat([R, torch.tensor([[0, 0, 0]]).T], dim=1), torch.tensor([[0, 0, 0, 1]])])
    T = torch.eye(4, dtype=torch.float32)
    T[:3, 3:] = current_pos
    
    T = (R @ T).inverse()
    
    current_state = (*current_pos.squeeze().tolist(), rotX, rotY)
    
    return T.to(device), current_state


# TODO this func is redundant remove later
def get_tensor_info(tensor: torch.Tensor, is_image) -> dict:
    if not tensor.is_cuda:
        raise RuntimeError("Tensor must be on CUDA device")
    
    if is_image:
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    else:
        tensor = torch.where(tensor == 0, 1e+3, tensor)
        tensor = tensor.clamp(1e-3, 1e+3) # from 1mm to 1km
        tensor = 1.0 / tensor
        tensor = tensor / tensor.max()
    
    tensor = (tensor * 255).to(dtype=torch.uint8)
    
    if is_image:
        tensor = einx.id('c h w -> h w c', tensor)
    else:
        tensor = einx.id('h w -> h w 3', tensor)
    
    # alpha channel
    ones = torch.ones((*tensor.shape[-3:-1], 1), dtype=tensor.dtype, device=tensor.device) * 255
    tensor = torch.cat([tensor, ones], dim=-1)
    
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    h, w, c = tensor.shape
    row_bytes = w * c * tensor.element_size()
    rows = h
    
    res = edict(
        tensor=tensor,
        pointer=tensor.data_ptr(),
        row_bytes=row_bytes,
        rows=rows,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        device_id=tensor.device.index or 0,
    )
    
    return res


def render_func(frame_index, render, T, sources):
    """
    Returns a GPU tensor of shape (3, 512, 512) in CUDA memory.
    This should not be changed bc it will be used later with images this shape.
    Implement this with your actual tensor generation logic.
    """
    
    image, depth = render(T, frame_index)
    image = get_tensor_info(image, is_image=True)
    depth = get_tensor_info(depth, is_image=False)
    if sources is not None:
        source_images = get_tensor_info(einx.id('b c h w -> c h (b w)', sources.images), is_image=True).tensor
        source_depths = get_tensor_info(einx.id('b c h w -> (c h) (b w)', sources.depths), is_image=False).tensor
        sources = einx.id('b h w c -> (b h) w c', torch.stack([source_images, source_depths]))
    
    return image, depth, sources


def get_controls(dt, is_navigating):
    if not is_navigating:
        return edict(right=0.0, up=0.0, forward=0.0, mouseDelta=(0.0, 0.0))
    
    speed = 1.0
    mouse_sensitivity = 0.001
    
    forward = 0.0
    right = 0.0
    up = 0.0
    mouseDelta = pygame.mouse.get_rel()
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        forward += 1
    if keys[pygame.K_s]:
        forward -= 1
    if keys[pygame.K_a]:
        right -= 1
    if keys[pygame.K_d]:
        right += 1
    if keys[pygame.K_LCTRL]:
        up -= 1
    if keys[pygame.K_SPACE]:
        up += 1
    
    controls = edict()
    
    vmp, mmp = speed * dt, mouse_sensitivity * torch.pi
    
    controls.right = right * vmp
    controls.up = up * vmp
    controls.forward = forward * vmp
    controls.mouseDelta = (mouseDelta[0] * mmp, mouseDelta[1] * mmp)
    
    return controls


def render_model(sources, n_frames, initial_T, render, device, render_resolution, window_resolution=(800, 800)):
    w, h = render_resolution
    win_w, win_h = window_resolution
    current_res = window_resolution
    
    R = initial_T[:3, :3]
    x, y, z = (-R.T @ initial_T[:3, 3:]).squeeze().tolist()
    rotX, rotY = 0.0, 0.0 # TODO compute from R.T
    current_state = (x, y, z, rotX, rotY)
    
    # Setup pygame for keyboard input
    pygame.init()
    screen = pygame.display.set_mode(window_resolution)
    manager = pygame_gui.UIManager(window_resolution)
    pygame.display.set_caption("NVS Renderer")
    
    is_navigating = True
    is_playing = False
    is_rendering_depth = False
    is_displaying_source_images = False
    
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    surface = pygame.Surface(render_resolution)
    
    frame_index = 0
    time_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((0, -50), (win_w / 2, 30)),
        start_value=0,
        value_range=(0, 1 if n_frames == 1 else n_frames - 1),
        manager=manager,
        anchors={
            'centerx': 'centerx',
            'bottom': 'bottom'
        }
    )
    play_button = pygame_gui.elements.UIButton(
        relative_rect=pygame.Rect((20, -50), (100, 30)), # Position and size (x, y), (width, height)
        text='Play',
        manager=manager,
        anchors={
            'left': 'left',
            'bottom': 'bottom'
        }
    )
    if n_frames == 1:
        time_slider.hide()
        play_button.hide()
    
    # Control loopimport asyncio
    clock = pygame.time.Clock()
    while True:
        with profiler.RegionProfiler('main_loop'):
            dt = clock.tick() / 1000.0
            print(f'fps: {clock.get_fps()}')
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        is_rendering_depth = not is_rendering_depth
                    if event.key == pygame.K_r and sources is not None:
                        is_displaying_source_images = not is_displaying_source_images
                        if is_displaying_source_images:
                            current_res = (window_resolution[0] * 2, window_resolution[1])
                            screen = pygame.display.set_mode(current_res)
                            surface = pygame.Surface((render_resolution[0] * 4, render_resolution[1] * 2))
                        else:
                            current_res = window_resolution
                            screen = pygame.display.set_mode(current_res)
                            surface = pygame.Surface(render_resolution)
                    if event.key == pygame.K_ESCAPE:
                        if not is_navigating:
                            pygame.quit()
                            return
                        
                        is_navigating = False
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                
                consumed = False
                
                if not is_navigating:
                    # Time slider
                    if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                        if event.ui_element == time_slider:
                            frame_index = int(event.value)
                    
                    # Play button
                    if event.type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == play_button:
                            is_playing = not is_playing
                            play_button.set_text('Pause' if is_playing else 'Play')
                    
                    consumed = manager.process_events(event)
                
                # Check click on screen
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # left mouse
                        if not consumed:
                            is_navigating = True
                            pygame.mouse.set_visible(False)
                            pygame.event.set_grab(True)
                            
                            # discards current mouse delta to prevent mouse drift when switching back
                            _ = pygame.mouse.get_rel()
            
            if is_playing:
                frame_index = (frame_index + 1) % n_frames
                time_slider.set_current_value(frame_index)
            
            manager.update(dt)
            
            controls = get_controls(dt, is_navigating)
            T, current_state = compute_transform_matrix(controls, current_state, device)
            image, depth, sources_new = render_func(frame_index, render, T, sources)
            img_to_render = depth if is_rendering_depth and depth is not None else image
            img_to_render = img_to_render.tensor
            if is_displaying_source_images and sources is not None:
                img_to_render = img_to_render.repeat_interleave(repeats=2, dim=0).repeat_interleave(repeats=2, dim=1)
                img_to_render = einx.id('b h w c -> h (b w) c', torch.stack([sources_new, img_to_render]))
            canvas = einx.id('h w c -> w h c', img_to_render)[:, :, :3].cpu().numpy()
            
            pygame.surfarray.blit_array(surface, canvas)
            
            # Render camera feed
            screen.blit(pygame.transform.scale(surface, current_res), (0, 0))
            
            if not is_navigating:
                manager.draw_ui(screen)
            
            pygame.display.flip()
        
        profiler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--scene_index", required=False, default=0)
    args = parser.parse_args()
    path = args.module
    scene_index = int(args.scene_index)
    
    name = os.path.splitext(os.path.split(path)[-1])[0]
    module_spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(module_spec)
    module.scene_index = scene_index
    module_spec.loader.exec_module(module)
    
    (
        n_frames,
        initial_T,
        render,
        device,
        render_resolution,
        sources
    ) = [getattr(module, n) for n in [
        'n_frames',
        'initial_T',
        'render',
        'device',
        'render_resolution'
    ]] + [getattr(module, n, None) for n in [
        'sources'
    ]]
    
    window_resolution = getattr(module, 'window_resolution', None)
    
    with profiler.Profiler(should_profile=True, warmup=20):
        # T (4, 4), render(T) -> img (c, h, w), device, resolution
        render_model(sources, n_frames, initial_T, render, device, render_resolution, window_resolution or (800, 800))
    
    profiler.print_results()
    # profiler.dump('profiler_dump.pt')
