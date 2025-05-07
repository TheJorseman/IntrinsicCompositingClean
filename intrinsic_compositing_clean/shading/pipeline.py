import torch
from torch.optim import Adam

import numpy as np

from skimage.transform import resize

from intrinsic_compositing_clean.lib.general import uninvert, invert, round_32, view

from intrinsic_compositing_clean.altered_midas.midas_net import MidasNet

def load_reshading_model(path, device='cpu'):
    state_dict = torch.load(path, map_location=device) 
    shd_model = MidasNet(input_channels=9)
    shd_model.load_state_dict(state_dict)
    shd_model = shd_model.eval()
    shd_model = shd_model.to(device)

    return shd_model

import numpy as np
from scipy.optimize import minimize

device = "cuda" if torch.cuda.is_available() else "cpu"

# Función para asegurar que un tensor tenga gradientes habilitados
def ensure_grad(tensor):
    if tensor is not None and not tensor.requires_grad:
        tensor.requires_grad_(True)
    return tensor

# Función para verificar si todos los parámetros tienen gradientes habilitados
def check_grad_status(model=None, named_tensors=None, tensors=None):
    print("=== Estado de Gradientes ===")
    
    # Verificar modelo si se proporciona
    if model is not None:
        for name, param in model.named_parameters():
            print(f"Parámetro: {name}, requires_grad: {param.requires_grad}, "
                  f"grad_fn: {param.grad_fn}")
    
    # Verificar tensores nombrados
    if named_tensors is not None:
        for name, tensor in named_tensors.items():
            if tensor is not None:
                print(f"Tensor: {name}, requires_grad: {tensor.requires_grad}, "
                      f"grad_fn: {tensor.grad_fn}")
    
    # Verificar lista de tensores
    if tensors is not None:
        for i, tensor in enumerate(tensors):
            if tensor is not None:
                print(f"Tensor[{i}], requires_grad: {tensor.requires_grad}, "
                      f"grad_fn: {tensor.grad_fn}")
    
    print("===========================")

# Función para habilitar gradientes en todos los parámetros de un modelo
def enable_grad_for_model(model):
    for param in model.parameters():
        param.requires_grad_(True)
    return model

# Función para habilitar gradientes en una lista o diccionario de tensores
def enable_grad_for_tensors(tensors):
    if isinstance(tensors, dict):
        for key, tensor in tensors.items():
            if tensor is not None:
                tensor.requires_grad_(True)
    elif isinstance(tensors, (list, tuple)):
        for tensor in tensors:
            if tensor is not None:
                tensor.requires_grad_(True)
    return tensors

def spherical2cart(r, theta, phi):
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return x, y, z



def spherical2cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def run_optimization_no_backward(params_init, A, b, max_iter=500):
    # Función objetivo para minimizar
    A = A.numpy()
    b = b.numpy()
    def objective(params):
        theta, phi, r, offset = params
        
        # Conversión de coordenadas esféricas a cartesianas
        x, y, z = spherical2cart(r, theta, phi)
        
        # Cálculo de la predicción
        dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
        pred_shd = dir_shd + offset
        
        # Cálculo del error (MSE)
        loss = np.mean((pred_shd - b) ** 2)
        return loss

    # Restricciones para los parámetros
    bounds = [
        (0, np.pi / 2),  # theta: 0 a 90 grados
        (0, 2 * np.pi),  # phi: 0 a 360 grados
        (0, None),       # r: radio positivo
        (0.1, None)      # offset: mínimo 0.1
    ]
    # Minimización usando Nelder-Mead (no requiere gradientes)
    result = minimize(
        objective, 
        params_init.numpy(), 
        method='Nelder-Mead', 
        bounds=bounds, 
        options={'maxiter': max_iter}
    )

    optimized_params = result.x
    final_loss = result.fun

    return final_loss, optimized_params


def run_optimization(params, A, b):
    import pdb;pdb.set_trace()
    params = params.clone().detach().requires_grad_(True)
    ensure_grad(A)
    ensure_grad(b)
    
    # Verificar estado inicial de gradientes
    check_grad_status(named_tensors={"params": params, "A": A, "b": b})
    
    optim = Adam([params], lr=0.01)
    prev_loss = 1000.0
    
    for i in range(500):
        optim.zero_grad()

        # Conversión esférica a cartesiana
        #x, y, z = 
        x = params[2] * torch.sin(params[0]) * torch.cos(params[1])
        y = params[2] * torch.sin(params[0]) * torch.sin(params[1])
        z = params[2] * torch.cos(params[0])
        
        #spherical2cart(params[2], params[0], params[1])

        # Cálculo de predicción
        dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
        pred_shd = dir_shd + params[3]

        # Cálculo de pérdida
        loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)

        # Retropropagación
        loss.backward()

        # Actualización de parámetros
        optim.step()

        # Aplicar restricciones de manera diferenciable usando torch.clamp()
        # Actualizando directamente params.data mantiene la diferenciabilidad
        params.data[0] = torch.clamp(params.data[0], min=0.0, max=float(np.pi / 2))
        params.data[1] = torch.clamp(params.data[1], min=0.0, max=float(2 * np.pi))
        params.data[2] = torch.clamp(params.data[2], min=0.0)
        params.data[3] = torch.clamp(params.data[3], min=0.1)
        
        # Usar .item() para obtener el valor escalar de loss para comparación
        current_loss = loss.item()
        delta = prev_loss - current_loss
            
        if delta < 0.0001:
            break
            
        prev_loss = current_loss
        
    return loss, params

def test_init(params, A, b):
    x, y, z = spherical2cart(params[2], params[0], params[1])

    dir_shd = (A[:, 0] * x) + (A[:, 1] * y) + (A[:, 2] * z)
    pred_shd = dir_shd + params[3]

    loss = torch.nn.functional.mse_loss(pred_shd.reshape(-1), b)
    return loss

def get_light_coeffs(shd, nrm, img, mask=None, bias=True):
    img = resize(img, shd.shape)

    reg_shd = uninvert(shd)
    valid = (img.mean(-1) > 0.05) * (img.mean(-1) < 0.95)

    if mask is not None:
        valid *= (mask == 0)
    
    nrm = (nrm * 2.0) - 1.0
    
    A = nrm[valid == 1]
    # A = nrm.reshape(-1, 3)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    
    b = reg_shd[valid == 1]
    # b = reg_shd.reshape(-1)
    
    # parameters are theta, phi, and bias (c)
    A = torch.from_numpy(A)
    b = torch.from_numpy(b)
    
    min_init = 1000
    for t in np.arange(0, np.pi/2, 0.1):
        for p in np.arange(0, 2*np.pi, 0.25):
            params = torch.nn.Parameter(torch.tensor([t, p, 1, 0.5]))
            init_loss = test_init(params, A, b)
    
            if init_loss < min_init:
                best_init = params
                min_init = init_loss
                # print('new min:', min_init)
    
    loss, params = run_optimization_no_backward(best_init, A, b)
    
    nrm_vis = nrm.copy()
    nrm_vis = draw_normal_circle(nrm_vis, (50, 50), 40)
    
    x, y, z = spherical2cart(params[2], params[0], params[1])

    coeffs = torch.tensor([x, y, z]).reshape(3, 1).detach().numpy()
    out_shd = (nrm_vis.reshape(-1, 3) @ coeffs) + params[3].item()

    coeffs = np.array([x.item(), y.item(), z.item(), params[3].item()])

    return coeffs, out_shd.reshape(shd.shape)

def draw_normal_circle(nrm, loc, rad):
    size = rad * 2

    lin = np.linspace(-1, 1, num=size)
    ys, xs = np.meshgrid(lin, lin)

    zs = np.sqrt((1.0 - (xs**2 + ys**2)).clip(0))
    valid = (zs != 0)
    normals = np.stack((ys[valid], -xs[valid], zs[valid]), 1)

    valid_mask = np.zeros((size, size))
    valid_mask[valid] = 1

    full_mask = np.zeros((nrm.shape[0], nrm.shape[1]))
    x = loc[0] - rad
    y = loc[1] - rad
    full_mask[y : y + size, x : x + size] = valid_mask
    # nrm[full_mask > 0] = (normals + 1.0) / 2.0
    nrm[full_mask > 0] = normals

    return nrm

def generate_shd(nrm, coeffs, msk, bias=True, viz=False):
    
    # if viz:
        # nrm = draw_normal_circle(nrm.copy(), (50, 50), 40)

    nrm = (nrm * 2.0) - 1.0

    A = nrm.reshape(-1, 3)
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    A_fg = nrm[msk == 1]
    A_fg /= np.linalg.norm(A_fg, axis=1, keepdims=True)

    if bias:
        A = np.concatenate((A, np.ones((A.shape[0], 1))), 1)
        A_fg = np.concatenate((A_fg, np.ones((A_fg.shape[0], 1))), 1)
    
    inf_shd = (A_fg @ coeffs)
    inf_shd = inf_shd.clip(0) + 0.2

    if viz:
        shd_viz = (A @ coeffs).reshape(nrm.shape[:2])
        shd_viz = shd_viz.clip(0) + 0.2
        return inf_shd, shd_viz


    return inf_shd

def compute_reshading(orig, msk, inv_shd, depth, normals, alb, coeffs, model):

    # expects no channel dim on msk, shd and depth
    if len(inv_shd.shape) == 3:
        inv_shd = inv_shd[:, :, 0]

    if len(msk.shape) == 3:
        msk = msk[:, :, 0]

    if len(depth.shape) == 3:
        depth = depth[:, :, 0]

    h, w, _ = orig.shape

    # max_dim = max(h, w)
    # if max_dim > 1024:
    #     scale = 1024 / max_dim
    # else:
    #     scale = 1.0

    orig = resize(orig, (round_32(h), round_32(w)))
    alb = resize(alb, (round_32(h), round_32(w)))
    msk = resize(msk, (round_32(h), round_32(w)))
    inv_shd = resize(inv_shd, (round_32(h), round_32(w)))
    dpt = resize(depth, (round_32(h), round_32(w)))
    nrm = resize(normals, (round_32(h), round_32(w)))
    msk = msk.astype(np.single)

    hard_msk = (msk > 0.5)

    reg_shd = uninvert(inv_shd)
    img = (alb * reg_shd[:, :, None]).clip(0, 1)

    orig_alb = orig / reg_shd[:, :, None].clip(1e-4)
    
    bad_shd_np = reg_shd.copy()
    inf_shd = generate_shd(nrm, coeffs, hard_msk)
    bad_shd_np[hard_msk == 1] = inf_shd

    bad_img_np = alb * bad_shd_np[:, :, None]

    sem_msk = torch.from_numpy(msk).unsqueeze(0)
    bad_img = torch.from_numpy(bad_img_np).permute(2, 0, 1)
    bad_shd = torch.from_numpy(invert(bad_shd_np)).unsqueeze(0)
    in_nrm = torch.from_numpy(nrm).permute(2, 0, 1)
    in_dpt = torch.from_numpy(dpt).unsqueeze(0)
    # inp = torch.cat((sem_msk, bad_img, bad_shd), dim=0).unsqueeze(0)
    inp = torch.cat((sem_msk, bad_img, bad_shd, in_nrm, in_dpt), dim=0).unsqueeze(0)
    inp = inp.to(device)
    
    with torch.no_grad():
        out = model(inp).squeeze()

    fin_shd = out.detach().cpu().numpy()
    fin_shd = uninvert(fin_shd)
    fin_img = alb * fin_shd[:, :, None]

    normals = resize(nrm, (h, w))
    fin_shd = resize(fin_shd, (h, w))
    fin_img = resize(fin_img, (h, w))
    bad_shd_np = resize(bad_shd_np, (h, w))

    result = {}
    result['reshading'] = fin_shd
    result['init_shading'] = bad_shd_np
    result['composite'] = (fin_img ** (1/2.2)).clip(0, 1)
    result['normals'] = normals

    return result
