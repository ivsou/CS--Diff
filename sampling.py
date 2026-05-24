import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop


# Adapted from: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond,seq, model, b,ii,jj,osize,eta=0.):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            ii = ii.to(x.device)
            jj = jj.to(x.device)
            osize = osize.to(x.device)
            et = model(torch.cat([x_cond, xt], dim=1), t,ii,jj,osize)

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def generalized_steps_overlapping(x, x_cond, seq, model, b,ii,jj,osize, eta=0., corners=None, p_size=None,corners1=None,corners2=None,manual_batching=True):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        x_grid_mask = torch.zeros(x_cond.size(0),3,x_cond.size(2),x_cond.size(3), device=x.device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1
        if corners1!=None:
            p_size1=96
            x_grid_mask1 = torch.zeros(x_cond.size(0),3,x_cond.size(2),x_cond.size(3), device=x.device)
            for (hi, wi) in corners1:
                x_grid_mask1[:, :, hi:hi + p_size1, wi:wi + p_size1] += 1
        if corners2 != None:
            p_size2=128
            x_grid_mask2 = torch.zeros(x_cond.size(0),3,x_cond.size(2),x_cond.size(3), device=x.device)
            for (hi, wi) in corners2:
                x_grid_mask2[:, :, hi:hi + p_size2, wi:wi + p_size2] += 1


        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et_output = torch.zeros(x_cond.size(0),3,x_cond.size(2),x_cond.size(3), device=x.device)
            
            if manual_batching==True:
                # manual_batching_size should remain small (for example, 8 or 16), not p_size.
                manual_batching_size = min(16, max(1, len(corners)))
                # x_cond is expected to already be in the model input range
                # (for example, transformed to [-1, 1]) by the caller. Do not
                # re-apply data_transform here or the condition will be
                # transformed twice and produce scale errors.
                xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
                for s in range(0, len(corners), manual_batching_size):
                    e = min(len(corners), s + manual_batching_size)
                    # build batch-level ii/jj/osize
                    ii_input = ii[s:e].to(x.device)
                    jj_input = jj[s:e].to(x.device)
                    osize_input = osize[s:e].to(x.device)
                    x_input = torch.cat([x_cond_patch[s:e], xt_patch[s:e]], dim=1)
                    outputs = model(x_input, t, ii_input, jj_input, osize_input)
                    for idx, (hi, wi) in enumerate(corners[s:e]):
                        et_output[0, :, hi:hi + p_size, wi:wi + p_size] += outputs[idx]

                if corners1 != None:
                    et_output1 = torch.zeros(x_cond.size(0), 3, x_cond.size(2), x_cond.size(3), device=x.device)
                    manual_batching_size = min(16, max(1, len(corners1)))
                    xt_patch = torch.cat([crop(xt, hi, wi, p_size1, p_size1) for (hi, wi) in corners1], dim=0)
                    # See note above: do not re-apply data_transform here.
                    x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size1, p_size1) for (hi, wi) in corners1], dim=0)
                    for s in range(0, len(corners1), manual_batching_size):
                        e = min(len(corners1), s + manual_batching_size)
                        ii_input = ii[s:e].to(x.device)
                        jj_input = jj[s:e].to(x.device)
                        osize_input = osize[s:e].to(x.device)
                        x_input = torch.cat([x_cond_patch[s:e], xt_patch[s:e]], dim=1)
                        outputs1 = model(x_input, t, ii_input, jj_input, osize_input)
                        for idx, (hi, wi) in enumerate(corners1[s:e]):
                            et_output1[0, :, hi:hi + p_size1, wi:wi + p_size1] += outputs1[idx]
                if corners2 != None:
                    et_output2= torch.zeros(x_cond.size(0), 3, x_cond.size(2), x_cond.size(3), device=x.device)
                    manual_batching_size = min(16, max(1, len(corners2)))
                    xt_patch = torch.cat([crop(xt, hi, wi, p_size2, p_size2) for (hi, wi) in corners2], dim=0)
                    x_cond_patch = torch.cat([crop(x_cond, hi, wi, p_size2, p_size2) for (hi, wi) in corners2], dim=0)
                    for s in range(0, len(corners2), manual_batching_size):
                        e = min(len(corners2), s + manual_batching_size)
                        ii_input = ii[s:e].to(x.device)
                        jj_input = jj[s:e].to(x.device)
                        osize_input = osize[s:e].to(x.device)
                        x_input = torch.cat([x_cond_patch[s:e], xt_patch[s:e]], dim=1)
                        outputs2 = model(x_input, t, ii_input, jj_input, osize_input)
                        for idx, (hi, wi) in enumerate(corners2[s:e]):
                            et_output2[0, :, hi:hi + p_size2, wi:wi + p_size2] += outputs2[idx]

            else:
                for (hi, wi) in corners:
                    xt_patch = crop(xt, hi, wi, p_size, p_size)
                    x_cond_patch = crop(x_cond, hi, wi, p_size, p_size)
                    # Safety: do not blindly re-apply data_transform here.
                    # If the caller provided raw [0, 1] images, convert to [-1, 1].
                    # If the caller already provided [-1, 1], skip conversion.
                    try:
                        mn = float(x_cond_patch.min())
                        mx = float(x_cond_patch.max())
                    except Exception:
                        mn, mx = None, None
                    if mn is not None and mn >= -0.05 and mx <= 1.05:
                        # Values appear to be in [0, 1]; convert them.
                        x_cond_patch = data_transform(x_cond_patch)
                    else:
                        # Assume the tensor is already in the model input range.
                        pass
                    et_output[:, :, hi:hi + p_size, wi:wi + p_size] += model(torch.cat([x_cond_patch, xt_patch], dim=1), t)

            # Avoid division by zero when averaging overlapping patches.
            # Only divide where the mask is positive.
            denom = x_grid_mask.clone()
            denom[denom == 0] = 1.0
            et0 = et_output / denom

            if corners1 != None:
                denom1 = x_grid_mask1.clone()
                denom1[denom1 == 0] = 1.0
                et1 = et_output1 / denom1
                et = (et0 + et1) / 2.0
                if corners2 != None:
                    denom2 = x_grid_mask2.clone()
                    denom2[denom2 == 0] = 1.0
                    et2 = et_output2 / denom2
                    et = (et0 + et1 + et2) / 3.0
            else:
                et = et0
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
            
                
            
            
            
    return xs, x0_preds
