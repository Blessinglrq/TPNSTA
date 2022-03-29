import argparse
import os
import torch
import numpy as np
from scipy import interpolate
import random

from attrdict import AttrDict

from tpnsta.data.loader import data_loader, data_loader_test, data_loader_TPN_test
from tpnsta.models import TrajectoryGenerator, MergeModelConv, TrajectoryGeneratorMultiMergeSlinearAveF
from tpnsta.losses import displacement_error, final_displacement_error
from tpnsta.utils import relative_to_abs, get_dset_path


parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', default='../models/tpnsta-p-models/zara1_12_model.pt', type=str)  # FIXME (tpnsta-models/tpnsta-p-models)(eth_8_model/eth_12_model/hotel/univ/zara1/zara2)
parser.add_argument('--model_path', default='../models/TPNSTA/TPNSTA/univ_TPNSTA_with_model.pt', type=str)
parser.add_argument('--num_samples', default=20, type=int)  # FIXME default=20
parser.add_argument('--seed', help='manual seed to use, default is 321',
                    type=int, default=321)
parser.add_argument('--dset_type', default='test', type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    # generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.train()
    return generator


def get_generator_merge(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGeneratorMultiMergeSlinearAveF(
        obs_len=args.obs_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    # generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.train()
    return generator

def get_merge_model(checkpoint):
    args = AttrDict(checkpoint['args'])
    merge_model = MergeModelConv()
    merge_model.load_state_dict(checkpoint['merge_state'])
    # generator.load_state_dict(checkpoint['g_best_state'])
    merge_model.cuda()
    merge_model.train()
    return merge_model


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def interpolation(batch, inter_scale=0.5):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch  # loss_mask 表示有人的位置都为1
    traj_pos = torch.cat((obs_traj, pred_traj_gt), dim=0)
    traj_pos_new = torch.FloatTensor(int(traj_pos.shape[0]*inter_scale), traj_pos.shape[1], traj_pos.shape[2]).cuda()
    if inter_scale <= 1:
        traj_pos_new = traj_pos[[int(1/inter_scale) * i for i in range(int(traj_pos.shape[0]*inter_scale))]]
    else:
        traj_pos_new[[int(inter_scale) * i for i in range(int(traj_pos.shape[0]))]] = traj_pos
        last_step = (traj_pos[-1] + (traj_pos[-1] - traj_pos[-2])).unsqueeze(0)
        traj_pos = torch.cat((traj_pos, last_step), dim=0)
        if inter_scale == 2:
            traj_pos_new[[int(inter_scale) * i + 1 for i in range(int(traj_pos.shape[0]-1))]] = (traj_pos[:-1] + traj_pos[1:])/inter_scale
        elif inter_scale == 4:
            traj_pos_new[[int(inter_scale) * i + 1 for i in range(int(traj_pos.shape[0] - 1))]] = (traj_pos[1:] - traj_pos[:-1]) * (1/inter_scale) + traj_pos[:-1]
            traj_pos_new[[int(inter_scale) * i + 2 for i in range(int(traj_pos.shape[0] - 1))]] = (traj_pos[1:] - traj_pos[:-1]) * (2/inter_scale) + traj_pos[:-1]
            traj_pos_new[[int(inter_scale) * i + 3 for i in range(int(traj_pos.shape[0] - 1))]] = (traj_pos[1:] - traj_pos[:-1]) * (3/inter_scale) + traj_pos[:-1]
    # print(traj_pos_new)
    obs_traj_new = traj_pos_new[:int(obs_traj.shape[0]*inter_scale)]
    pred_traj_gt_new = traj_pos_new[int(obs_traj.shape[0]*inter_scale):]
    rel_curr_ped_seq = torch.zeros(traj_pos_new.shape)
    rel_curr_ped_seq[1:] = \
        traj_pos_new[1:] - traj_pos_new[:-1]  # 坐标位置相减得到速度，默认第一个位置的速度为0
    obs_traj_rel_new = rel_curr_ped_seq[:int(obs_traj.shape[0]*inter_scale)]
    pred_traj_gt_rel = rel_curr_ped_seq[int(obs_traj.shape[0]*inter_scale):]
    out = [obs_traj_new, pred_traj_gt_new, obs_traj_rel_new, pred_traj_gt_rel, non_linear_ped, loss_mask, seq_start_end]
    return out


def interpolation_for_pred(pred_traj_rel, inter_scale=0.5):
    pred_traj_pos_new = torch.FloatTensor(int(pred_traj_rel.shape[0] * inter_scale), pred_traj_rel.shape[1], pred_traj_rel.shape[2]).cuda()
    if inter_scale == 2:
        pred_traj_pos_new[[int(inter_scale) * i for i in range(int(pred_traj_rel.shape[0]))]] = pred_traj_rel
        last_step = (pred_traj_rel[-1] + (pred_traj_rel[-1] - pred_traj_rel[-2])).unsqueeze(0)
        pred_traj_rel = torch.cat((pred_traj_rel, last_step), dim=0)
        pred_traj_pos_new[[int(inter_scale) * i + 1 for i in range(int(pred_traj_rel.shape[0] - 1))]] = (pred_traj_rel[:-1] + pred_traj_rel[1:]) / inter_scale
    elif inter_scale <= 1:
        pred_traj_pos_new = pred_traj_rel[[int(1 / inter_scale) * i for i in range(int(pred_traj_rel.shape[0] * inter_scale))]]
    return pred_traj_pos_new


def new_interpolation(batch, inter_scale=0.5):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch  # loss_mask 表示有人的位置都为1
    traj_pos = torch.cat((obs_traj, pred_traj_gt), dim=0)
    traj_pos_new = torch.FloatTensor(int(traj_pos.shape[0]*inter_scale), traj_pos.shape[1], traj_pos.shape[2]).numpy()
    traj_pos_rel = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0).cpu().numpy()
    if inter_scale <= 1:
        traj_pos = traj_pos.cpu().numpy()
        traj_pos_new = traj_pos[[int(1/inter_scale) * i for i in range(int(traj_pos.shape[0]*inter_scale))]]
    else:
        last_step = (traj_pos[-1] + (traj_pos[-1] - traj_pos[-2])).unsqueeze(0)
        traj_pos = torch.cat((traj_pos, last_step), dim=0)
        timestep = torch.linspace(1, traj_pos.shape[0], steps=traj_pos.shape[0], out=None).numpy()
        new_timestep = torch.linspace(1, traj_pos.shape[0], steps=(traj_pos_new.shape[0]+1), out=None).numpy()
        for i in range(traj_pos_rel.shape[1]):
            f_x = interpolate.interp1d(timestep, traj_pos[:, :, 0][:, i], kind='cubic')  # Fixme ['linear','zero', 'slinear', 'quadratic', 'cubic', 4, 5]
            traj_pos_new[:, i, 0] = f_x(new_timestep)[:-1]
            f_y = interpolate.interp1d(timestep, traj_pos[:, :, 1][:, i], kind='cubic')
            traj_pos_new[:, i, 1] = f_y(new_timestep)[:-1]
            #  FIXME plot 3D
            # from mpl_toolkits.mplot3d import axes3d
            # import matplotlib.pyplot as plt
            # fig = plt.figure(1)
            # ax = fig.gca(projection='3d')
            # figure = ax.plot(traj_pos_new[:, 0, 0], traj_pos_new[:, 0, 1], new_timestep[:-1], 'ro', c='r')
            # plt.show()
    obs_traj_new = traj_pos_new[:int(obs_traj.shape[0] * inter_scale)]
    pred_traj_gt_new = traj_pos_new[int(obs_traj.shape[0] * inter_scale):]
    traj_pos_rel_new = torch.zeros(traj_pos_new.shape).numpy()
    traj_pos_rel_new[1:] = \
        traj_pos_new[1:] - traj_pos_new[:-1]  # FIXME 坐标位置相减得到速度，默认第一个位置的速度为原始的速度
    # traj_pos_rel_new[0] = obs_traj_rel[0]
    obs_traj_new = torch.from_numpy(obs_traj_new).cuda()
    pred_traj_gt_new = torch.from_numpy(pred_traj_gt_new).cuda()
    traj_pos_rel_new = torch.from_numpy(traj_pos_rel_new).cuda()
    obs_traj_rel_new = traj_pos_rel_new[:int(obs_traj.shape[0] * inter_scale)]
    pred_traj_gt_rel = traj_pos_rel_new[int(obs_traj.shape[0] * inter_scale):]
    out = [obs_traj_new, pred_traj_gt_new, obs_traj_rel_new, pred_traj_gt_rel, non_linear_ped, loss_mask,
           seq_start_end]
    return out


def new_interpolation_for_pred(pred_traj, inter_scale=0.5):
    pred_traj_pos_new = torch.FloatTensor(int(pred_traj.shape[0] * inter_scale), pred_traj.shape[1], pred_traj.shape[2]).cuda()
    if inter_scale == 2:
        last_step = (pred_traj[-1] + (pred_traj[-1] - pred_traj[-2])).unsqueeze(0)
        pred_traj = torch.cat((pred_traj, last_step), dim=0)
        pred_traj_copy = pred_traj.clone().detach()
        timestep = torch.linspace(1, pred_traj.shape[0], steps=pred_traj.shape[0], out=None).numpy()
        new_timestep = torch.linspace(1, pred_traj.shape[0], steps=(pred_traj_pos_new.shape[0] + 1), out=None).numpy()
        for i in range(pred_traj.shape[1]):
            f_x = interpolate.interp1d(timestep, pred_traj_copy[:, :, 0][:, i],
                                       kind='cubic')  # Fixme ['linear','zero', 'slinear', 'quadratic', 'cubic', 4, 5]
            pred_traj_pos_new[:, i, 0] = torch.from_numpy(f_x(new_timestep)[:-1]).cuda()
            f_y = interpolate.interp1d(timestep, pred_traj_copy[:, :, 1][:, i], kind='cubic')
            pred_traj_pos_new[:, i, 1] = torch.from_numpy(f_y(new_timestep)[:-1]).cuda()
    elif inter_scale <= 1:
        pred_traj_pos_new = pred_traj[[int(1 / inter_scale) * i for i in range(int(pred_traj.shape[0] * inter_scale))]]
    return pred_traj_pos_new


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        for batches in loader:
            (batch, batch_4, batch_8, batch_16, batch_32, mean_sta) = batches
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
             loss_mask, seq_start_end) = batch
            (traj_mean_x_8, traj_std_x_8, traj_mean_y_8, traj_std_y_8,
             traj_rel_mean_x_8, traj_rel_std_x_8, traj_rel_mean_y_8, traj_rel_std_y_8) = mean_sta
            obs_traj = obs_traj.cuda()
            pred_traj_gt = pred_traj_gt.cuda()
            obs_traj_rel = obs_traj_rel.cuda()
            pred_traj_gt_rel = pred_traj_gt_rel.cuda()
            non_linear_ped = non_linear_ped.cuda()
            seq_start_end = seq_start_end.cuda()
            batch_4 = [tensor.cuda() for tensor in batch_4]
            batch_8 = [tensor.cuda() for tensor in batch_8]
            batch_16 = [tensor.cuda() for tensor in batch_16]
            batch_32 = [tensor.cuda() for tensor in batch_32]
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            (obs_traj_4, pred_traj_gt_4, obs_traj_rel_4, pred_traj_gt_rel_4, non_linear_ped,
             seq_start_end) = batch_4
            (obs_traj_8, pred_traj_gt_8, obs_traj_rel_8, pred_traj_gt_rel_8, non_linear_ped,
             seq_start_end) = batch_8
            (obs_traj_16, pred_traj_gt_16, obs_traj_rel_16, pred_traj_gt_rel_16, non_linear_ped,
             seq_start_end) = batch_16
            (obs_traj_32, pred_traj_gt_32, obs_traj_rel_32, pred_traj_gt_rel_32, non_linear_ped,
             seq_start_end) = batch_32

            for _ in range(num_samples):

                generator_out_4, generator_out_8, generator_out_16, generator_out_32, generator_out_final = generator(
                    obs_traj_4, obs_traj_rel_4, obs_traj_8, obs_traj_rel_8, obs_traj_16, obs_traj_rel_16, obs_traj_32,
                    obs_traj_rel_32, seq_start_end, pred_traj_gt_4.shape[0])
                pred_traj_fake_rel_final = generator_out_final

                obsv_v = torch.sqrt(torch.sum(torch.pow(obs_traj_rel, 2), dim=2))
            #     # FIXME change the output when the history is stopping.
                history_error = torch.sum(obsv_v, dim=0) / 8
                pred_traj_fake_rel_final[:, history_error < 0.05] = 0.0   # FIXME setting the threshold

                # FIXME unstandardize
                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel_final, obs_traj_8[-1]
                )
                unstandar_pred_traj_fake = unstandardize(pred_traj_fake, traj_mean_x_8,
                                                             traj_std_x_8, traj_mean_y_8,
                                                             traj_std_y_8)

                ade.append(displacement_error(
                    unstandar_pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    unstandar_pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)
        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde

def unstandardize(data, data_mean_x, data_std_x, data_mean_y, data_std_y):
    data_copy = np.copy(data.cpu())
    data_copy[:, :, 0] = data_copy[:, :, 0] * data_std_x + data_mean_x
    data_copy[:, :, 1] = data_copy[:, :, 1] * data_std_y + data_mean_y
    data_copy = torch.FloatTensor(data_copy).cuda()
    return data_copy


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path)
        # generator = get_generator(checkpoint)
        generator = get_generator_merge(checkpoint)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, args.dset_type)
        _, loader = data_loader_TPN_test(_args, path)  # FIXME data_loader_TPN_test for Spline Offline
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))


if __name__ == '__main__':
    main(args)
