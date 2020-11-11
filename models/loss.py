# coding=utf8
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import permutations
import librosa

EPS = 1e-8


def rank_feas(raw_tgt, feas_list, out_type='torch'):
    final_num = []
    for each_feas, each_line in zip(feas_list, raw_tgt):
        for spk in each_line:
            final_num.append(each_feas[spk])
    if out_type == 'numpy':
        return np.array(final_num)
    else:
        return torch.from_numpy(np.array(final_num))


def criterion(tgt_vocab_size, use_cuda, loss):
    weight = torch.ones(tgt_vocab_size)
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


def compute_score(hiddens, targets, metric_fc, score_fc='arc_margin'):
    if score_fc in ['add_margin', 'arc_margin', 'sphere']:
        scores = metric_fc(hiddens, targets)
    elif score_fc == 'linear':
        scores = metric_fc(hiddens)
    else:
        raise ValueError(
            "score_fc should be in ['add_margin', 'arc_margin', 'sphere' and 'linear']")
    return scores


def cross_entropy_loss(hidden_outputs, targets, criterion, metric_fc, score_fc):
    targets = torch.tensor(targets).cuda().view(-1)
    scores = compute_score(hidden_outputs, targets, metric_fc, score_fc)
    loss = criterion(scores, targets)
    pred = scores.max(1)[1]
    num_correct = pred.eq(targets).sum()
    num_total = targets.size()[0]
    loss = loss.div(num_total)
    return loss, torch.tensor(num_total).cuda(), torch.tensor(num_correct).cuda()


def ss_loss(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func, wav_loss):
    predict_multi_map = multi_mask * x_input_map_multi
    y_multi_map = Variable(y_multi_map)
    loss_multi_speech = loss_multi_func(predict_multi_map, y_multi_map)
    return loss_multi_speech


def ss_tas_loss(aim_wav, predicted, mix_length):
    loss = cal_loss_with_order(aim_wav, predicted, mix_length)[0]
    return loss


def cal_loss_with_order(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr = cal_si_snr_with_order(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss,


def cal_loss_with_PIT(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(
        estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    print(pair_wise_si_snr)

    return torch.sum(pair_wise_si_snr, dim=1)/C


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    # perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    perms_one_hot = source.new_zeros(
        (perms.size()[0], perms.size()[1], C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    # B, C, *_ = source.size()
    B, C, __ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


def ss_loss_MLMSE(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func, Var):
    try:
        if Var == None:
            Var = Variable(torch.eye(
                config.fre_size, config.fre_size).cuda(), requires_grad=0)
            print('Set Var to:', Var)
    except:
        pass
    assert Var.size() == (config.fre_size, config.fre_size)

    predict_multi_map = torch.mean(
        multi_mask * x_input_map_multi, -2)
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map = torch.mean(Variable(y_multi_map), -2)

    loss_vector = (y_multi_map - predict_multi_map).view(-1,
                                                         config.fre_size).unsqueeze(1)

    Var_inverse = torch.inverse(Var)
    Var_inverse = Var_inverse.unsqueeze(0).expand(loss_vector.size()[0], config.fre_size,
                                                  config.fre_size)
    loss_multi_speech = torch.bmm(
        torch.bmm(loss_vector, Var_inverse), loss_vector.transpose(1, 2))
    loss_multi_speech = torch.mean(loss_multi_speech, 0)

    y_sum_map = Variable(torch.ones(
        config.batch_size, config.frame_num, config.fre_size)).cuda()
    predict_sum_map = torch.sum(multi_mask, 1)
    loss_multi_sum_speech = loss_multi_func(predict_sum_map, y_sum_map)
    print('loss 1 eval, losssum eval : ', loss_multi_speech.data.cpu(
    ).numpy(), loss_multi_sum_speech.data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print('evaling multi-abs norm this eval batch:',
          torch.abs(y_multi_map - predict_multi_map).norm().data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print('loss for whole separation part:',
          loss_multi_speech.data.cpu().numpy())
    # return F.relu(loss_multi_speech)
    return loss_multi_speech


def dis_loss(config, top_k_num, dis_model, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func):
    predict_multi_map = multi_mask * x_input_map_multi
    y_multi_map = Variable(y_multi_map).cuda()
    score_true = dis_model(y_multi_map)
    score_false = dis_model(predict_multi_map)
    acc_true = torch.sum(score_true > 0.5).data.cpu(
    ).numpy() / float(score_true.size()[0])
    acc_false = torch.sum(score_false < 0.5).data.cpu(
    ).numpy() / float(score_true.size()[0])
    acc_dis = (acc_false + acc_true) / 2
    print('acc for dis:(ture,false,aver)', acc_true, acc_false, acc_dis)

    loss_dis_true = loss_multi_func(score_true, Variable(
        torch.ones(config.batch_size * top_k_num, 1)).cuda())
    loss_dis_false = loss_multi_func(score_false, Variable(
        torch.zeros(config.batch_size * top_k_num, 1)).cuda())
    loss_dis = loss_dis_true + loss_dis_false
    print('loss for dis:(ture,false)', loss_dis_true.data.cpu().numpy(),
          loss_dis_false.data.cpu().numpy())
    return loss_dis
