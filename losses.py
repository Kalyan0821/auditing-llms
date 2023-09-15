import torch
from torch import nn
import torch.nn.functional as F


def log_prob_loss(output, labels, temp=1):
    """ labels: B x (L_prefix + L_attack + L_output)
        The first B x (L_prefix + L_attack) are all -100.
        curr_toks in output positions repeated B times. If tok_idx is an output position, then just that position is replaced with B random tokens. """

    logits = output.logits  # B x (L_prefix + L_attack + L_output) x V
    if torch.isnan(logits).any():
        assert False

    shift_logits = logits[..., :-1, :].contiguous()  # B x (L_prefix + L_attack + L_output - 1) x V
                                                     # all but the last logit needs to be optimized

    shift_labels = labels[..., 1:].contiguous()  # B x (L_prefix + L_attack -1 + L_output)
                                                 # all but the first token needs to be predicted
                                                 # the first B x (L_prefix + L_attack -1) are all -100
    shift_logits = shift_logits / temp

    loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)  # will avg. over L_output positions, since -100 is ignored
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),  # B*(L-1) x V
                    shift_labels.view(-1)  # B*(L-1)
                    )
    return loss


def log_perplexity(output, prompts, ret_all=False):
    """ prompts: B x (L_prefix + L_attack)
        If tok_idx is an input position, then just that position is replaced with B random tokens. """
    
    shift_prompts = prompts[:, 1:]  # B x (L_prefix + L_attack - 1)
                                    # all but the first token needs to be predicted
    L = shift_prompts.shape[1]  # (L_prefix + L_attack - 1)

    shift_logits = output.logits[:, :L, :]  # B x (L_prefix + L_attack - 1) x V 
    log_probs = F.log_softmax(shift_logits, dim=2)  # B x (L_prefix + L_attack - 1) x V
    B = log_probs.shape[0]
    assert L == log_probs.shape[1]
    V = log_probs.shape[2]

    # Can do this or the stuff after the return    
    stacked_perplexities = torch.stack(
                                [log_probs[i, torch.arange(L), shift_prompts[i]].mean()  # mean cross entropy loss over (L_prefix + L_attack - 1) positions 
                                    for i in range(B)])
    if ret_all:
        return -stacked_perplexities  # B
    
    avg_loss = -stacked_perplexities.mean()  # mean over B
    return avg_loss

    loss_fct = nn.CrossEntropyLoss(reduction='mean')  # will avg. over (L_prefix + L_attack - 1) positions
    loss = loss_fct(shift_logits.reshape(-1, V),  # B*(L-1) x V
                    torch.tensor(shift_prompts.reshape(-1), device=shift_logits.device)  # B*(L-1)
                    )
    return loss
