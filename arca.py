from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from losses import log_prob_loss, log_perplexity
from utils import get_forbidden_toks, filter_forbidden_toks, get_unigram_probs

np.random.seed(42)


def run_arca(args, model, tokenizer, embedding_table, output_str=None):
    # (Fixed output is used in the reverse case)
    fixed_output = (output_str is not None)
    run_metadata = {}
    args.batch_size = args.arca_batch_size

    vocab_size = embedding_table.shape[0]  # V
    embedding_dim = embedding_table.shape[1]  # d
    
    # (Avoid degenerate solutions + additional constraints specified in args)
    forbidden_input_toks = get_forbidden_toks(args, 
                                              tokenizer, 
                                              n_total_toks=embedding_table.shape[0],  # V
                                              output=False, 
                                              output_str=output_str)  # list of tokens to avoid in x
    
    if not fixed_output:
        forbidden_output_toks = get_forbidden_toks(args, 
                                                   tokenizer, 
                                                   n_total_toks=embedding_table.shape[0], 
                                                   output=True, 
                                                   output_str=output_str)  # empty set
        
    # (Whether or not to use a fixed prompt prefix)
    use_prefix = (args.prompt_prefix is not None)
    if use_prefix:
        prefix_toks = torch.Tensor(tokenizer(args.prompt_prefix)['input_ids']
                                   ).long().to(args.device)  # L_prefix
        prefix_embeddings = embedding_table[prefix_toks].unsqueeze(0)  # 1 x L_prefix x d
        prefix_embeddings = prefix_embeddings.repeat(args.batch_size, 1, 1).detach()  # B x L_prefix x d
        prefix_length = prefix_embeddings.shape[1]  # L_prefix

    if fixed_output:  # True for reversal
        output_toks = np.array(tokenizer(output_str)['input_ids']
                              )  # L_output
        args.output_length = output_toks.shape[0]  # L_output
        run_metadata['n_output_toks'] = args.output_length
        assert args.unigram_output_constraint is None

    curr_toks = np.random.choice(vocab_size, 
                                 size=args.prompt_length + args.output_length,
                                 replace=True)  # L_attack + L_output
    if fixed_output:  # True for reversal
        curr_toks[args.prompt_length:] = output_toks
    if use_prefix:
        curr_toks = np.concatenate([prefix_toks.detach().cpu().numpy(), curr_toks], axis=0)  # L_prefix + L_attack + L_output

    stacked_curr_toks = np.tile(curr_toks, (args.batch_size, 1))  # B x (L_prefix + L_attack + L_output)
    stacked_curr_toks = torch.Tensor(stacked_curr_toks).long().to(args.device)

    if args.unigram_output_constraint is not None:  # "toxic" for joint optimization
        output_unigram_lps = get_unigram_probs(args.unigram_output_constraint, 
                                               device=args.device, 
                                               gptj=(args.model_id == 'gptj'))  # V, unigram probs that words are (toxic/etc.)
        
    if args.unigram_input_constraint is not None:  # "not_toxic" for joint optimization
        input_unigram_lps = get_unigram_probs(args.unigram_input_constraint, 
                                              device=args.device, 
                                              gptj=(args.model_id == 'gptj'))  # V, unigram probs that words are NOT (toxic, etc.)

    output_start = args.prompt_length + prefix_length if use_prefix else args.prompt_length  # (L_prefix + L_attack) or (L_attack)

    # MINE
    L_prefix = prefix_length if use_prefix else 0
    L_attack = args.prompt_length
    L_out = args.output_length

    # (Initialize full embeddings)
    full_embeddings = torch.zeros(args.batch_size, 
                                  args.prompt_length + args.output_length, 
                                  embedding_dim).to(args.device)  # B x (L_attack + L_output) x d
    for i in range(args.prompt_length + args.output_length):
        rel_idx = (i + prefix_length) if use_prefix else i  # (L_prefix + i)
        full_embeddings[:, i, :] = embedding_table[curr_toks[rel_idx]].unsqueeze(0  # 1 x d
                                                                              ).repeat(args.batch_size, 1)  # B x d

    # (Iterate)
    for it in tqdm(range(args.arca_iters)):  # number of iters to optimize ENTIRE attack
        for tok_idx in range(args.prompt_length + args.output_length):  # cycle through positions: L_attack + L_output
            tok_in_output = (tok_idx >= args.prompt_length)
            tok_in_attack = not tok_in_output
            # (Output tokens remain fixed in the reversing case)
            if tok_in_output and fixed_output:
                continue

            update_idx = (tok_idx + prefix_length) if use_prefix else tok_idx  # (L_prefix + tok_idx)

            # # THEIRS. Sampling with replacement
            # random_vocab_indices = np.random.choice(vocab_size, 
            #                                size=args.batch_size, 
            #                                replace=True)  # to avg gradients over B random tokens (same as K in paper?)
            # MINE. Sampling without replacement
            random_vocab_indices = np.random.choice(vocab_size, 
                                           size=args.batch_size, 
                                           replace=False)  # to avg gradients over B random tokens (same as K in paper?)
            
            if args.autoprompt:
                random_vocab_indices = curr_toks[update_idx].repeat(args.batch_size)  # doesn't use random tokens, so just repeat the same one B times

            # redundant if running AutoPrompt since random_vocab_indices is just the current token repeated            
            full_embeddings[:, tok_idx, :] = embedding_table[random_vocab_indices, :]  # B x d
            if args.model_id == 'gptj':
                full_embeddings = full_embeddings.half()

            # # NOT REQUIRED, no need to populate targets/labels with the random tokens as well
            # stacked_curr_toks[:, update_idx] = random_vocab_indices  # B, random tokens
            # stacked_curr_toks[:, update_idx] = torch.Tensor(random_vocab_indices).long().to(args.device)  # B, random tokens

            # Losses
            if use_prefix:
                labels = torch.cat([-100*torch.ones(args.prompt_length + prefix_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1),  # B x (L_prefix + L_attack), all -100
                                    stacked_curr_toks[:, args.prompt_length + prefix_length:]  # B x L_output
                                    ], 
                                dim=1).long()  # B x (L_prefix + L_attack + L_output)
                                               # @ output positions, curr_toks is repeated B times.
            else:
                labels = torch.cat([-100 * torch.ones(args.prompt_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1),  # B x L_attack, all -100
                                    stacked_curr_toks[:, args.prompt_length:]
                                    ], 
                                dim=1).long()  # B x (L_attack + L_output)
                                               # @ output positions, curr_toks is repeated B times.

            full_embeddings = full_embeddings.detach()  # requires_grad => False, grad => None
            if full_embeddings.requires_grad:  # not called!
                full_embeddings.grad.zero_()
            full_embeddings.requires_grad = True  # grad will be populated during backward()
            full_embeddings.retain_grad()  # grad => still None
            
            if use_prefix:
                out = model(inputs_embeds=torch.cat([prefix_embeddings,  # requires_grad = False
                                                     full_embeddings],  # requires_grad = True
                                                dim=1),  # B x (L_prefix + L_attack + L_output) x d
                            labels=labels  # B x (L_prefix + L_attack + L_output)
                            )  # CausalLMOutputWithCrossAttentions object
            else:
                out = model(inputs_embeds=full_embeddings,  # B x (L_attack + L_output) x d
                            labels=labels  # B x (L_attack + L_output) x d
                            )  # CausalLMOutputWithCrossAttentions object
            
            # 1a. for ln p(o>i|oi, o<i + x) OR ln p(o|x<i, xi, x>i) loss
            loss = log_prob_loss(out, labels, temp=1) * L_out  # scalar (averaged over B, summed over L_output --> MINE)

            # 2a. for ln p(x>i|xi, x<i) loss
            if args.lam_perp > 0:
                perp_loss = log_perplexity(out, 
                                           stacked_curr_toks[:, :output_start]  # B x (L_prefix + L_attack)
                                           ) * (L_prefix + L_attack - 1)  # scalar (averaged over B, summed over (L_prefix + L_attack - 1) --> MINE). ALSO INCLUDES PERPLEXITY OF PREFIX
                loss += args.lam_perp * perp_loss
            
            loss.backward(retain_graph=True)  # computes gradients and allows subsequent .backward() calls
            grad = full_embeddings.grad  # B x (L_attack + L_output) x d
            
            # # Theirs. DO sum(dim=0) INSTEAD? ALREADY DIVIDED BY B WHEN CALCULATING LOSSES ????????????????????????
            # backward_scores = - torch.matmul(embedding_table,  # V x d
            #                                  grad[:, tok_idx, :].mean(dim=0)  # B x d => d
            #                                  )  # V scores, higher is better
            # MINE
            backward_scores = - torch.matmul(embedding_table,  # V x d
                                             grad[:, tok_idx, :].sum(dim=0)  # B x d => d
                                             )  # V scores, higher is better

            # 1b. for ln p(oi|o<i + x) term
            if tok_in_output and not args.autoprompt:
                forward_log_probs = F.log_softmax(out.logits[0, update_idx-1, :],  # B x (L_prefix + L_attack + L_output) x V => V
                                                  dim=0)  # V, ln p(oi|o<i + x)
                scores = backward_scores + forward_log_probs  # V, higher is better

                if args.unigram_output_constraint is not None:  # "toxic"
                    scores += args.unigram_weight * output_unigram_lps  # V, higher is better

            else:
                scores = backward_scores
                if args.unigram_input_constraint is not None:  # "not toxic"
                    scores += args.unigram_weight * input_unigram_lps  # V, higher is better

            # 2b. for ln perp(xi|x<i) term ???
            # ==================================== MINE ====================================
            if (args.lam_perp > 0) and tok_in_attack and (use_prefix or update_idx >= 1) and not args.autoprompt:
                forward_log_probs = F.log_softmax(out.logits[0, update_idx-1, :],  # B x (L_prefix + L_attack + L_output) x V => V
                                                    dim=0)  # V, ln p(oi|o<i + x)
                scores += args.lam_perp * forward_log_probs  # V, higher is better
            # ==============================================================================

            best_scores_idxs = scores.argsort(descending=True)  # V

            if tok_in_output:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_output_toks)
            else:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_input_toks)

            full_embeddings = full_embeddings.detach()

            # Calculate actual losses ?
            with torch.no_grad():
                top_B_toks = best_scores_idxs[:args.batch_size]  # B
                full_embeddings[:, tok_idx, :] = embedding_table[top_B_toks, :]  # B x d
                # MINE
                stacked_curr_toks[:, update_idx] = top_B_toks  # B
                # # THEIRS
                # stacked_curr_toks[:, tok_idx] = top_B_toks
                
                if use_prefix:
                    out = model(inputs_embeds=torch.cat([prefix_embeddings, 
                                                         full_embeddings], 
                                                    dim = 1)  # B x (L_prefix + L_attack + L_output) x d
                               )
                else:
                    out = model(inputs_embeds=full_embeddings  # B x (L_attack + L_output) x d
                               )

                # 1. exact ln p(o|x)
                log_probs = F.log_softmax(out.logits[:, (-1-L_out):-1, :],  # B x L_output x V
                                          dim=2)
                batch_log_probs = torch.stack(
                                        [log_probs[i, torch.arange(L_out), stacked_curr_toks[i,output_start:]].sum()  # scalar, summed over L_output
                                            for i in range(args.batch_size)])  # B
                # 2. exact ln p(x_>=1|prefix) or ln p(x_>1|x_1)
                if args.lam_perp > 0:
                    output_perps = log_perplexity(out, 
                                                  stacked_curr_toks[:, :output_start],  # B x (L_prefix + L_attack)
                                                  ret_all=True) * (L_prefix + L_attack - 1)  # B, summed over (L_prefix + L_attack-1) --> MINE
                    batch_log_probs -= args.lam_perp * output_perps
                

                if args.unigram_output_constraint is not None and tok_in_output:  # "toxic"
                    batch_log_probs += args.unigram_weight * output_unigram_lps[top_B_toks]  # B
                elif args.unigram_input_constraint is not None and not tok_in_output:  # "not toxic"
                    batch_log_probs += args.unigram_weight * input_unigram_lps[top_B_toks]  # B

                best_batch_idx = batch_log_probs.argmax()
                best_idx = top_B_toks[best_batch_idx]  # best replacement token

                assert best_idx == best_scores_idxs[best_batch_idx]
                
                curr_toks[update_idx] = best_idx.item()  # replace with best token
                stacked_curr_toks[:, update_idx] = best_idx.item()
                full_embeddings[:, tok_idx, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)  # B x d
                optimized_output = stacked_curr_toks[0][output_start:]  # L_output

                greedy_output = log_probs[best_batch_idx].argmax(dim=1)  # L_output

                # (Can modify success checks here to stop running the algorithm)
                output_matches = (optimized_output == greedy_output).all().item()
                if args.unigram_input_constraint is not None:  # "not toxic"
                    input_unigram_satisfied  = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()  # min over (L_prefix + L_attack). Why INCLUDE PREFIX TOKENS HERE ???????????
                                                         ).item() > 0.99
                else:
                    input_unigram_satisfied = True

                if not fixed_output and args.unigram_output_constraint is not None:  # "toxic"
                    output_unigram_satisfied = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()  # max over L_output
                                                         ).item() > 0.5
                else:
                    output_unigram_satisfied = True

                # (Success according to above checks)
                if output_matches and input_unigram_satisfied and output_unigram_satisfied:
                    if args.lam_perp > 0:
                        run_metadata['perplexity'] = output_perps[best_batch_idx].item()
                    if args.unigram_output_constraint is not None:
                        run_metadata['output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]]).mean().item()
                        run_metadata['max_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item()
                        run_metadata['min_output_unigram'] = torch.exp(output_unigram_lps[curr_toks[output_start:]].min()).item()
                    if args.unigram_input_constraint is not None:
                        run_metadata['input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]]).mean().item()
                        run_metadata['max_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].max()).item()
                        run_metadata['min_input_unigram'] = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item()
                    if fixed_output:
                        curr_toks = curr_toks[:-args.output_length]

                    prompt = tokenizer.decode(curr_toks)
                    print(f"Found string: {prompt}")
                    return curr_toks, it, run_metadata
                
    # (Failure case)
    if args.lam_perp > 0:
        run_metadata['perplexity'] = None
        if args.unigram_output_constraint is not None:
            run_metadata['output_unigram'] = -1
        elif args.unigram_input_constraint is not None:
            run_metadata['input_unigram'] = -1

    if fixed_output:
        curr_toks = curr_toks[:-args.output_length]

    prompt = tokenizer.decode(curr_toks)
    print(f"Failed to find a string: {prompt}")

    return -1, -1, run_metadata
