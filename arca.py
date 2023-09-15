from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from losses import log_prob_loss, log_perplexity
from utils import get_forbidden_toks, filter_forbidden_toks, get_unigram_probs 


def run_arca(args, model, tokenizer, embedding_table, output_str=None):
    # (Fixed output is used in the reverse case)
    fixed_output = (output_str is not None)
    run_metadata = {}
    args.batch_size = args.arca_batch_size
    embedding_dim = embedding_table.shape[1]  # d (V x d)
    
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

    vocab_size = embedding_table.shape[0]  # V
    embedding_dim = embedding_table.shape[1]  # d

    if fixed_output:  # True for reversal
        output_toks = np.array(tokenizer(output_str)['input_ids']
                              )  # L_output
        # output_toks_tensor = torch.Tensor(tokenizer(output_str)['input_ids']
        #                                   ).long().to(args.device)  # L_output
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

    stacked_cur_toks = np.tile(curr_toks, (args.batch_size, 1))  # B x (L_prefix + L_attack + L_output)
    curr_toks_tensor = torch.Tensor(stacked_cur_toks).long().to(args.device)

    if args.unigram_output_constraint is not None:  # "toxic" for joint optimization
        output_unigram_lps = get_unigram_probs(args.unigram_output_constraint, 
                                               device=args.device, 
                                               gptj=(args.model_id == 'gptj'))  # V x 6, unigram probs that words are (toxic, etc.)
    if args.unigram_input_constraint is not None:  # "not_toxic" for joint optimization
        input_unigram_lps = get_unigram_probs(args.unigram_input_constraint, 
                                              device=args.device, 
                                              gptj=(args.model_id == 'gptj'))  # V x 6, unigram probs that words are NOT (toxic, etc.)

    output_start = args.prompt_length + prefix_length if use_prefix else args.prompt_length  # (L_prefix + L_attack) or (L_attack)

    # (Initialize full embeddings)
    full_embeddings = torch.zeros(args.batch_size, 
                                  args.prompt_length + args.output_length, 
                                  embedding_dim).to(args.device)  # B x (L_attack + L_output) x d
    for i in range(args.prompt_length + args.output_length):
        rel_idx = (i + prefix_length) if use_prefix else i  # (L_prefix + i)
        full_embeddings[:, i, :] = embedding_table[curr_toks[rel_idx]].unsqueeze(0  # 1 x d
                                                                              ).repeat(args.batch_size, 1)  # B x d

    # (Iterate)
    for it in tqdm(range(args.arca_iters)):  # number of iters to optimize entire attack

        for tok_idx in range(args.prompt_length + args.output_length):  # cycle through positions: L_attack + L_output
            tok_in_output = (tok_idx >= args.prompt_length)
            # (Output tokens remain fixed in the reversing case)
            if tok_in_output and fixed_output:
                continue

            update_idx = (tok_idx + prefix_length) if use_prefix else tok_idx  # (L_prefix + tok_idx)

            random_vocab_indices = np.random.choice(vocab_size, 
                                           size=args.batch_size, 
                                           replace=True)  # to avg gradients over B random tokens (same as K in paper?)            
            if args.autoprompt:
                random_vocab_indices = curr_toks[update_idx].repeat(args.batch_size)  # doesn't use random tokens, so just repeat the same one B times

            # redundant if running AutoPrompt since random_vocab_indices is just the current token repeated            
            full_embeddings[:, tok_idx, :] = embedding_table[random_vocab_indices, :]  # B x d
            if args.model_id == 'gptj':
                full_embeddings = full_embeddings.half()

            # (Update to compute likelihood and perplexity losses)
            stacked_cur_toks[:, update_idx] = random_vocab_indices  # B, random tokens
            curr_toks_tensor[:, update_idx] = torch.Tensor(random_vocab_indices).long().to(args.device)

            # for ln p(o|x) loss
            if use_prefix:
                labels = torch.cat([-100*torch.ones(args.prompt_length + prefix_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1),  # B x (L_prefix + L_attack), all -100
                                    curr_toks_tensor[:, args.prompt_length + prefix_length:]  # B x L_output
                                    ], 
                                dim=1).long()  # B x (L_prefix + L_attack + L_output)
                                                 # curr_toks in output positions repeated B times. If tok_idx is an
                                                 # output position, then just that position is replaced with B random tokens.
            else:
                labels = torch.cat([-100 * torch.ones(args.prompt_length).to(args.device).unsqueeze(0).repeat(args.batch_size, 1),  # B x L_attack, all -100
                                    curr_toks_tensor[:, args.prompt_length:]
                                    ], 
                                dim=1).long()  # B x (L_attack + L_output)

            full_embeddings = full_embeddings.detach()  # requires_grad => False, grad => None
            # not called!
            if full_embeddings.requires_grad:
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
                
            loss = log_prob_loss(out, labels, temp=1)  # scalar (averaged over B and L)

            # (Compute the perplexity loss)
            if args.lam_perp > 0:
                perp_loss = log_perplexity(out, 
                                           stacked_cur_toks[:, :output_start]  # B x (L_prefix + L_attack)
                                           )  # scalar (averaged over B and L)
                                              # ALSO INCLUDES PERPLEXITY OF PREFIX. WHY ???????????????????????????
                loss += args.lam_perp * perp_loss
            
            loss.backward(retain_graph=True)  # computes gradients and allows subsequent .backward() calls

            grad = full_embeddings.grad  # B x (L_attack + L_output) x d




            backward_scores = -torch.matmul(embedding_table, grad[:,tok_idx,:].mean(dim = 0))
            



            if tok_in_output and not args.autoprompt:
                forward_log_probs = F.log_softmax(out.logits[0, update_idx - 1, :], dim = 0)
                scores = backward_scores + forward_log_probs
                if args.unigram_output_constraint is not None:
                    scores += args.unigram_weight * output_unigram_lps
            else:
                scores = backward_scores
                if args.unigram_input_constraint is not None:
                    scores += args.unigram_weight * input_unigram_lps
                    
            best_scores_idxs = scores.argsort(descending = True)
            if tok_in_output:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_output_toks)
            else:
                best_scores_idxs = filter_forbidden_toks(best_scores_idxs, forbidden_input_toks)
            full_embeddings = full_embeddings.detach()

            with torch.no_grad():
                full_embeddings[:, tok_idx, :] = embedding_table[best_scores_idxs[:args.batch_size], :]                
                stacked_cur_toks[:, update_idx] = best_scores_idxs[:args.batch_size].cpu().detach().numpy()
                curr_toks_tensor[:, tok_idx] = best_scores_idxs[:args.batch_size]
                if use_prefix:
                    out = model(inputs_embeds = torch.cat([prefix_embeddings, full_embeddings], dim = 1))
                else:
                    out = model(inputs_embeds = full_embeddings)
                log_probs = F.log_softmax(out.logits[:, -1 - args.output_length: -1, :], dim = 2)
                batch_log_probs = torch.stack([log_probs[i, torch.arange(args.output_length), curr_toks_tensor[i, output_start:]].sum() for i in range(args.batch_size)])
                if args.lam_perp > 0:
                    output_perps = log_perplexity(out, stacked_cur_toks[:,:output_start], ret_all = True)
                    batch_log_probs -= args.lam_perp * output_perps
                if args.unigram_output_constraint is not None and tok_in_output:
                    batch_log_probs += args.unigram_weight * output_unigram_lps[best_scores_idxs[:args.batch_size]]
                elif args.unigram_input_constraint is not None and not tok_in_output:
                    batch_log_probs += args.unigram_weight * input_unigram_lps[best_scores_idxs[:args.batch_size]]
                best_batch_idx = batch_log_probs.argmax()
                best_idx = best_scores_idxs[best_batch_idx]
                curr_toks[update_idx] = best_idx.item()
                stacked_cur_toks[:, update_idx] = best_idx.item()
                curr_toks_tensor[:, update_idx] = best_idx.item()

                full_embeddings[:, tok_idx, :] = embedding_table[best_idx].unsqueeze(0).repeat(args.batch_size, 1)
                
                gen_output = log_probs[best_batch_idx].argmax(dim = 1)
                actual_output = curr_toks_tensor[0][output_start:]

                # Can modify success conditions here to stop running the algorithm
                output_matches = (actual_output == gen_output).all().item()
                if args.unigram_input_constraint is not None:
                    input_unigram_satisfied  = torch.exp(input_unigram_lps[curr_toks[:output_start]].min()).item() > 0.99
                else:
                    input_unigram_satisfied = True
                if args.unigram_output_constraint is not None and not fixed_output:
                    output_unigram_satisfied = torch.exp(output_unigram_lps[curr_toks[output_start:]].max()).item() > 0.5
                else:
                    output_unigram_satisfied = True


                # Success condition
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
                    return curr_toks, it, run_metadata
                
    # Failure case
    if args.lam_perp > 0:
        run_metadata['perplexity'] = None
        if args.unigram_output_constraint is not None:
            run_metadata['output_unigram'] = -1
        elif args.unigram_input_constraint is not None:
            run_metadata['input_unigram'] = -1
    return -1, -1, run_metadata
