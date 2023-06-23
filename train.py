
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os
import time
import wandb
from datetime import datetime

from configs import cfg
from datasets import *

from models.seq2seq import *
from models.attention import *
from utils import setup_logger

from utils.metrics import *

def combine_cfg(config_dir=None):
    cfg_base = cfg.clone()
    if config_dir:
        cfg_base.merge_from_file(config_dir)
    return cfg_base 

def train(cfg, logger, encoder_weight = None, decoder_weight = None, output_dir= None, max_iter =None):


    input_lang, output_lang, train_loader = get_dataloader(cfg.SOLVER.BATCH_SIZE, filename = cfg.DATASETS.TRAIN)

    input_lang_test, output_lang_test, pairs = prepareData('eng', 'para', True, filename= cfg.DATASETS.TEST)

    logger.info("Begin the training process")
    wandb.login(key="cbcabc061fb4a62d6ebeae24db563b71d7747fb6")
    wandb.init(project= "NLP Project", name= "Seq2Seq", config= {"batch_size": cfg.SOLVER.BATCH_SIZE,
                                             "hidden_size": cfg.MODEL.HIDDEN_SIZE,
                                             "device": cfg.MODEL.DEVICE})
    device = torch.device(cfg.MODEL.DEVICE)
    iter = 0
    encoder = EncoderRNN(input_lang.n_words, cfg.MODEL.HIDDEN_SIZE).to(device)
    if encoder_weight is not None:
        save_encoder = torch.load(encoder_weight)
        encoder.load_state_dict(save_encoder["encoder_state_dict"])
        iter = save_decoder["iteration"]
    decoder = AttnDecoderRNN(cfg.MODEL.HIDDEN_SIZE, output_lang.n_words).to(device)
    if decoder_weight is not None:
        save_decoder = torch.load(decoder_weight)
        decoder.load_state_dict(save_decoder["decoder_state_dict"])

    if not max_iter:
        max_iter = 40000
    if not output_dir:
        output_dir = cfg.OUTPUT_DIR

    criterion = nn.NLLLoss()
    encoder.train()
    decoder.train()


    encoder_optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    encoder_optimizer.zero_grad()
    decoder_optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], 
    lr=cfg.SOLVER.LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    decoder_optimizer.zero_grad()
    
    logger.info("Number of train data : " + str(len(train_loader)*cfg.SOLVER.BATCH_SIZE))
    
    logger.info("Start training")
    model.train()
    end = time.time()
    best_bleu = 0
    while iter < max_iter:
        for data in train_loader:
            model.train()
            data_time = time.time() - end
            end = time.time()
            
            input_tensor, target_tensor = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            encoder_optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iter/max_iter)**cfg.SOLVER.POWER
            decoder_optimizer.param_groups[0]['lr'] = cfg.SOLVER.LR * (1 - iter/max_iter)**cfg.SOLVER.POWER
            infor = {"encoder_lr": encoder_optimizer.param_groups[0]['lr'],
                 "decoder_lr": decoder_optimizer.param_groups[0]['lr'],
                    "loss": loss,
                    "time/iter": data_time,
                    "iter": iter/max_iter}
            if iter % 20 == 0:

                logger.info("Iter [%d/%d], loss: [%f] , Time/iter [%f]" %(iter, max_iter,loss, data_time))

            if iter % 100 == 0:

                logger.info("Validation mode")
                model.eval()
                with torch.no_grad():
                    bleu = 0
                    for pair in pairs:
                        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang_test, output_lang_test)
                        output_sentence = ' '.join(output_words)
                        bleu += bleu_score(pair[0], output_sentence)
                    bleu = bleu/len(pairs)
                    infor.update({"bleu": bleu})
                    logger.info("Bleu score: [%f]" %(bleu))

                    torch.save({"encoder_state_dict": encoder.state_dict(), 
                                            "iteration": iter,
                                            }, os.path.join(output_dir, "current_encoder.pkl"))
                    torch.save({"decoder_state_dict": decoder.state_dict(), 
                                            "iteration": iter,
                                            }, os.path.join(output_dir, "current_decoder.pkl"))
                    if bleu > best_bleu:
                        best_bleu = bleu
                        torch.save({"encoder_state_dict": encoder.state_dict(), 
                                            "iteration": iter,
                                            }, os.path.join(output_dir, "best_encoder.pkl"))
                        torch.save({"decoder_state_dict": decoder.state_dict(), 
                                            "iteration": iter,
                                            }, os.path.join(output_dir, "best_decoder.pkl"))
                    logger.info(evaluateRandomly(encoder, decoder, input_lang_test, output_lang_test))
                    
                
            wandb.log(infor)
            if iter == max_iter:
                break
    return model 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch training")
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iteration", default=None)
    args = parser.parse_args()
    
    logger = setup_logger("NLP_train", args.output_dir , str(datetime.now()) + ".log")
    model = train(cfg, logger,args.encoder, args.decoder , output_dir= args.output_dir, max_iter = args.max_iteration )
