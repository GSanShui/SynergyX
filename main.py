import os
import argparse
import os.path as osp
import time
import pandas as pd
import torch
import numpy as np

from models.model import SynergyxNet
from utlis import (EarlyStopping, collect_env, load_dataloader, load_infer_dataloader, 
                   set_random_seed, train, validate, infer)




def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='seed')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='device')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='maximum number of epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=50,
                        help='patience for earlystopping (default: 50)')
    parser.add_argument('--resume-from', type=str, 
                        help='the path of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        help='train or test or infer')               
    parser.add_argument('--omic', type=str, default= 'exp,mut,cn,eff,dep,met',
                        help="omics_data included in this training, separated by commas, for example: exp,mut,cn")   
    parser.add_argument('--workdir',type=str, default= os.getcwd(),
                        help='workdir of running this model')
    parser.add_argument('--celldataset', type=int, default=2,
                        help='Using which geneset to train the model(1 for 18498g, 2 for 4079g, 3 for 963g)')
    parser.add_argument('--cellencoder', type=str, default='cellCNNTrans',
                    help='cell encoder(cellTrans or cellCNNTrans)')         
    parser.add_argument('--nfold', type=str, default='0',
                        help='set index of the dataset(for example:0,1,2,indep0,blind0)'  ) 
    parser.add_argument('--saved-model', type=str, 
                        help='the path of trained_model', default='./saved_model/0_fold_SynergyX.pth')  
    parser.add_argument('--infer-path', type=str, default='./data/infer_data/sample_infer_items.npy',
                        help="The path of the infer_data_items")
    parser.add_argument('--output-attn', type=int, default=0,
                        help="whether to output the attention matrix and cell embedding in the Infer mode(0 for not, 1 for yes)")   
    return parser.parse_args()



def main():

    # pass args
    args = arg_parse()
    set_random_seed(args.seed)
    device = args.device

    # set work_dir
    work_dir = args.workdir

    # set expr_dir
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    expt_folder = osp.join('experiment/', f'{timestamp}')
    if not os.path.exists(expt_folder):
        os.makedirs(expt_folder)

    # save environmant info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()]) 
    dash_line = '-' * 60 + '\n'  
    print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    print('\n--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('\n')
    
    
    if args.mode == 'train':
        
        nfold = [i for i in args.nfold.split(',')]
        
        for k in nfold:

            model = SynergyxNet(args=args).to(device)  
            # total = sum([param.nelement() for param in model.parameters()])
            # print("Number of parameter: %.2fM" % (total/1e6))
            model.init_weights()
            criterion = torch.nn.MSELoss(reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.)
            start_epoch = 0

            # In case of unexpected interruption of model training, load the saved model and continue training
            if args.resume_from:
                resume_path=args.resume_from
                pretrain_dict = torch.load(resume_path)
                model_dict = model.state_dict()
                pretrained_dict = {k:v for k,v in pretrain_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                start_epoch = int(osp.basename(resume_path).split('_')[0]) + 1
                print(f'Load pre-trained parameters sucessfully! From epoch {start_epoch} to train……')

            tr_dataloader , val_dataloader, test_dataloader =  load_dataloader(n_fold=k,args=args)

            start_time = time.time()
            print(f'{k}_Fold_Training is starting. Start_time:{timestamp}')

            stopper = EarlyStopping(mode='lower', metric='mse', patience=args.patience, n_fold=k, folder=expt_folder) 
            for epoch in range(start_epoch, args.epochs):
                train_loss, lr_list = train(model=model,criterion=criterion,opt=optimizer,dataloader=tr_dataloader,device=device,args=args)
                val_loss, _, _, _, _, _, _ = validate(model=model,criterion=criterion,dataloader=val_dataloader,device=device,args=args)
                print('Epoch %d, Train_loss %f, Valid_loss %f'%(epoch,train_loss,val_loss)) 

                early_stop = stopper.step(val_loss, model)  
                if early_stop:
                    print('EarlyStopping! Finish training!')
                    break 
            
            # output the performance after training
            print(f'{k}_fold training is done! Training_time:{(time.time() - start_time)/60}min')
            print('Start testing ... ')

            stopper.load_checkpoint(model)
            m1,m2,m3,m4,m5,m6,m7 = validate(model=model,criterion=criterion,dataloader=tr_dataloader,device=device,args=args)
            n1,n2,n3,n4,n5,n6,n7 = validate(model=model,criterion=criterion,dataloader=val_dataloader,device=device,args=args)
            #l1,l2,l3,l4,l5,l6,l7 = validate(model=model,criterion=criterion,dataloader=test_dataloader,device=device,args=args)   
            print('Train reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f} '.format(m1,m2,m3,m4,m5,m6))
            print('Val reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f}'.format(n1,n2,n3,n4,n5,n6))
            #print('Test reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f}\n'.format(l1,l2,l3,l4,l5,l6))

        
        print('All folds training is completed!')


    elif args.mode == 'test':
        
        print('Test mode:')
        
        device = args.device
        model = SynergyxNet(args=args).to(device)  
        # load model
        saved_model = args.saved_model
        model.load_state_dict(torch.load(saved_model))  
        criterion = torch.nn.MSELoss(reduction='mean')

        k = osp.basename(saved_model).split('_')[0]
        tr_dataloader , val_dataloader, test_dataloader = load_dataloader(n_fold=k,args=args)
        l1,l2,l3,l4,l5,l6,_ = validate(model=model,criterion=criterion,dataloader=test_dataloader,device=device,args=args)
        print('Test reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f}\n'.format(l1,l2,l3,l4,l5,l6))


    elif args.mode == 'infer':
        
        print('Infer mode:')
        
        device = args.device
        model = SynergyxNet(args=args).to(device) 
        # load model
        saved_model = args.saved_model
        model.load_state_dict(torch.load(saved_model))  
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        infer_dataloader, infer_data_arr = load_infer_dataloader(args=args)
        y_pred_arr, cell_embed_arr, attn_arr = infer(model=model,dataloader=infer_dataloader,device=device,args=args)
        output_arr = np.concatenate((infer_data_arr,y_pred_arr),axis=1)

        print('Inferrence step is done! Saving to file……')
        output_df = pd.DataFrame(output_arr,columns=['drugA','drugB','sample_id','label','S_pred'])
        output_df.to_csv(f'experiment/{timestamp}/predict_res.csv')

        if args.output_attn:
            np.save(f'experiment/{timestamp}/cell_embed.npy',cell_embed_arr)
            np.save(f'experiment/{timestamp}/attn.npy',attn_arr)


if __name__ == '__main__':
   
    main()

