import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
from skimage.measure import compare_psnr as psnr
import os
def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')
if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data##加载数据集，打印数据量
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    ##logger ##创建一个日志文件，并将一些信息写入日志文件中
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    #validation data#加载验证集，打印数据量
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size)
    if opt.model=='cycle_gan':#为训练过程准备数据结构，并创建模型和可视化器
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])            
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])       
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()#在每个epoch开始时记录当前时间epoch_start_time，并初始化一些计时变量
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'#将opt.phase设置为'train'，表示当前处于训练阶段
        for i, data in enumerate(dataset):#遍历训练数据集dataset，对每个数据进行训练
            iter_start_time = time.time()#对于每个数据，记录当前时间iter_start_time
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time#计算数据加载时间t_data
            visualizer.reset()
            total_steps += opt.batchSize#记录总的训练步数
            epoch_iter += opt.batchSize#记录当前epoch中已处理的样本数量
            model.set_input(data)
            
            model.optimize_parameters()#优化模型参数，即进行一次参数更新

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0#如果当前的总训练步数total_steps能被opt.display_freq整除，表示需要显示当前模型生成的结果
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()#如果数据集模式为aligned_mat，则从模型中获取当前的可视化结果temp_visuals
                    visualizer.display_current_results(temp_visuals, epoch, save_result)#显示和保存结果
                elif  opt.dataset_mode=='unaligned_mat':  #表示如果opt.dataset_mode的取值为unaligned_mat，则执行下面的代码块 
                    temp_visuals=model.get_current_visuals()#从模型中获取当前的可视化结果，并将其存储在temp_visuals变量中
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]#对temp_visuals中的real_A图像进行处理，将其通道数裁剪为前3个通道
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:#检查opt.lambda_identity是否大于0，如果是，则执行以下操作
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]#对temp_visuals中的idt_A图像进行处理，将其通道数裁剪为前3个通道
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result) #显示和保存处理后的可视化结果                   
                else:#这里直接显示和保存未经过额外处理的可视化结果
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        #Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase='val'
            for i, data_val in enumerate(dataset_val):
#        		    
                model.set_input(data_val)
#        		    
                model.test()
#        		    
                fake_im=model.fake_B.cpu().data.numpy()
#        		    
                real_im=model.real_B.cpu().data.numpy() 
#        		    
                real_im=real_im*0.5+0.5
#        		    
                fake_im=fake_im*0.5+0.5
                if real_im.max() <= 0:
                    continue
                L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max())
#                  
#                 
            l1_avg_loss = np.mean(L1_avg[epoch-1])
#                
            mean_psnr = np.mean(psnr_avg[epoch-1])
#                
            std_psnr = np.std(psnr_avg[epoch-1])
            print(mean_psnr)
            print(l1_avg_loss)    
            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f ' % \
            (epoch, l1_avg_loss, mean_psnr,std_psnr))
#                
            print_log(logger,'')
            logger.close()
    		   #
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
#        		    
            model.save('latest')
#        		   
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
