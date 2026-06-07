from simple_slurm import Slurm

DATASET = 'SD'

HOST = 'TUHH'

META_DATA_FEATURES = {
        'SD': 38,
        'GBA': 62,
        'GLA': 74,
        'CA': 189           
    }


GBA_H100 = {
    'AGCRN': 64, 
    'D2STGNN': 8,
    'DCRNN': 64, 
    'DGCRN': 64, 
    'GMAN': 1, 
    'DSTAGNN': 64,
    'ASTGCN': 64,
    'OPCR': 64, 
    'SPIN': 4,
    'STTN': 8,
    'STGODE': 64,
    'PDFORMER': 4}
GBA_V100 = {
    'BIGST': 64, 
    'DSGNN': 64,
    'GSNET': 64,
    'GWNET': 64,
    'STGCN': 64,
    

}




def create_slurm_job(model_name:str, mask_name:str | None, mask_iter:int, gpu_h100 = False, additional_cmd_str:str = '', fire = True, gpu_selection = 'none', job_name_suffix = ''):

    if mask_name is None:  
        job_name = f'{model_name}_nomask'
    else:
        job_name = f'{model_name}_{mask_name}_{mask_iter}' 
    if len(job_name_suffix) > 0 and job_name_suffix is not None:
        job_name = job_name + '_' + job_name_suffix


    if HOST == 'TUHH':
        slurm = Slurm(
            "--job_name", job_name,
            "--ntasks ", 1,
            "--cpus_per_task", 8,
            "--gres", 'gpu:1', 
            "--mem", "64000",
            "--time", "23:00:00",
            "--output", f'scripts_logs/{job_name}_%j.out',
        )

        slurm.add_cmd('export OMP_NUM_THREADS=8')
        slurm.add_cmd('module load cuda')
        slurm.add_cmd('source .venv/bin/activate')

    else:

        slurm = Slurm(
            "--job_name", job_name,
            "--nodes", 1,
            "--ntasks_per_node", 1,
            "--cpus_per_task", 8,
            "--gpus_per_node", 1, 
            "--mem", "64000",
            "--time", "23:00:00",
            "--partition", "gpu",
            "--output", f'scripts_logs/{job_name}_%j.out',
        )

        if gpu_h100:
            slurm.add_arguments("--constraint", "H100")
        elif gpu_selection != 'none':
            slurm.add_arguments("--constraint", gpu_selection)

        slurm.add_cmd('export OMP_NUM_THREADS=8')
        slurm.add_cmd('export http_proxy=http://10.0.7.235:3128')
        slurm.add_cmd('export https_proxy=http://10.0.7.235:3128')
        slurm.add_cmd('export ftp_proxy=http://10.0.7.235:3128')
        slurm.add_cmd('module load gpu-env')
        slurm.add_cmd('module load cuda cudnn miniconda3')
        slurm.add_cmd('source ~/.bashrc')
        slurm.add_cmd('conda activate multi_context2.0')
        slurm.add_cmd('echo $CUDA_VISIBLE_DEVICES')


    slurm.add_cmd('mkdir -p /tmp')


    cmd_str = f'python experiments/{model_name.lower()}/main.py --dataset {DATASET} --years 2019 --device cuda --model_name {model_name} --run_description {job_name}'
    
    if mask_name is not None:
        cmd_str += f' --mask_name {mask_name} --mask_iter {mask_iter}'

    if len(additional_cmd_str) > 0:
        cmd_str += ' ' + additional_cmd_str


    slurm.add_cmd(cmd_str)




    #--model_description gsnet 
    #--input_dim 74 - GLA
    #--input_dim 38 - SD
    #--input_dim 186 -CA

    
    if HOST != 'TUHH':
        slurm.add_cmd('jobinfo')

    if fire:
        slurm.sbatch()

    return slurm

def ablation_dsn_embed_dim(dsn_embed_dims): 
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050']:#,'point_missing_075']: 
        model_name = 'DSGNN'

        for dsn_embed_dim in dsn_embed_dims:
            additional_cmd_str = f' --n_context_emb {dsn_embed_dim}'

            create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_dsn_embed_dim --training_timeout_min 600')

def ablation_gcn_layers(gcn_layers):
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050','point_missing_075']: 
        model_name = 'DSGNN'

        for gcn_layer in gcn_layers:
            additional_cmd_str = f' --gcn_layers {gcn_layer}'

            create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_gcn_layers --training_timeout_min 600', gpu_selection= 'V100')


def ablation_concat_queries():
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050','point_missing_075']: 
        model_name = 'DSGNN'

        for adding_query_to_dsn in [True, False]:
            additional_cmd_str = f' --adding_query_to_dsn {adding_query_to_dsn}'

            create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_concat_queries --training_timeout_min 600')
            
            
def ablation_repulsion_loss(): 
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050','point_missing_075']:
        model_name = 'DSGNN'
        for repulsion_margin in [0.0] :
            for repulsion_loss_weight in [0.0]:
                additional_cmd_str = f' --dsn_div_weight {repulsion_loss_weight} --dsn_div_margin {repulsion_margin}'

                create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_repulsion_loss_v2 --training_timeout_min 600 --n_hid 64 --gcn_layers 2 --dropout 0.1 --n_context_emb 64 --additional_loss_weight 0')
                
def ablation_dropout(dropout_rates):
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050','point_missing_075']:
        model_name = 'DSGNN'
        for dropout_rate in dropout_rates:
            additional_cmd_str = f' --dropout {dropout_rate}'

            create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_dropout --training_timeout_min 600')
            
            
def ablation_hidden_dim():
    mask_iter = 1
    # for mask_name in [None, 'tr_drop_050', 'tr_drop_075','point_missing_075']:
    for mask_name in ['point_missing_075']:
        model_name = 'DSGNN'
        for n_hid in [16, 32, 64, 128]:
            additional_cmd_str = f' --n_hid {n_hid}'

            command = create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100=False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_hidden_dim --training_timeout_min 600 --gcn_layers 2 --dropout 0.1 --n_context_emb 64 --additional_loss_weight 0 --dsn_div_weight 1 --dsn_div_margin 0.001', gpu_selection= 'V100', job_name_suffix= f'n_hid_{n_hid}_ablation')
            


def dsgnn_test():
    mask_iter = 0
    for mask_name in [None]:
        model_name = 'DSGNN'
        additional_cmd_str = f''
        create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= True, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags dsgnn_idea --training_timeout_min 600 --train_data_percentage 0.2 --max_epochs 15 --n_hid 32 --gcn_layers 1 --dropout 0.1 --n_context_emb 64 --additional_loss_weight 0.0 --dsn_div_weight 10 --dsn_div_margin 0.001' )


def ablation_static_prefilter_mode():
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050', 'tr_drop_075','point_missing_075']:
        model_name = 'DSGNN'
        
        dsn_number_options ={
            'fixed': [50, 100, 200, 350],
            'static_dsn': [64],
            'identity': [64],
            'none': [50, 100, 200, 350]
        }
        
        
        
        for static_prefilter_mode in ['fixed', 'static_dsn', 'identity', 'none']:
            
            for n_context in dsn_number_options[static_prefilter_mode]:
            
                additional_cmd_str = f' --static_prefilter_mode {static_prefilter_mode} --n_context {n_context} '
                
                create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100=False, additional_cmd_str=additional_cmd_str + f'--use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_static_prefilter_mode --training_timeout_min 600 --gcn_layers 2 --dropout 0.1 --n_context_emb 64 --additional_loss_weight 0 --dsn_div_weight 1 --dsn_div_margin 0.001 --max_epochs 50')
            
def ablation_attention_method():
    mask_iter = 0
    for mask_name in [None, 'tr_drop_050', 'tr_drop_075','point_missing_075']:
        model_name = 'DSGNN'
        
        for attention_method in ['MLA', 'MHA']:
            
            additional_cmd_str = f' --attention_method {attention_method} '
            
            create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100=False, additional_cmd_str=additional_cmd_str + f'--use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags ablation_attention_method --training_timeout_min 600 --gcn_layers 2 --dropout 0.1 --n_context_emb 64 --additional_loss_weight 0 --dsn_div_weight 0.01 --dsn_div_margin 0.003 --max_epochs 50')


if __name__ == '__main__':

    # dsgnn_test()
    # ablation_dsn_embed_dim([32,512, 1024])
    # ablation_gcn_layers([4])
    # ablation_concat_queries()
    # ablation_repulsion_loss()
    # ablation_dropout([0, 0.01, 0.1, 0.3, 0.5])
    # ablation_hidden_dim()
    # ablation_static_prefilter_mode()
    # ablation_attention_method()

    for mask_iter in [0,1]:
        for mask_name in [None,'point_missing_050', 'point_missing_075', 'point_missing_095']: #[None, 'tr_drop_025', 'tr_drop_050', 'tr_drop_075', 'point_missing_050', 'point_missing_075', 'point_missing_095']: 
            # for model_name in ['BigST', 'GSNet', 'AGCRN', 'ASTGCN', 'STGCN', 'DCRNN', 'D2STGNN', 'GMAN', 'GWNET', 'OPCR', 'STGode', 'DSGNN']:
            for model_name in ['DSGNN', 'SparseStateGNN']:#['D2STGNN', 'GMAN', 'PDFormer', 'SPIN', 'OPCR']: #['BigST', 'GSNet', 'AGCRN', 'ASTGCN', 'STGCN', 'DCRNN', 'D2STGNN', 'GMAN', 'GWNET', 'OPCR', 'STGode', 'DSGNN' ,'PDFormer', 'SPIN']:
                gpu_h100 = model_name in ['D2STGNN', 'GMAN', 'DCRNN', 'SPIN']

                if DATASET == 'SD':
                    if model_name.lower() == 'dstagnn':
                        additional_cmd_str = '--input_dim 1'
                    elif model_name.lower() == 'gman' or model_name.lower() == 'spin' or model_name.lower() == 'pdformer':
                        additional_cmd_str = '--bs 16'
                    else:
                        additional_cmd_str = ''

                # if mask_iter == 2 and mask_name != 'point_missing_075' and mask_name != 'point_missing_095':
                #     continue

                # if mask_iter == 3 and mask_name == 'tr_drop_050':
                #     continue
                

                elif DATASET == 'CA':

                    additional_cmd_str = ''
                    if model_name.lower() == 'd2stgnn':
                        additional_cmd_str += '--bs 32'

                elif DATASET == 'GBA':
                     
                    additional_cmd_str = ''
                    if model_name.upper() in GBA_H100:
                        gpu_h100 = True
                        
                        if GBA_H100[model_name.upper()] < 64:
                            additional_cmd_str  += f' --bs {GBA_H100[model_name.upper()]}'

                    elif model_name.upper() in GBA_V100:
                        gpu_h100 = False
                        if GBA_V100[model_name.upper()] < 64:
                            additional_cmd_str  += f' --bs {GBA_V100[model_name.upper()]}'
                else:
                    additional_cmd_str = ''
                    
                if model_name.lower() == 'pdformer':
                        additional_cmd_str += ' --enc_depth 4 --add_time_in_day True --add_day_in_week True'
                    
                    
                if model_name.lower() == 'dsgnn' or model_name.lower() == 'sparsestategnn':
                    additional_cmd_str = additional_cmd_str + ' --n_hid 64 --gcn_layers 2 --dropout 0.1 --n_context_emb 32 --additional_loss_weight 0 --dsn_div_weight 1 --dsn_div_margin 0.001 '
                    # additional_cmd_str = additional_cmd_str + ' --n_hid 32 --gcn_layers 2 --dropout 0.1 --n_context_emb 16 --additional_loss_weight 0 --dsn_div_weight 0.01 --dsn_div_margin 0.003 '
                    
                slurm = create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100=True, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]} --wandb_tags sparsestate --max_epochs 50 --train_data_percentage 1.0', fire = False)
                print(slurm)
                                 
                #'--wandb_tags inference_time_benchmark --max_epochs 1 --train_data_percentage 0.02')
                # create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= True, additional_cmd_str=additional_cmd_str + f' --wandb_tags benchmark_GBA --training_timeout_min 600')

    # mask_name = 'None'
    # model_name = 'DSGNN'
    # tags = 'dev_bench'˚
    # jobs = []
    # for mask_iter in [0,1,2]: 


    #     additional_cmd_str = f'--wandb_tags {tags} --training_timeout_min 600 --dropout .123 --dsn_div_weight 0.24'
    #     potential_job = create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str=additional_cmd_str + f' --use_metadata True --input_dim {META_DATA_FEATURES[DATASET]}')
    #     jobs.append(potential_job)


    # print(f'Total jobs to submit: {len(jobs)}')

    # #Take random 128 jobs to submit
    # import random
    # random.shuffle(jobs)

    # for job in jobs[:25]:
    #     job.sbatch()

    # for mask_iter in [0]:#[0,1,2]: # [0,1,2,3,4]
    #     for mask_name in ['point_missing_075']:
    #         model_name = 'DSGNN'

    #         for n_context in [350, 200, 100, 50]:
    #             for n_context_emb in [64]: #[16,64,128,256]:

    #                 create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str= f' --use_metadata True --input_dim 38 --wandb_tags fixed_context --n_context {n_context} --n_context_emb {n_context_emb} --static_prefilter_mode fixed --max_epochs 64' )

    #         create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str= f' --use_metadata True --input_dim 38 --wandb_tags fixed_context --n_context {n_context} --n_context_emb {n_context_emb} --static_prefilter_mode static_dsn --max_epochs 64' )
    #         create_slurm_job(model_name=model_name, mask_name=mask_name, mask_iter=mask_iter, gpu_h100= False, additional_cmd_str= f' --use_metadata True --input_dim 38 --wandb_tags fixed_context --n_context {n_context} --n_context_emb {n_context_emb} --static_prefilter_mode identity --max_epochs 64' )




