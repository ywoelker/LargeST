from src.base.experiment import BaseExperiment, SparsityExperiment

def get_experiment(experiment_id = None, args = None):


    if experiment_id is None or experiment_id == '000':
        return BaseExperiment(
            name='Base Experiment',
            description='This is the base experiment with no specific configuration.',
        )
    
    else:
        assert args is not None, "Arguments must be provided for specific experiments."

        if experiment_id == '001':
            return SparsityExperiment(
                name='001',
                description='BigST point wise input sparsity (0.8 drop)',
                input_sparseness='point',
                input_dropout=0.8,
                output_sparseness='none',
                output_dropout=0,
                train_dropout=0,
                seed=args.seed,
                n_sensors=args.num_nodes,
                device = args.device,
            )
        
        elif experiment_id == '002':
            return SparsityExperiment(
                name='002',
                description='BigST sensor wise input sparsity (0.8 drop)',
                input_sparseness='point',
                input_dropout=0.8,
                output_sparseness='none',
                output_dropout=0,
                train_dropout=0,
                seed=args.seed,
                n_sensors=args.num_nodes,
                device = args.device,
            )
        elif experiment_id == '003':
            return SparsityExperiment(
                name='003',
                description='BigST point wise input sparsity (0.8 drop) and output sparsity (0.8 drop)',
                input_sparseness='point',
                input_dropout=0.8,
                output_sparseness='point',
                output_dropout=0.8,
                train_dropout=0,
                seed=args.seed,
                n_sensors=args.num_nodes,
                device = args.device,
            )
        elif experiment_id == '004':
            return SparsityExperiment(
                name='004',
                description='BigST training sensor excluded (0.4 drop)',
                input_sparseness='none',
                input_dropout=0,
                output_sparseness='none',
                output_dropout=0,
                train_dropout=0.4,
                seed=args.seed,
                n_sensors=args.num_nodes,
                device = args.device,
            )
        elif experiment_id == '005':
            return SparsityExperiment(
                name='005',
                description='BigST training sensor excluded (0.4 drop) with additional point wise in&out sparsity',
                input_sparseness='point',
                input_dropout=0.6,
                output_sparseness='point',
                output_dropout=0.6,
                train_dropout=0.4,
                seed=args.seed,
                n_sensors=args.num_nodes,
                device = args.device,
            )
            
