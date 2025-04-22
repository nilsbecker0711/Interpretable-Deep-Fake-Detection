class ComparisonRunner:
    def __init__(self, model_config_pairs, grid_size=(3,3), max_grids=5, threshold_steps=10):
        self.model_config_pairs = model_config_pairs  # list of (model_path, config_path)
        self.grid_size = grid_size
        self.max_grids = max_grids
        self.threshold_steps = threshold_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def prepare_models(self):
        self.runners = []
        for model_path, config_path in self.model_config_pairs:
            config = load_config(model_path, config_path)
            model = load_model(config)
            model.to(self.device)
            model.eval()

            test_data_loaders = prepare_testing_data(config)
            dataset = list(test_data_loaders.values())[0].dataset

            runner = GridPointingGameCreator(
                base_output_dir="results/comparemode",
                grid_size=self.grid_size,
                xai_method=config["xai_method"],
                max_grids=self.max_grids,
                model=model,
                model_name=config.get("model_name", "default"),
                config_name=os.path.basename(config_path).split('.')[0],
                test_data_loaders=test_data_loaders,
                dataset=dataset,
                device=self.device,
                grid_split=config["grid_split"],
                overwrite=False,
                quantitativ=False,
                threshold_steps=self.threshold_steps
            )
            self.runners.append(runner)
    
    def build_shared_grid_dataset(self):

        #for model/config ccombi 

            # sample them most confident fakes from the ranking.pkl

            # sample random reals from  ranking.pkl       always shuffle reals ?

            # create x amount of grids and add to list

        
        #all models eval the final list

        





        shared_real = shared_real[:self.max_grids * (self.grid_size[0]*self.grid_size[1]-1)]
        shared_fake = shared_fake[:self.max_grids]

        # Speichern als zentrales Grid-Set mit Labels
        # Z.B. in results/comparemode/grids/
        # Verwende z.B. runner[0] um die Grids zu bauen und zu speichern
        reference_runner = self.runners[0]
        reference_runner.grid_dir = "results/comparemode/grids"
        reference_runner.results_dir = "results/comparemode/results"
        reference_runner.ranking_file = None  # kein Ranking nötig
        reference_runner.create_grids_from_paths(shared_real, shared_fake)

    def run_comparisons(self):
        for runner in self.runners:
            runner.grid_dir = "results/comparemode/grids"
            runner.results_dir = f"results/comparemode/results_{runner.model_name}_{runner.config_name}"
            os.makedirs(runner.results_dir, exist_ok=True)
            runner.run()  # führt analysis + save_results aus