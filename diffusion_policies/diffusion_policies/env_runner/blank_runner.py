from diffusion_policies.env_runner.base_runner import BaseRunner

class BlankRunner(BaseRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy):
        return dict()
