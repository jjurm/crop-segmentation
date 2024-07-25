import copy
import yaml

with open("unet_template.yaml", "r") as f:
    template = yaml.safe_load(f)

# Define the grid of experiments
experiment_overrides = {
    "unet_bl2_ccm": {
        "block_size": 2,
        "active_sampling_relevancy_score": "uncertainty-margin",
        "learning_rate": 0.0003,
    },
    "unet_bl2_loss": {
        "block_size": 2,
        "active_sampling_relevancy_score": "loss",
        "learning_rate": 0.001,
    },
    "unet_bl2_random": {
        "block_size": 2,
        "active_sampling_relevancy_score": "random",
        "learning_rate": [0.03, 0.01, 0.003],
    },
    "unet_bl2_rho": {
        "block_size": 2,
        "active_sampling_relevancy_score": "rho-loss-model-brisk-lion-807:best",
        "learning_rate": [0.01, 0.003, 0.001, 0.0003],
    },

    "unet_bl8_ccm": {
        "block_size": 8,
        "active_sampling_relevancy_score": "uncertainty-margin",
        "learning_rate": 0.0003,
    },
    "unet_bl8_loss": {
        "block_size": 8,
        "active_sampling_relevancy_score": "loss",
        "learning_rate": 0.001,
    },
    "unet_bl8_random": {
        "block_size": 8,
        "active_sampling_relevancy_score": "random",
        "learning_rate": [0.03, 0.01, 0.003],
    },
    "unet_bl8_rho": {
        "block_size": 8,
        "active_sampling_relevancy_score": "rho-loss-model-brisk-lion-807:best",
        "learning_rate": [0.003, 0.001, 0.0003],
    },
}

for experiment_name, experiment in experiment_overrides.items():
    template_copy = copy.deepcopy(template)
    template_copy["name"] = template_copy["name"].replace("$", experiment_name.replace("_", "-"))
    for key, value in experiment.items():
        template_copy["parameters"][key] = {
            ("values" if isinstance(value, list) else "value"): value,
        }
    with open(f"{experiment_name}.yaml", "w") as f:
        yaml.dump(template_copy, f)
