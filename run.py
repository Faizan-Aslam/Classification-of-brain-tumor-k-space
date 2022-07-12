import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./configs", config_name='config')
def main(config: DictConfig):
    from train_complex import train_complex

    print(OmegaConf.to_yaml(config))

    train_complex(config)


if __name__ == "__main__":
    main()
