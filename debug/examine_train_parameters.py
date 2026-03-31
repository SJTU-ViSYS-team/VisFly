from stable_baselines3 import PPO

def main(path):
    model = PPO.load(path)
    print(model.policy)
    print(model.get_parameters())
    for param in model.policy.parameters():
        print(param)
        

if __name__ == "__main__":
    path = "exps/std/saved/objTracking/SHAC_forwardTrack_Col_ColScene_slowTar_3.zip"
    main(path)