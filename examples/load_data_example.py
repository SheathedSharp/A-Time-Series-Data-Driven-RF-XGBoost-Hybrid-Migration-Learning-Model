from utils.data_loader import DataLoader

if __name__ == "__main__":
    # Load pipeline fault data
    data_loader = DataLoader()
    data = data_loader.prepare_data(1, temporal=False)
    data = data_loader.prepare_data(1, temporal=True)
    print(f"Data shape: {data.shape}")
