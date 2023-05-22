from src.fix_seed import seed_everything
from src.create_df import pivot_df_data
from src.model_SNN import SimpleCNN
import torch
import torch.nn as nn
import config
from src.train_model import train_model

def run_train() -> None:
    seed_everything(config.seed)
    print(config.path_data + '/*/*/*.png')
    df = pivot_df_data(config.path_data)
    print(df.head())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_model(
        model,
        config.learning_rate,
        optimizer,
        criterion,
        epochs=config.epochs,
        df=df
      )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_train()
