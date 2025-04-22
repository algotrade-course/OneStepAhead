import json
import torch

from torch import optim 
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from configs import config
from dataset import StockPriceDateset, split_data
from model import CustomTransformerModel, load_checkpoint, save_checkpoint


def train(
    model: CustomTransformerModel,
    data_loader: DataLoader,
    optimizer,
    device: str
):
    model = model.train()
    model = model.to(device)

    total_loss = 0
    sample_cnt = 0

    for i, [past_values, past_time_features, past_masks, future_values, future_additional_features, future_masks, _] in enumerate(data_loader):        
        past_values = past_values.to(device)
        past_time_features = past_time_features.to(device)
        past_masks = past_masks.to(device)
        future_values = future_values.to(device)
        future_additional_features = future_additional_features.to(device)
        future_masks = future_masks.to(device)


        output = model(
            past_values=past_values, 
            past_time_features=past_time_features, 
            past_observed_mask=past_masks, 
            future_values=future_values, 
            future_time_features=future_additional_features,
            future_observed_mask=future_masks
        )

        loss = output.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * past_values.shape[0]
        sample_cnt += past_values.shape[0]

    print(f"Training: Loss {total_loss / sample_cnt}")
    #     print(f"Training {i + 1}/{len(data_loader)}: Loss {total_loss / sample_cnt}", end='\r')
    # print()
    return total_loss / sample_cnt


def validate(
    model: CustomTransformerModel,
    data_loader: DataLoader,
    device: str
):
    model = model.eval()
    model = model.to(device)

    total_loss = 0
    sample_cnt = 0
    
    for i, [past_values, past_time_features, past_masks, future_values, future_additional_features, future_masks, _] in enumerate(data_loader):        
        past_values = past_values.to(device)
        past_time_features = past_time_features.to(device)
        past_masks = past_masks.to(device)
        future_values = future_values.to(device)
        future_additional_features = future_additional_features.to(device)
        future_masks = future_masks.to(device)
    

        with torch.no_grad():
            output = model(
                past_values=past_values, 
                past_time_features=past_time_features, 
                past_observed_mask=past_masks, 
                future_values=future_values, 
                future_time_features=future_additional_features,
                future_observed_mask=future_masks
            )

            loss = output.loss
            total_loss += loss.item() * past_values.shape[0]
            sample_cnt += past_values.shape[0]

    print(f"Validating: Loss {total_loss / sample_cnt}")
    #     print(f"Validating {i + 1}/{len(data_loader)}: Loss {total_loss / sample_cnt}", end='\r')
    # print()
    return total_loss / sample_cnt


def RMSEValidating(
    model: CustomTransformerModel,
    data_loader: DataLoader,
    device: str
):
    model = model.eval()
    model = model.to(device)

    total_mse = 0
    values_cnt = 0
    mse_fn = MSELoss(reduction='none')
    
    for i, [past_values, past_time_features, past_masks, future_values, future_additional_features, future_masks, _] in enumerate(data_loader):        
        past_values = past_values.to(device)
        past_time_features = past_time_features.to(device)
        past_masks = past_masks.to(device)
        future_values = future_values.to(device)
        future_additional_features = future_additional_features.to(device)
        future_masks = future_masks.to(device)

        with torch.no_grad():
            [means, stds] = model.generate(
                past_values=past_values, 
                past_time_features=past_time_features, 
                past_observed_mask=past_masks, 
                future_time_features=future_additional_features
            )
            # print(means.size())
            prediction = means
            # predicted_samples = output.sequences
            # prediction = torch.mean(predicted_samples, dim=1)

            mse = mse_fn(prediction, future_values)
            masked_mse = (torch.where(future_masks > 0, mse, torch.nan)).nansum()
            total_mse += masked_mse.item()
            values_cnt += torch.sum(future_masks).item()

    print(f"Validating: RMSE {(total_mse / values_cnt) ** 0.5}")
    #     print(f"Validating {i + 1}/{len(data_loader)}: RMSE {(total_mse / values_cnt) ** 0.5}", end='\r')
    # print()
    return (total_mse / values_cnt) ** 0.5


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    SEED = 0
    torch.manual_seed(SEED)

    # Data Preparation
    with open(config.data_path, "r") as fin:
        data = json.load(fin)

    train_data, val_data, test_data = split_data(data, ratio=(6, 2, 2))
    max_lag = max(config.lags_sequence)
    train_set = StockPriceDateset(
        data=train_data,
        token_interval=config.time_step_interval,
        context_length=config.context_length + max_lag,
        prediction_length=config.prediction_length
    )
    val_set = StockPriceDateset(
        data=val_data,
        token_interval=config.time_step_interval,
        context_length=config.context_length + max_lag,
        prediction_length=config.prediction_length
    )
    test_set = StockPriceDateset(
        data=test_data,
        token_interval=config.time_step_interval,
        context_length=config.context_length + max_lag,
        prediction_length=config.prediction_length
    )

    n_workers = 4
    epochs = 200
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=n_workers)

    # Model
    model = CustomTransformerModel(config.model_config)


    weight_decay = 1e-5
    initial_lr = 1e-3
    final_lr = 1e-6
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    # lr_scheduler = ChainedScheduler(optimizer, T_0=20, T_mul=1, eta_min=final_lr, max_lr=initial_lr, warmup_steps=3, gamma=0.8)

    last_epoch = -1
    min_loss = 1e9
    checkpoint_path = config.model_path

    loading = False
    if loading:
        [model, optimizer, last_epoch, _] = load_checkpoint(checkpoint_path, model)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=final_lr, last_epoch=last_epoch)

    while last_epoch + 1 < epochs:
        last_epoch += 1
        print(f"Epoch {last_epoch + 1}:")
        print(f"LR {lr_scheduler.get_last_lr()}")
        train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        lr_scheduler.step()

        if val_loss < min_loss:
            min_loss = val_loss
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, epoch=last_epoch, loss=val_loss)
    
    [model, _, _, _] = load_checkpoint(checkpoint_path, model)
    print("Testing:")
    validate(model, test_loader, device)
    RMSEValidating(model, test_loader, device)