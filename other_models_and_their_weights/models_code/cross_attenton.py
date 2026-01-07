
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms 
import rasterio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- CONFIG ---
CSV_FILE = 'train_final_fusion.csv' 
IMG_DIR = 'naip_images/train_640'
MODEL_SAVE_PATH = 'sota_fusion_best.pth'
SUBMISSION_SAVE_PATH = 'final_model_submissions.csv'

# HYPERPARAMETERS
BATCH_SIZE = 128      
LR = 0.0005         
EPOCHS = 100        
WEIGHT_DECAY = 0.1    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATIENCE_EARLY_STOPPING = 8  
PATIENCE_SCHEDULER = 4       

# ==========================================
# 0. AUGMENTATION
# ==========================================
aug_flip_h = transforms.RandomHorizontalFlip(p=0.5)
aug_flip_v = transforms.RandomVerticalFlip(p=0.5)
aug_rotate = transforms.RandomRotation(degrees=30) 
aug_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)

# ==========================================
# 1. DATASET
# ==========================================
class MultimodalDataset(Dataset):
    def __init__(self, df, img_dir, is_train=False):
        self.df = df
        self.img_dir = img_dir
        self.is_train = is_train
        
        # TARGET: Log-Residual
        self.df['log_price'] = np.log(self.df['price'])
        self.df['log_xgb'] = np.log(self.df['price_pred_xgb'])
        self.df['target_residual'] = self.df['log_price'] - self.df['log_xgb']

        # Feature Selection
        excluded_cols = [
            'id', 'date', 'price', 'log_price', 'price_pred_xgb', 
            'residual', 'residual_log', 'target_residual', 'abs_residual',
            'error_category', 'alpha', 'log_price_pred', 'log_xgb'
        ]
        
        self.features = [c for c in self.df.columns if c not in excluded_cols]
        if 'xgb_pred_log' not in self.features and 'xgb_pred_log' in self.df.columns:
            self.features.append('xgb_pred_log')

        self.tab_data = self.df[self.features].values.astype(np.float32)
        self.tab_mean = self.tab_data.mean(axis=0)
        self.tab_std = self.tab_data.std(axis=0) + 1e-6
        self.tab_data = (self.tab_data - self.tab_mean) / self.tab_std
        
        self.targets = self.df['target_residual'].values.astype(np.float32)
        self.ids = self.df['id'].values
        
        # 4-Channel Normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485], 
            std=[0.229, 0.224, 0.225, 0.229]
        )

        print(f"[{'TRAIN' if is_train else 'VAL'}] Pre-loading {len(self.ids)} images...")
        self.image_cache = {}
        self._preload_images()

    def _preload_images(self):
        from tqdm import tqdm
        for img_id in tqdm(self.ids, desc="Caching"):
            img_path = os.path.join(self.img_dir, f"{img_id}.tif")
            try:
                with rasterio.open(img_path) as src:
                    image = src.read([1, 2, 3, 4]) 
                    image = torch.from_numpy(image).float()
                    if image.shape[1] != 224:
                         image = torch.nn.functional.interpolate(
                            image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                        ).squeeze(0)
                    self.image_cache[img_id] = image.byte() 
            except Exception:
                self.image_cache[img_id] = torch.zeros((4, 224, 224), dtype=torch.uint8)

    def __len__(self):
        if self.is_train: return len(self.df) * 5 
        return len(self.df)

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        aug_mode = idx // len(self.df)
        img_id = self.ids[real_idx]
        image = self.image_cache[img_id].float() / 255.0  
        
        if self.is_train:
            if aug_mode == 1: image = aug_flip_h(image)
            elif aug_mode == 2: image = aug_flip_v(image)
            elif aug_mode == 3: image = aug_rotate(image)
            elif aug_mode == 4:
                rgb = image[:3, :, :]
                nir = image[3:, :, :]
                rgb = aug_color(rgb)
                image = torch.cat([rgb, nir], dim=0)

        image = self.normalize(image)
        tab = torch.tensor(self.tab_data[real_idx], dtype=torch.float32)
        target = torch.tensor(self.targets[real_idx], dtype=torch.float32)
        return image, tab, target, img_id

# ==========================================
# 2. MODEL (CROSS-ATTENTION ARCHITECTURE)
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, tab_input_dim, embed_dim=128, num_heads=4):
        super(FusionModel, self).__init__()
        
        # 1. ResNet Backbone
        self.cnn = models.resnet50(weights='IMAGENET1K_V1')
        
        # 4-Channel Fix
        original_weights = self.cnn.conv1.weight.data
        new_conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data[:, :3] = original_weights
        new_conv1.weight.data[:, 3] = original_weights[:, 0] 
        self.cnn.conv1 = new_conv1
        self.cnn.fc = nn.Identity() 

        # FREEZE BACKBONE (As requested)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # --- NEW ARCHITECTURE: PROJECTION ---
        # Project Image (2048) -> Embed Dim (128)
        self.img_proj = nn.Sequential(
            nn.Linear(2048, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Project Tabular (N) -> Embed Dim (128)
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # --- CROSS ATTENTION ---
        # Query: Tabular (We want to enhance this)
        # Key/Value: Image (Source of visual context)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        
        # Norm for Residual Connection
        self.norm = nn.LayerNorm(embed_dim)

        # Final Regressor Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        # 1. Extract Features
        img_feat = self.cnn(img)  # (Batch, 2048)
        
        # 2. Project to Common Dimension
        img_emb = self.img_proj(img_feat)  # (Batch, 128)
        tab_emb = self.tab_proj(tab)       # (Batch, 128)
        
        # 3. Reshape for Attention (Batch, SeqLen, Embed)
        # We treat these as sequences of length 1
        query = tab_emb.unsqueeze(1)  # (Batch, 1, 128)
        key_val = img_emb.unsqueeze(1) # (Batch, 1, 128)
        
        # 4. Cross Attention
        # attn_output represents "Image features weighted by their relevance to the Tabular data"
        attn_output, _ = self.cross_attn(query=query, key=key_val, value=key_val)
        
        # 5. Residual Connection + Norm (Transformer Block Style)
        # We add the original Tabular embedding back. 
        # This is CRITICAL: If image is useless, model falls back to Tabular.
        combined = self.norm(query + attn_output) 
        
        # 6. Squeeze and Predict
        combined = combined.squeeze(1) # (Batch, 128)
        return self.head(combined).squeeze()

# ==========================================
# 3. MAIN
# ==========================================
def main():
    df = pd.read_csv(CSV_FILE)
    df = df.drop_duplicates(subset=['id'], keep='first')
    df = df[df['price'] > 0]
    df = df[df['price_pred_xgb'] > 0]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = MultimodalDataset(train_df, IMG_DIR, is_train=True)
    val_dataset = MultimodalDataset(val_df, IMG_DIR, is_train=False)

    print(f"Training Samples (Virtual 5x): {len(train_dataset)}") 
    print(f"Validation Samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Init Cross-Attention Model
    model = FusionModel(tab_input_dim=len(train_dataset.features))
    model.to(DEVICE)
    
    # Optimizer (Backbone is frozen, so optimizer only sees new layers)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PATIENCE_SCHEDULER
    )
    criterion = nn.MSELoss() 
    
    best_val_loss = float('inf')
    trigger_times = 0 
    
    print(f"Starting Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train() 
        train_loss = 0
        loop = tqdm(train_loader, leave=False)
        
        for imgs, tabs, targets, _ in loop:
            imgs, tabs, targets = imgs.to(DEVICE), tabs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs, tabs)
            loss = criterion(preds, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, tabs, targets, _ in val_loader:
                imgs, tabs, targets = imgs.to(DEVICE), tabs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs, tabs)
                loss = criterion(preds, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f" >>> Saved Best Model (Val Loss: {best_val_loss:.6f})")
        else:
            trigger_times += 1
            print(f" >>> No Improvement ({trigger_times}/{PATIENCE_EARLY_STOPPING})")
            
            if trigger_times >= PATIENCE_EARLY_STOPPING:
                print(" >>> Early Stopping Triggered! Stopping Training.")
                break

    # ==========================================
    # 4. FINAL EVALUATION
    # ==========================================
    print("\n--- Generating Final Evaluation CSV ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    all_ids = []
    all_pred_log_residuals = []
    
    with torch.no_grad():
        for imgs, tabs, _, batch_ids in tqdm(val_loader, desc="Evaluating"):
            imgs, tabs = imgs.to(DEVICE), tabs.to(DEVICE)
            preds = model(imgs, tabs)
            all_pred_log_residuals.extend(preds.cpu().numpy())
            all_ids.extend(batch_ids.numpy())
    
    results_df = pd.DataFrame({
        'id': all_ids,
        'Y_pred_log_residual': all_pred_log_residuals
    })
    
    final_eval_df = pd.merge(results_df, df[['id', 'price_pred_xgb', 'price']], on='id', how='left')
    final_eval_df['predicted_alpha'] = np.exp(final_eval_df['Y_pred_log_residual'])
    final_eval_df['total_model_y_pred'] = final_eval_df['price_pred_xgb'] * final_eval_df['predicted_alpha']
    
    mse = mean_squared_error(final_eval_df['price'], final_eval_df['total_model_y_pred'])
    r2 = r2_score(final_eval_df['price'], final_eval_df['total_model_y_pred'])
    
    print(f"\nFinal Test Split Results:")
    print(f"MSE: {mse:,.2f}")
    print(f"R^2: {r2:.5f}")
    
    output_cols = ['id', 'price_pred_xgb', 'Y_pred_log_residual', 'predicted_alpha', 'total_model_y_pred', 'price']
    final_eval_df[output_cols].to_csv(SUBMISSION_SAVE_PATH, index=False)
    print(f"Saved 'final_model_submissions.csv' to {SUBMISSION_SAVE_PATH} successfully.")

if __name__ == "__main__":
    main()