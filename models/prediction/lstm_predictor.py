import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import precision_score
from utils.progress_display import create_progress_display


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """LSTM model for fault prediction."""
    
    def __init__(self, input_size, lstm_units=64, lstm_layers=1, 
                 dropout_rate=0.2, dense_units=32):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(lstm_units, dense_units)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_output = lstm_out[:, -1, :]
        
        x = self.dropout1(last_output)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        
        return x


class LSTMPredictor:
    """LSTM predictor for fault prediction.
    
    This predictor uses raw data without any preprocessing (no temporal features,
    no CBSS sampling, no RF feature selection).
    Uses simple default parameters without optimization.
    """

    def __init__(self, random_state=42, sequence_length=10, show_progress=True):
        """Initialize LSTM predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.show_progress = show_progress
        self.progress = create_progress_display() if show_progress else None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        # Fixed parameters (no optimization, fast training)
        self.default_params = {
            'lstm_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'batch_size': 128,
            'epochs': 20,
            'dense_units': 16,
            'lstm_layers': 1
        }

    def _create_sequences(self, data, target):
        """Create sequences for LSTM training."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length + 1):
            seq = data[i:(i + self.sequence_length)]
            sequences.append(seq)
            targets.append(target[i + self.sequence_length - 1])
            
        return np.array(sequences), np.array(targets)

    def _build_model(self, input_size):
        """Build LSTM model architecture with default parameters."""
        model = LSTMModel(
            input_size=input_size,
            lstm_units=self.default_params['lstm_units'],
            lstm_layers=self.default_params['lstm_layers'],
            dropout_rate=self.default_params['dropout_rate'],
            dense_units=self.default_params['dense_units']
        ).to(self.device)
        
        return model

    def _train_model(self, model, train_loader, val_loader, learning_rate=0.001, 
                     epochs=50, patience=5):
        """Train the LSTM model."""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model

    def train(self, x_train, x_test, y_train, y_test):
        """Train the LSTM model with integrated progress display.
        
        This model uses raw data without any preprocessing.
        """
        if self.show_progress:
            with self.progress.model_training_status() as status:
                return self._train_with_progress(x_train, x_test, y_train, y_test, status)
        else:
            return self._train_without_progress(x_train, x_test, y_train, y_test)
    
    def _train_with_progress(self, x_train, x_test, y_train, y_test, status):
        """Internal training method with progress updates."""
        status.update("Initializing LSTM training...")
        
        # Reset random seed before training
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        status.update("Scaling raw features...")
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        status.update("Creating time sequences from raw data...")
        x_train_seq, y_train_seq = self._create_sequences(x_train_scaled, y_train.values)
        x_test_seq, y_test_seq = self._create_sequences(x_test_scaled, y_test.values)
        
        status.update("Building LSTM model...")
        input_size = x_train_seq.shape[2]
        self.model = self._build_model(input_size)
        
        status.update("Preparing data loaders...")
        train_dataset = TimeSeriesDataset(x_train_seq, y_train_seq)
        val_dataset = TimeSeriesDataset(x_test_seq, y_test_seq)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.default_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.default_params['batch_size'], 
            shuffle=False
        )
        
        status.update("Training LSTM model...")
        self.model = self._train_model(
            self.model, 
            train_loader, 
            val_loader,
            learning_rate=self.default_params['learning_rate'],
            epochs=self.default_params['epochs']
        )
        
        status.complete("LSTM model training completed successfully")
        return self.model
    
    def _train_without_progress(self, x_train, x_test, y_train, y_test):
        """Internal training method without progress updates."""
        # Reset random seed before training
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # Scale the features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        # Create sequences
        x_train_seq, y_train_seq = self._create_sequences(x_train_scaled, y_train.values)
        x_test_seq, y_test_seq = self._create_sequences(x_test_scaled, y_test.values)
        
        # Build model
        input_size = x_train_seq.shape[2]
        self.model = self._build_model(input_size)
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(x_train_seq, y_train_seq)
        val_dataset = TimeSeriesDataset(x_test_seq, y_test_seq)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.default_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.default_params['batch_size'], 
            shuffle=False
        )
        
        # Train the model
        self.model = self._train_model(
            self.model, 
            train_loader, 
            val_loader,
            learning_rate=self.default_params['learning_rate'],
            epochs=self.default_params['epochs']
        )
        
        return self.model

    def predict(self, x):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        x_scaled = self.scaler.transform(x)
        x_seq, _ = self._create_sequences(x_scaled, np.zeros(len(x_scaled)))
        
        if len(x_seq) == 0:
            # Handle case where sequence length is larger than input
            return np.zeros(len(x), dtype=int), np.zeros(len(x))
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(x_seq, np.zeros(len(x_seq)))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
        
        predictions = np.array(predictions)
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Pad predictions to match original input length
        if len(binary_predictions) < len(x):
            padding_length = len(x) - len(binary_predictions)
            binary_predictions = np.concatenate([
                np.zeros(padding_length, dtype=int), 
                binary_predictions
            ])
            predictions = np.concatenate([
                np.zeros(padding_length), 
                predictions
            ])
        
        return binary_predictions, predictions

    def save_model(self, model_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        if self.show_progress and self.progress:
            with self.progress.model_training_status() as status:
                status.update("Saving trained model...")
                self._save_model_files(model_path)
                status.complete("Model saved successfully")
        else:
            self._save_model_files(model_path)
    
    def _save_model_files(self, model_path):
        """Internal method to save model files."""
        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path.replace('.pkl', '.pth'))
        
        # Save scaler and other attributes
        model_data = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'random_state': self.random_state,
            'default_params': self.default_params,
            'input_size': self.model.lstm.input_size,
            'lstm_units': self.model.lstm.hidden_size,
            'lstm_layers': self.model.lstm.num_layers,
            'dropout_rate': self.model.dropout1.p,
            'dense_units': self.model.dense1.out_features
        }
        joblib.dump(model_data, model_path)

    def load_model(self, model_path):
        """Load a pre-trained model from disk."""
        # Load model data
        model_data = joblib.load(model_path)
        self.scaler = model_data['scaler']
        self.sequence_length = model_data['sequence_length']
        self.random_state = model_data['random_state']
        self.default_params = model_data['default_params']
        
        # Rebuild model
        self.model = LSTMModel(
            input_size=model_data['input_size'],
            lstm_units=model_data['lstm_units'],
            lstm_layers=model_data['lstm_layers'],
            dropout_rate=model_data['dropout_rate'],
            dense_units=model_data['dense_units']
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path.replace('.pkl', '.pth'), 
                                             map_location=self.device))
        
        return self.model