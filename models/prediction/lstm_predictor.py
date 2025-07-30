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
from .parameter_optimizer import ParameterOptimizer


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
    """LSTM based predictor with hyperparameter optimization for time series fault prediction."""

    def __init__(self, random_state=42, sequence_length=10):
        """Initialize LSTM predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        self.initial_param_space = {
            'lstm_units': [32, 64],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64],
            'epochs': [20, 50],
            'dense_units': [16, 32],
            'lstm_layers': [1, 2]
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

    def _build_model(self, input_size, lstm_units=64, dropout_rate=0.2, 
                     dense_units=32, lstm_layers=1, **kwargs):
        """Build LSTM model architecture."""
        model = LSTMModel(
            input_size=input_size,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            dropout_rate=dropout_rate,
            dense_units=dense_units
        ).to(self.device)
        
        return model

    def _train_model(self, model, train_loader, val_loader, learning_rate=0.001, 
                     epochs=100, patience=10):
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

    def train(self, x_train, x_test, y_train, y_test, parameter_optimization=False):
        """Train the LSTM model."""
        # Reset random seed before training
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        # Scale the features
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        # Create sequences
        x_train_seq, y_train_seq = self._create_sequences(x_train_scaled, y_train.values)
        x_test_seq, y_test_seq = self._create_sequences(x_test_scaled, y_test.values)
        
        if parameter_optimization:
            best_params = self._optimize_parameters(x_train_seq, y_train_seq, 
                                                   x_test_seq, y_test_seq)
        else:
            best_params = self._simple_parameter_search(x_train_seq, y_train_seq)

        # Build and train final model
        input_size = x_train_seq.shape[2]
        self.model = self._build_model(input_size, **best_params)
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(x_train_seq, y_train_seq)
        val_dataset = TimeSeriesDataset(x_test_seq, y_test_seq)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=best_params.get('batch_size', 32), 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=best_params.get('batch_size', 32), 
            shuffle=False
        )
        
        # Train the model
        self.model = self._train_model(
            self.model, 
            train_loader, 
            val_loader,
            learning_rate=best_params.get('learning_rate', 0.001),
            epochs=best_params.get('epochs', 100)
        )
        
        return self.model

    def _simple_parameter_search(self, x_train, y_train, n_iter=3):
        """Perform simple random parameter search."""
        param_sampler = ParameterSampler(self.initial_param_space, 
                                       n_iter=n_iter, 
                                       random_state=self.random_state)
        
        best_score = 0
        best_params = None
        
        for params in param_sampler:
            try:
                # Create validation split
                val_size = int(0.2 * len(x_train))
                x_val = x_train[-val_size:]
                y_val = y_train[-val_size:]
                x_train_fold = x_train[:-val_size]
                y_train_fold = y_train[:-val_size]
                
                # Build model
                input_size = x_train.shape[2]
                model = self._build_model(input_size, **params)
                
                # Create data loaders
                train_dataset = TimeSeriesDataset(x_train_fold, y_train_fold)
                val_dataset = TimeSeriesDataset(x_val, y_val)
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=params.get('batch_size', 32), 
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=params.get('batch_size', 32), 
                    shuffle=False
                )
                
                # Train model with reduced epochs for search
                model = self._train_model(
                    model, 
                    train_loader, 
                    val_loader,
                    learning_rate=params.get('learning_rate', 0.001),
                    epochs=min(params.get('epochs', 20), 15),
                    patience=3
                )
                
                # Evaluate
                model.eval()
                y_pred_list = []
                with torch.no_grad():
                    for batch_x, _ in val_loader:
                        batch_x = batch_x.to(self.device)
                        outputs = model(batch_x).squeeze()
                        y_pred_list.extend((outputs > 0.5).cpu().numpy())
                
                score = precision_score(y_val, y_pred_list, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception:
                continue
        
        # Return default params if no valid params found
        if best_params is None:
            best_params = {
                'lstm_units': 32,
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 20,
                'dense_units': 16,
                'lstm_layers': 1
            }
        
        return best_params

    def _optimize_parameters(self, x_train, y_train, x_test, y_test):
        """Advanced parameter optimization using validation set."""
        return self._simple_parameter_search(x_train, y_train, n_iter=5)

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

        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path.replace('.pkl', '.pth'))
        
        # Save scaler and other attributes
        model_data = {
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'random_state': self.random_state,
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

    def transfer_learning(self, target_x_train, target_y_train):
        """Apply transfer learning to adapt model to new data."""
        if self.model is None:
            raise ValueError("No source model available for transfer learning")

        # Prepare target data
        target_x_scaled = self.scaler.transform(target_x_train)
        target_x_seq, target_y_seq = self._create_sequences(
            target_x_scaled, target_y_train.values
        )

        # Freeze early layers
        for param in self.model.lstm.parameters():
            param.requires_grad = False

        # Create data loader
        target_dataset = TimeSeriesDataset(target_x_seq, target_y_seq)
        target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

        # Fine-tune with lower learning rate
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

        # Fine-tune the model
        self.model.train()
        for epoch in range(50):
            for batch_x, batch_y in target_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self.model