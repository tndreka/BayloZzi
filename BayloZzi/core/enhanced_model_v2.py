# core/enhanced_model_v2.py

import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import xgboost as xgb
from core.advanced_strategy import AdvancedForexStrategy
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

MODEL_PATH = "models/enhanced_model_v2.pkl"
SCALER_PATH = "models/scaler_v2.pkl"

class EnhancedForexPredictorV2:
    """
    Advanced Forex prediction model incorporating:
    1. Smart Money Concepts (ICT)
    2. Harmonic Pattern Recognition
    3. Supply/Demand Analysis
    4. Multiple timeframe analysis
    5. Ensemble learning with XGBoost
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        self.strategy = AdvancedForexStrategy()
        
    def create_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from multiple timeframes for better context.
        """
        df = df.copy()
        
        # Higher timeframe trends (simulate daily from hourly data)
        for period in [5, 10, 20]:  # 5-day, 10-day, 20-day equivalents
            df[f'htf_trend_{period}'] = df['close'].rolling(window=period*24).mean()
            df[f'htf_momentum_{period}'] = df['close'] / df['close'].shift(period*24) - 1
            
        # Lower timeframe patterns
        for period in [3, 5, 8]:
            df[f'ltf_volatility_{period}'] = df['close'].rolling(window=period).std()
            df[f'ltf_range_{period}'] = (df['high'].rolling(window=period).max() - 
                                         df['low'].rolling(window=period).min())
            
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sophisticated features including Smart Money concepts.
        """
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['price_acceleration'] = df['price_change'].diff()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        df['volatility_10'] = df['price_change'].rolling(window=10).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility_10'] / df['volatility_20']
        
        # Advanced moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_distance_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']
            
        # Moving average convergence
        df['ma_5_10_cross'] = np.where(df['ma_5'] > df['ma_10'], 1, -1)
        df['ma_10_20_cross'] = np.where(df['ma_10'] > df['ma_20'], 1, -1)
        df['ma_20_50_cross'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
        
        # Enhanced Bollinger Bands
        for period in [20, 50]:
            bb_mean = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}'] = bb_mean + (2 * bb_std)
            df[f'bb_lower_{period}'] = bb_mean - (2 * bb_std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / bb_mean
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            
        # RSI variations
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_oversold_{period}'] = (df[f'rsi_{period}'] < 30).astype(int)
            df[f'rsi_overbought_{period}'] = (df[f'rsi_{period}'] > 70).astype(int)
            
        # Stochastic RSI
        rsi = df['rsi_14']
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min)
        
        # MACD variations
        for fast, slow in [(12, 26), (5, 35)]:
            exp_fast = df['close'].ewm(span=fast).mean()
            exp_slow = df['close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = exp_fast - exp_slow
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=9).mean()
            df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
            
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        df['atr_normalized'] = df['atr_14'] / df['close']
        
        # Ichimoku Cloud components
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_10']
            df['price_volume_trend'] = (df['volume'] * df['price_change']).cumsum()
            df['on_balance_volume'] = (np.sign(df['price_change']) * df['volume']).cumsum()
        
        # Market microstructure features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_to_high'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['close_to_low'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & 
                          (df['upper_shadow'] < df['body_size'])).astype(int)
        
        # Price action patterns
        df['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                           (df['low'] > df['low'].shift(1))).astype(int)
        df['pin_bar'] = ((df['upper_shadow'] > 2 * df['body_size']) | 
                        (df['lower_shadow'] > 2 * df['body_size'])).astype(int)
        
        # Support and resistance levels
        df['resistance_distance'] = df['high'].rolling(window=20).max() - df['close']
        df['support_distance'] = df['close'] - df['low'].rolling(window=20).min()
        
        # Momentum oscillators
        df['williams_r'] = ((df['high'].rolling(window=14).max() - df['close']) / 
                           (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * -100
        
        # Rate of change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                  df['close'].shift(period)) * 100
            
        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df.get('volume', 1)
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Fractal dimensions (simplified)
        df['fractal_dim'] = self.calculate_fractal_dimension(df['close'].values)
        
        # Add multi-timeframe features
        df = self.create_multi_timeframe_features(df)
        
        return df.fillna(method='ffill').fillna(0)
    
    def calculate_fractal_dimension(self, prices, window=30):
        """
        Calculate fractal dimension of price series.
        """
        if len(prices) < window:
            return pd.Series([1.5] * len(prices))
            
        fractal_dims = []
        for i in range(len(prices)):
            if i < window:
                fractal_dims.append(1.5)
            else:
                # Simplified fractal dimension calculation
                price_window = prices[i-window:i]
                log_n = np.log(window)
                log_r = np.log(np.max(price_window) - np.min(price_window))
                fd = 1 + (log_n / log_r) if log_r != 0 else 1.5
                fractal_dims.append(np.clip(fd, 1, 2))
                
        return pd.Series(fractal_dims)
    
    def create_smart_money_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on Smart Money concepts.
        """
        df = df.copy()
        
        # Detect market structure
        market_struct = self.strategy.detect_market_structure(df)
        df['market_trend'] = 1 if market_struct['trend'] == 'bullish' else (-1 if market_struct['trend'] == 'bearish' else 0)
        
        # Order blocks
        order_blocks = self.strategy.identify_order_blocks(df)
        df['near_bullish_ob'] = 0
        df['near_bearish_ob'] = 0
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            for ob in order_blocks:
                if ob['type'] == 'bullish' and ob['low'] <= current_price <= ob['high']:
                    df.iloc[i, df.columns.get_loc('near_bullish_ob')] = ob['strength']
                elif ob['type'] == 'bearish' and ob['low'] <= current_price <= ob['high']:
                    df.iloc[i, df.columns.get_loc('near_bearish_ob')] = ob['strength']
                    
        # Fair Value Gaps
        fvgs = self.strategy.detect_fair_value_gaps(df)
        df['in_bullish_fvg'] = 0
        df['in_bearish_fvg'] = 0
        
        for fvg in fvgs:
            fvg_idx = df.index.get_loc(fvg['date'])
            if fvg['type'] == 'bullish':
                df.iloc[fvg_idx:min(fvg_idx+5, len(df)), df.columns.get_loc('in_bullish_fvg')] = 1
            else:
                df.iloc[fvg_idx:min(fvg_idx+5, len(df)), df.columns.get_loc('in_bearish_fvg')] = 1
                
        # Liquidity zones
        liquidity = self.strategy.identify_liquidity_zones(df)
        df['near_buy_liquidity'] = 0
        df['near_sell_liquidity'] = 0
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            # Check if near buy-side liquidity
            recent_buy_liq = liquidity['buy_side'].tail(5)
            if not recent_buy_liq.empty:
                min_distance = abs(recent_buy_liq['high'] - current_price).min()
                if min_distance < df['atr_14'].iloc[i] * 2:
                    df.iloc[i, df.columns.get_loc('near_buy_liquidity')] = 1
                    
            # Check if near sell-side liquidity
            recent_sell_liq = liquidity['sell_side'].tail(5)
            if not recent_sell_liq.empty:
                min_distance = abs(recent_sell_liq['low'] - current_price).min()
                if min_distance < df['atr_14'].iloc[i] * 2:
                    df.iloc[i, df.columns.get_loc('near_sell_liquidity')] = 1
                    
        return df
    
    def create_enhanced_labels(self, df: pd.DataFrame, lookahead=5, 
                             min_profit_threshold=0.002, use_dynamic_targets=True):
        """
        Create sophisticated labels considering risk-reward and market conditions.
        """
        df = df.copy()
        
        # Calculate future price movements
        future_returns = []
        future_max_profit = []
        future_max_loss = []
        
        for i in range(len(df)):
            if i + lookahead < len(df):
                current_price = df.iloc[i]['close']
                future_window = df.iloc[i+1:i+lookahead+1]
                
                # Maximum favorable and adverse excursions
                max_high = future_window['high'].max()
                min_low = future_window['low'].min()
                
                max_profit = (max_high - current_price) / current_price
                max_loss = (current_price - min_low) / current_price
                
                # Final return
                final_return = (future_window.iloc[-1]['close'] - current_price) / current_price
                
                future_returns.append(final_return)
                future_max_profit.append(max_profit)
                future_max_loss.append(max_loss)
            else:
                future_returns.append(0)
                future_max_profit.append(0)
                future_max_loss.append(0)
                
        df['future_return'] = future_returns
        df['future_max_profit'] = future_max_profit
        df['future_max_loss'] = future_max_loss
        
        # Dynamic profit threshold based on volatility
        if use_dynamic_targets:
            volatility = df['atr_normalized'].rolling(window=20).mean()
            dynamic_threshold = np.maximum(min_profit_threshold, volatility * 1.5)
        else:
            dynamic_threshold = min_profit_threshold
            
        # Create signals based on risk-reward ratio
        signals = []
        for i in range(len(df)):
            if i >= len(dynamic_threshold):
                threshold = min_profit_threshold
            else:
                threshold = dynamic_threshold.iloc[i] if use_dynamic_targets else min_profit_threshold
                
            # Long signal: Good profit potential with limited downside
            if (df.iloc[i]['future_max_profit'] > threshold * 2 and 
                df.iloc[i]['future_max_loss'] < threshold and
                df.iloc[i]['future_return'] > threshold):
                signals.append(1)
            # Short signal: Good profit potential on downside
            elif (df.iloc[i]['future_max_loss'] > threshold * 2 and 
                  df.iloc[i]['future_max_profit'] < threshold and
                  df.iloc[i]['future_return'] < -threshold):
                signals.append(-1)
            else:
                signals.append(0)
                
        df['Signal'] = signals
        
        # Binary classification (1: Buy, 0: Not Buy)
        df['Binary_Signal'] = (df['Signal'] == 1).astype(int)
        
        # Add signal quality metric
        df['signal_quality'] = np.where(
            df['Signal'] != 0,
            abs(df['future_return']) / (df['future_max_loss'] + 0.0001),
            0
        )
        
        return df
    
    def train_ensemble_model(self, df: pd.DataFrame, target_col='Binary_Signal', 
                           test_size=0.2, use_time_series_cv=True):
        """
        Train advanced ensemble model with multiple algorithms.
        """
        logger.info("Starting enhanced model V2 training...")
        
        # Feature engineering
        df_enhanced = self.create_advanced_features(df)
        df_enhanced = self.create_smart_money_features(df_enhanced)
        df_enhanced = self.create_enhanced_labels(df_enhanced)
        
        # Remove any remaining NaN values
        df_enhanced = df_enhanced.dropna()
        
        # Prepare features
        feature_cols = [col for col in df_enhanced.columns 
                       if col not in [target_col, 'Signal', 'Binary_Signal', 'future_return', 
                                     'future_max_profit', 'future_max_loss', 'signal_quality',
                                     'Date', 'open', 'high', 'low', 'close', 'volume'] 
                       and not col.startswith('Unnamed')]
        
        X = df_enhanced[feature_cols].fillna(0)
        y = df_enhanced[target_col].fillna(0)
        
        # Store feature names
        self.feature_names = feature_cols
        logger.info(f"Total features: {len(feature_cols)}")
        
        # Split data (time series aware)
        if use_time_series_cv:
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection - select top features
        k_features = min(50, len(feature_cols))
        self.feature_selector = SelectKBest(mutual_info_classif, k=k_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_features = np.array(self.feature_names)[self.feature_selector.get_support()]
        logger.info(f"Selected {len(selected_features)} features")
        
        # Define advanced models
        models_to_test = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        
        # Train and evaluate individual models
        best_score = 0
        best_model_name = None
        model_scores = {}
        
        for name, model in models_to_test.items():
            logger.info(f"Training {name}...")
            
            try:
                # Time series cross-validation
                if use_time_series_cv and len(y_train) > 100:
                    tscv = TimeSeriesSplit(n_splits=3)
                    cv_scores = cross_val_score(model, X_train_selected, y_train, 
                                              cv=tscv, scoring='f1')
                    mean_cv_score = cv_scores.mean()
                else:
                    mean_cv_score = 0.5
                
                # Train on full training set
                model.fit(X_train_selected, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_selected)
                y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted', zero_division=0
                )
                
                # Custom win rate calculation
                win_signals = y_test == 1
                if win_signals.sum() > 0:
                    win_rate = accuracy_score(y_test[win_signals], y_pred[win_signals])
                else:
                    win_rate = 0
                
                logger.info(f"{name} Results:")
                logger.info(f"  CV Score: {mean_cv_score:.4f}")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                logger.info(f"  Win Rate: {win_rate:.4f}")
                
                model_scores[name] = {
                    'model': model,
                    'cv_score': mean_cv_score,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'win_rate': win_rate
                }
                
                # Select based on combined score
                combined_score = 0.3 * f1 + 0.3 * win_rate + 0.2 * precision + 0.2 * recall
                if combined_score > best_score:
                    best_score = combined_score
                    best_model_name = name
                    
            except Exception as e:
                logger.warning(f"Error training {name}: {e}")
                continue
        
        # Create ensemble of top models
        if len(model_scores) >= 3:
            # Select top 3 models
            sorted_models = sorted(model_scores.items(), 
                                 key=lambda x: x[1]['f1'], reverse=True)[:3]
            
            ensemble_models = [(name, model['model']) for name, model in sorted_models]
            
            self.best_model = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'
            )
            self.best_model.fit(X_train_selected, y_train)
            
            # Evaluate ensemble
            y_pred_ensemble = self.best_model.predict(X_test_selected)
            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            
            logger.info(f"Ensemble model created with: {[m[0] for m in ensemble_models]}")
            logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        else:
            # Use best individual model
            self.best_model = model_scores[best_model_name]['model']
            
        logger.info(f"Best model selected: {best_model_name}")
        
        # Feature importance analysis
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif best_model_name == 'xgboost' and hasattr(model_scores[best_model_name]['model'], 'feature_importances_'):
            importances = model_scores[best_model_name]['model'].feature_importances_
        else:
            importances = None
            
        if importances is not None:
            feature_importance = list(zip(selected_features, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            logger.info("Top 15 Most Important Features:")
            for feat, imp in feature_importance[:15]:
                logger.info(f"  {feat}: {imp:.4f}")
        
        # Store results
        self.models = model_scores
        
        return {
            'best_model': best_model_name,
            'model_scores': model_scores,
            'win_rate': model_scores[best_model_name]['win_rate'],
            'f1_score': model_scores[best_model_name]['f1'],
            'X_test': X_test_selected,
            'y_test': y_test,
            'selected_features': selected_features
        }
    
    def predict_with_confidence(self, features):
        """
        Make prediction with confidence score and additional insights.
        """
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure features match training format
        if isinstance(features, pd.DataFrame):
            features_df = features[self.feature_names].fillna(0)
        else:
            features_df = pd.DataFrame([features], columns=self.feature_names).fillna(0)
        
        # Scale and select features
        features_scaled = self.scaler.transform(features_df.values)
        features_selected = self.feature_selector.transform(features_scaled)
        
        # Prediction
        prediction = self.best_model.predict(features_selected)[0]
        
        # Confidence (probability of predicted class)
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(features_selected)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 0.5
        
        # Get advanced strategy signals for additional confirmation
        # This would require the full dataframe context in practice
        
        return prediction, confidence
    
    def save_models(self):
        """Save all trained components."""
        model_data = {
            'models': self.models,
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, MODEL_PATH)
        logger.info(f"Enhanced model V2 saved to {MODEL_PATH}")
    
    def load_models(self):
        """Load all trained components."""
        model_data = joblib.load(MODEL_PATH)
        
        self.models = model_data['models']
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Enhanced model V2 loaded from {MODEL_PATH}")


def train_enhanced_forex_model_v2(df):
    """
    Main function to train the enhanced forex prediction model V2.
    Target: 70-80% win rate with advanced strategies.
    """
    predictor = EnhancedForexPredictorV2()
    results = predictor.train_ensemble_model(df)
    predictor.save_models()
    
    logger.info("=" * 60)
    logger.info("ENHANCED FOREX MODEL V2 TRAINING COMPLETE")
    logger.info(f"Best Model: {results['best_model']}")
    logger.info(f"Expected Win Rate: {results['win_rate']:.2%}")
    logger.info(f"F1-Score: {results['f1_score']:.4f}")
    logger.info("=" * 60)
    
    return predictor, results