import os
import joblib
import numpy as np
import warnings

class LiveAuctionAgent:
    def __init__(self):
        # Initial bankroll is $500,000
        self.bankroll = 500000.0
        self.predicted_value = 0.0
        
        # Custom parameters for our deterministic math
        self.target_margin = 0.13 
        self.bid_round = 0 
        self.current_condition = 3.0 
        
        # Get the path to where this script is saved
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Load the tuned LightGBM model and the dictionary of LabelEncoders
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(os.path.join(base_path, "model_Anjali_Pogulwad.pkl"))
            self.encoders = joblib.load(os.path.join(base_path, "encoders_Anjali_Pogulwad.pkl"))

    def analyze_item(self, item_features: dict):
        # Reset the internal bidding round counter for this new car
        self.bid_round = 0 
        
        # 1. Define columns matching the EXACT 11-feature order the model expects
        numeric_cols = ['year', 'condition', 'odometer']
        categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
        
        features = []
        
        # 2. Extract base numerics safely
        for col in numeric_cols:
            val = item_features.get(col, 0)
            if val is None or str(val).lower() == 'nan':
                val = 0
            features.append(float(val))
            
        # Save condition score for our bidding risk multiplier
        self.current_condition = item_features.get('condition', 3.0)
        
        # 3. Process categorical columns using our joblib dictionary
        for col in categorical_cols:
            text_value = str(item_features.get(col, "")).lower().strip()
            encoder = self.encoders[col]
            
            # Safe transform: If we recognize the word, encode it. Otherwise, use 0 to prevent a crash.
            if text_value in encoder.classes_:
                encoded_val = int(encoder.transform([text_value])[0])
                features.append(encoded_val)
            else:
                features.append(0)
        
        # 4. Predict the price using the tuned 11-feature LightGBM model
        input_data = np.array([features])       
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self.model.predict(input_data)
            self.predicted_value = float(prediction[0])

    def place_bid(self, current_highest_bid: float) -> float:
        # Increment our internal round tracker
        self.bid_round += 1 
        
        # Apply Risk Multiplier (If condition is excellent, we bid 5% more aggressively)
        m_service = 1.05 if self.current_condition >= 4.0 else 1.00
        
        # Calculate Maximum Willing Bid
        adjusted_value = self.predicted_value * m_service
        my_max_price = min(adjusted_value * (1 - self.target_margin), self.bankroll)
        
        # Drop out if the current bid is already too high or bankroll is dead
        if current_highest_bid >= my_max_price or self.bankroll <= 0:
            return 0.0
            
        # Deterministic Sequence Math
        base_increment = 100.0
        distance_to_max = my_max_price - current_highest_bid
        k = 0.6 
        
        # Calculate dynamic increment using exponential decay
        dynamic_increment = distance_to_max / np.exp(k * self.bid_round)
        
        # Finalize next bid
        my_next_bid = current_highest_bid + base_increment + dynamic_increment
        my_next_bid = min(my_next_bid, my_max_price) 
        
        return float(int(round(my_next_bid)))

    def auction_result(self, won, winning_bid, actual_price, current_bankroll):
        # The evaluator updates our official bankroll after the round finishes
        self.bankroll = float(current_bankroll)
        self.bid_round = 0
