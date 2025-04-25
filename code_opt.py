import itertools

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load datasets
crop_df = pd.read_csv('data.csv')
yield_df = pd.read_csv('commodity_yield.csv')
icrisat_df = pd.read_csv('ICRISAT-District Level Data.csv')
price_df = pd.read_csv('pricedataset.csv')

def preprocess_data():
    """Extract crop price data from multiple sources"""
    # Get market prices
    market_prices = price_df.groupby('Commodity')['Modal Price'].mean().to_dict()
    
    # Get ICRISAT prices
    price_columns = [col for col in icrisat_df.columns if 'HARVEST PRICE' in col]
    icrisat_prices = {}
    for col in price_columns:
        crop_name = col.split(' HARVEST')[0].strip()
        icrisat_prices[crop_name] = icrisat_df[col].mean()
    
    # Merge prices with market priority
    return {**market_prices, **icrisat_prices}, price_columns

def get_suitable_crops(farm_conditions):
    """Get crops sorted by suitability using KNN"""
    X = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = crop_df['label']
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    probabilities = knn.predict_proba([farm_conditions])[0]
    return sorted(zip(knn.classes_, probabilities), key=lambda x: x[1], reverse=True)

def get_crop_data(suitable_crops, yield_df, price_data):
    """Get crop details with fallback handling"""
    crop_details = {}
    for crop, prob in suitable_crops:
        crop_upper = crop.upper()
        # Get yield with fallback
        try:
            yield_val = yield_df[yield_df['Commodity'].str.lower() == crop.lower()]['Yield_Quintals_Per_Hectare'].values[0]
        except:
            yield_val = 0  # Default to 0 if no yield data
            
        # Get price with fallback
        price = price_data.get(crop_upper, 0)
        
        crop_details[crop_upper] = {
            'yield': yield_val,
            'price': price,
            'probability': prob
        }
    return crop_details

def generate_combinations(crop_data, total_area, step_size, top_crops):
    """Generate all possible combinations with step-based allocations"""
    combinations = []
    
    # Always include top crops even with 0 values
    for crop in top_crops:
        if crop not in crop_data:
            crop_data[crop] = {'yield': 0, 'price': 0, 'probability': 0}
    
    crops = list(crop_data.keys())
    
    # Create all possible pairs including top crops
    for crop1, crop2 in itertools.permutations(crops, 2):
        # Generate area splits using user-defined step size
        for area1 in range(0, int(total_area)+1, step_size):
            area2 = total_area - area1
            if area2 < 0: continue
            
            revenue = (area1 * crop_data[crop1]['yield'] * crop_data[crop1]['price']) + \
                     (area2 * crop_data[crop2]['yield'] * crop_data[crop2]['price'])
            
            combinations.append({
                'crop1': crop1,
                'area1': area1,
                'crop2': crop2,
                'area2': area2,
                'revenue': revenue,
                'crop1_prob': crop_data[crop1]['probability'],
                'crop2_prob': crop_data[crop2]['probability']
            })
    
    # Sort by descending revenue
    return sorted(combinations, key=lambda x: x['revenue'], reverse=True)

def main():
    """Main recommendation workflow"""
    # Get farm parameters
    farm_conditions = [
        float(input(f"Enter {param}: ")) for param in 
        ['N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'Temperature (°C)', 
         'Humidity (%)', 'pH', 'Rainfall (mm)']
    ]
    total_area = float(input("Total land area (hectares): "))
    step_size = int(input("Allocation step size (e.g., 10 for 10% increments): "))
    
    # Preprocess data
    price_data, _ = preprocess_data()
    
    # Get suitability list
    suitable_crops = get_suitable_crops(farm_conditions)
    top_crops = [crop.upper() for crop, _ in suitable_crops[:2]]  # Top 2 suitable
    
    print("\nTop Suitable Crops:")
    for i, (crop, prob) in enumerate(suitable_crops[:5], 1):
        print(f"{i}. {crop} ({prob:.1%} suitability)")
    
    # Get crop details
    crop_data = get_crop_data(suitable_crops, yield_df, price_data)
    
    # Generate combinations
    combinations = generate_combinations(crop_data, total_area, step_size, top_crops)
    
    # Display results
    print("\nTested Combinations (Top 20):")
    for combo in combinations[:20]:
        print(f"{combo['crop1']} ({combo['area1']}ha) + {combo['crop2']} ({combo['area2']}ha)")
        print(f"  Revenue: ₹{combo['revenue']:,.2f}")
        print(f"  Suitability: {combo['crop1_prob']:.1%} + {combo['crop2_prob']:.1%}")
    
    # Show optimal combination
    optimal = combinations[0]
    print("\nOptimal Revenue Combination:")
    print(f"{optimal['crop1']} ({optimal['area1']}ha) + {optimal['crop2']} ({optimal['area2']}ha)")
    print(f"Maximum Revenue: ₹{optimal['revenue']:,.2f}")
    
    # Show best suitable combination
    suitable_combos = [c for c in combinations if c['crop1'] in top_crops or c['crop2'] in top_crops]
    if suitable_combos:
        best_suitable = max(suitable_combos, key=lambda x: x['revenue'])
        print("\nBest Suitable Crop Combination:")
        print(f"{best_suitable['crop1']} ({best_suitable['area1']}ha) + {best_suitable['crop2']} ({best_suitable['area2']}ha)")
        print(f"Revenue: ₹{best_suitable['revenue']:,.2f}")
 

if __name__ == "__main__":
    main()
