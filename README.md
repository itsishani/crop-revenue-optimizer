This project aims to assist farmers and agricultural planners in making informed decisions about crop selection and land allocation to maximize revenue. By analyzing a combination of soil and weather parameters—such as nitrogen, phosphorus, potassium (NPK) levels, temperature, humidity, pH, and rainfall—the system predicts the most suitable crops for a given farmland using a K-Nearest Neighbors (KNN) machine learning model.

Once crop suitability is determined, the project incorporates yield and market price data from multiple sources (local markets and agricultural research databases like ICRISAT) to compute the expected revenue for different crop combinations. It simulates all possible two-crop combinations and land distribution strategies using a step-based approach and identifies the most profitable and suitable combination based on the user’s total available land area.

The project uses Python as the programming language and relies on libraries like pandas, numpy, and scikit-learn. The system works in a command-line interface and outputs detailed results including:

Top 5 suitable crops with their suitability percentages.

All revenue-maximizing combinations for given land constraints.

The optimal revenue combination and the best combination that includes the top-suitable crops.

This decision-support system not only promotes smart farming through data-driven insights but also helps in sustainable agriculture by encouraging crop diversification and efficient land use.

